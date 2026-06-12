"""Smoke tests for the spatiotemporal BS-JEPA path."""

import pytest
import torch
from torch.utils.data import DataLoader

from brain_jepa.data.atlas import load_atlas
from brain_jepa.data.windowing import STSample, make_grid_ids, window_bold
from brain_jepa.evaluation import pooled_embeddings, representation_health
from brain_jepa.masking import SpatioTemporalMaskCollator
from brain_jepa.models import build_st_bsjepa
from brain_jepa.training.ema import EMAUpdater
from brain_jepa.training.losses import jepa_loss

ATLAS_CSV = "data/atlas/glasser379_to_rsn12.csv"
N, T, L, STRIDE = 379, 160, 40, 20
D = 64
B = 4


@pytest.fixture(scope="module")
def atlas():
    return load_atlas(ATLAS_CSV)


@pytest.fixture(scope="module")
def samples(atlas):
    out = []
    for i in range(B):
        x = window_bold(torch.randn(N, T), L, STRIDE)
        p = x.shape[0]
        rsn, tim = make_grid_ids(p, atlas.num_rsns)
        out.append(STSample(x_win=x, rsn_ids=rsn, time_ids=tim, num_windows=p,
                            subject_id=f"s{i}", age=20.0 + i, gender=i % 2))
    return out


@pytest.fixture(scope="module")
def model(atlas):
    return build_st_bsjepa(
        atlas=atlas, window_length=L, embed_dim=D, feature_mode="conv1d",
        time_max_windows=32, st_encoder_depth=2, st_encoder_heads=4,
        st_predictor_dim=32, st_predictor_depth=2, st_predictor_heads=4,
    )


# ------------------------------------------------------------------
# Windowing
# ------------------------------------------------------------------

def test_window_shape():
    x = window_bold(torch.randn(N, T), L, STRIDE)
    expected_p = (T - L) // STRIDE + 1
    assert x.shape == (expected_p, N, L)


def test_window_pad_last_adds_window():
    base = window_bold(torch.randn(N, 150), L, STRIDE, drop_last=True)
    padded = window_bold(torch.randn(N, 150), L, STRIDE, drop_last=False, pad_last=True)
    assert padded.shape[0] >= base.shape[0]
    assert padded.shape[1:] == (N, L)


# ------------------------------------------------------------------
# Masking
# ------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["block", "spatial", "temporal"])
def test_mask_shape_and_no_overlap(atlas, samples, mode):
    coll = SpatioTemporalMaskCollator(num_rsns=atlas.num_rsns, mode=mode, seed=0)
    batch, masks = coll(samples)
    p = samples[0].num_windows
    k = atlas.num_rsns
    assert masks.context_mask.shape == (B, p, k)
    assert masks.target_mask.shape == (B, p, k)
    # mutually exclusive and jointly cover the grid
    assert not (masks.context_mask & masks.target_mask).any()
    assert (masks.context_mask | masks.target_mask).all()
    # every subject has at least one target and one context token
    for i in range(B):
        assert masks.target_mask[i].any()
        assert masks.context_mask[i].any()


def test_mask_target_ids_align(atlas, samples):
    coll = SpatioTemporalMaskCollator(num_rsns=atlas.num_rsns, mode="block", seed=1)
    _, masks = coll(samples)
    for i in range(B):
        assert masks.target_rsn_ids[i].numel() == int(masks.target_mask[i].sum())
        assert masks.target_time_ids[i].numel() == int(masks.target_mask[i].sum())


# ------------------------------------------------------------------
# Model forward
# ------------------------------------------------------------------

def test_st_forward_shapes(model, atlas, samples):
    coll = SpatioTemporalMaskCollator(num_rsns=atlas.num_rsns, mode="block", seed=2)
    batch, masks = coll(samples)
    z_hat, z_tgt, ctx = model(batch, masks)
    n_tgt = sum(int(masks.target_mask[i].sum()) for i in range(B))
    n_ctx = sum(int(masks.context_mask[i].sum()) for i in range(B))
    assert z_hat.shape == (n_tgt, D)
    assert z_tgt.shape == (n_tgt, D)
    assert ctx.shape == (n_ctx, D)
    assert not z_tgt.requires_grad


def test_st_encode_returns_token_grid(model, atlas, samples):
    z = model.encode(samples[0])
    assert z.shape == (samples[0].num_windows * atlas.num_rsns, D)


def test_st_training_step(model, atlas, samples):
    coll = SpatioTemporalMaskCollator(num_rsns=atlas.num_rsns, mode="block", seed=3)
    batch, masks = coll(samples)
    opt = torch.optim.AdamW(
        list(model.tokenizer.parameters())
        + list(model.context_encoder.parameters())
        + list(model.predictor.parameters()),
        lr=1e-3,
    )
    ema = EMAUpdater(tau_start=0.99, tau_end=1.0, total_steps=1)

    opt.zero_grad()
    z_hat, z_tgt, ctx = model(batch, masks)
    loss, _ = jepa_loss(z_hat, z_tgt, ctx)
    loss.backward()
    grads = [p for p in model.context_encoder.parameters() if p.grad is not None]
    assert any(g.grad.abs().sum() > 0 for g in grads)
    # target encoder must never receive gradient
    assert all(p.grad is None for p in model.target_encoder.parameters())
    opt.step()
    ema.step(model.context_encoder, model.target_encoder)


# ------------------------------------------------------------------
# Diagnostics accept ST embeddings
# ------------------------------------------------------------------

def test_diagnostics_accept_st(model, samples):
    loader = DataLoader(samples, batch_size=2, collate_fn=list)
    pooled = pooled_embeddings(model, loader, torch.device("cpu"))
    assert pooled["mean"].shape[0] == B
    assert "rsn" in pooled
    assert "node_embedding_std_mean" in pooled
    assert "within_subject_node_std_mean" in pooled

    health = representation_health(pooled["mean"])
    for key in ("embedding_std_mean", "embedding_std_min", "mean_pairwise_cosine",
                "centered_mean_pairwise_cosine", "effective_rank"):
        assert key in health
