"""Tests for model forward-pass shapes, EMA update math, and loss."""

import copy

import pytest
import torch

from brain_jepa.data.atlas import load_atlas
from brain_jepa.data.dataset import SyntheticBrainDataset
from brain_jepa.masking import SubnetworkMaskCollator
from brain_jepa.models import build_bsjepa
from brain_jepa.models.encoders import GCNEncoder, GraphTransformerEncoder
from brain_jepa.models.pooling import AttentionPooling, MeanPooling
from brain_jepa.models.predictor import SubnetworkPredictor
from brain_jepa.training.ema import EMAUpdater
from brain_jepa.training.losses import jepa_loss

ATLAS_CSV = "data/atlas/glasser379_to_rsn12.csv"
N = 379
F = 32
D = 64
K = 12


@pytest.fixture(scope="module")
def atlas():
    return load_atlas(ATLAS_CSV)


@pytest.fixture(scope="module")
def dataset(atlas):
    return SyntheticBrainDataset(atlas=atlas, num_subjects=4, feature_dim=F, seed=2)


@pytest.fixture(scope="module")
def model(atlas):
    return build_bsjepa(
        atlas=atlas,
        encoder_type="gcn",
        in_channels=F,
        encoder_hidden=64,
        encoder_out=D,
        encoder_layers=2,
        predictor_dim=32,
        predictor_depth=2,
        predictor_heads=2,
    )


@pytest.fixture(scope="module")
def batch_and_masks(dataset, atlas):
    collator = SubnetworkMaskCollator(num_rsns=K, num_targets=1)
    raw = [dataset[i] for i in range(4)]
    return collator(raw)


# ------------------------------------------------------------------
# Encoder shapes
# ------------------------------------------------------------------

def test_gcn_output_shape(dataset):
    data = dataset[0]
    enc = GCNEncoder(in_channels=F, hidden_channels=64, out_channels=D, num_layers=2)
    enc.eval()
    with torch.no_grad():
        out = enc(data)
    assert out.shape == (N, D)


def test_graph_transformer_output_shape(dataset):
    data = dataset[0]
    enc = GraphTransformerEncoder(
        in_channels=F, hidden_channels=64, out_channels=D, num_layers=2, num_heads=2
    )
    enc.eval()
    with torch.no_grad():
        out = enc(data)
    assert out.shape == (N, D)


# ------------------------------------------------------------------
# Pooling shapes
# ------------------------------------------------------------------

@pytest.mark.parametrize("PoolCls", [MeanPooling, lambda: AttentionPooling(D)])
def test_pooling_shape(atlas, PoolCls):
    pool = PoolCls() if callable(PoolCls) else PoolCls
    node_emb = torch.randn(N, D)
    target_rsns = torch.tensor([0, 1, 2])
    tokens = pool(node_emb, atlas.rsn_ids, target_rsns)
    assert tokens.shape == (3, D)


# ------------------------------------------------------------------
# Predictor shapes
# ------------------------------------------------------------------

def test_predictor_output_shape():
    N_ctx, N_tgt, D_enc, P = 340, 39, D, 32
    pred = SubnetworkPredictor(
        encoder_dim=D_enc, predictor_dim=P, num_rsns=K, num_regions=N, depth=2, num_heads=2
    )
    pred.eval()
    context_tokens = torch.randn(N_ctx, D_enc)
    context_rsn_ids = torch.randint(0, K, (N_ctx,))
    context_region_ids = torch.arange(N_ctx)
    target_rsn_ids = torch.randint(0, K, (N_tgt,))
    target_region_ids = torch.arange(N_ctx, N_ctx + N_tgt)
    with torch.no_grad():
        out = pred(
            context_tokens, context_rsn_ids, context_region_ids,
            target_rsn_ids, target_region_ids,
        )
    assert out.shape == (N_tgt, D_enc)


def test_predictor_distinct_outputs_within_rsn():
    """Mask queries within one RSN must yield distinct node-level predictions."""
    N_ctx, N_tgt, D_enc, P = 340, 39, D, 32
    pred = SubnetworkPredictor(
        encoder_dim=D_enc, predictor_dim=P, num_rsns=K, num_regions=N, depth=2, num_heads=2
    )
    pred.eval()
    context_tokens = torch.randn(N_ctx, D_enc)
    context_rsn_ids = torch.randint(0, K, (N_ctx,))
    context_region_ids = torch.arange(N_ctx)
    # All target nodes belong to the same RSN
    target_rsn_ids = torch.zeros(N_tgt, dtype=torch.long)
    target_region_ids = torch.arange(N_ctx, N_ctx + N_tgt)
    with torch.no_grad():
        out = pred(
            context_tokens, context_rsn_ids, context_region_ids,
            target_rsn_ids, target_region_ids,
        )
    # Predictions for different nodes of the same RSN must not be identical
    assert not torch.allclose(out[0], out[1], atol=1e-5)


# ------------------------------------------------------------------
# Full model forward pass
# ------------------------------------------------------------------

def test_bsjepa_forward_shapes(model, batch_and_masks):
    batch, masks = batch_and_masks
    model.eval()
    with torch.no_grad():
        z_hat, z_tgt, ctx_embs = model(batch, masks)
    n_tgt_total = sum(int(m.sum()) for m in masks.target_node_masks)
    n_ctx_total = sum(int(m.sum()) for m in masks.context_node_masks)
    assert z_hat.shape == (n_tgt_total, D)
    assert z_tgt.shape == (n_tgt_total, D)
    assert ctx_embs.shape == (n_ctx_total, D)
    assert not z_tgt.requires_grad


def test_bsjepa_no_grad_on_target(model):
    for p in model.target_encoder.parameters():
        assert not p.requires_grad


# ------------------------------------------------------------------
# EMA update
# ------------------------------------------------------------------

def test_ema_update_math():
    """After one EMA step the target param should equal tau*t0 + (1-tau)*c0."""
    tau = 0.996
    ctx = torch.nn.Linear(4, 4)
    tgt = copy.deepcopy(ctx)

    # Perturb context params
    with torch.no_grad():
        for p in ctx.parameters():
            p.add_(torch.randn_like(p))

    ctx_before = {n: p.clone() for n, p in ctx.named_parameters()}
    tgt_before = {n: p.clone() for n, p in tgt.named_parameters()}

    updater = EMAUpdater(tau_start=tau, tau_end=tau, total_steps=1)
    updater.step(ctx, tgt)

    for (name, tgt_after) in tgt.named_parameters():
        expected = tau * tgt_before[name] + (1 - tau) * ctx_before[name]
        assert torch.allclose(tgt_after, expected, atol=1e-6), f"EMA mismatch at {name}"


def test_ema_schedule_monotone():
    """EMA momentum should increase monotonically from tau_start to tau_end."""
    updater = EMAUpdater(tau_start=0.996, tau_end=1.0, total_steps=100)
    ctx = torch.nn.Linear(2, 2)
    tgt = copy.deepcopy(ctx)
    taus = [updater.step(ctx, tgt) for _ in range(100)]
    for i in range(len(taus) - 1):
        assert taus[i] <= taus[i + 1] + 1e-9, "EMA schedule not monotone"


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------

def test_jepa_loss_perfect_prediction():
    """With z_hat == z_tgt and ample variance, the sim loss should be ~0."""
    x = torch.randn(64, D)
    total, metrics = jepa_loss(x, x.detach(), x.detach())
    assert metrics["sim"].item() == pytest.approx(0.0, abs=1e-5)


def test_jepa_loss_nonnegative():
    z_hat = torch.randn(64, D)
    z_tgt = torch.randn(64, D)
    ctx = torch.randn(128, D)
    total, metrics = jepa_loss(z_hat, z_tgt.detach(), ctx)
    assert total.item() >= 0.0
    assert set(metrics) == {"sim", "ctx_var", "hat_var", "ctx_cov", "tgt_std"}


def test_jepa_loss_covariance_penalises_correlated_dims():
    """Embeddings with perfectly correlated dimensions should pay a cov penalty."""
    base = torch.randn(128, 1)
    correlated = base.expand(128, D)  # rank-1: all dims identical
    decorrelated = torch.randn(128, D)
    z = torch.randn(32, D)
    _, m_corr = jepa_loss(z, z.detach(), correlated)
    _, m_decorr = jepa_loss(z, z.detach(), decorrelated)
    assert m_corr["ctx_cov"].item() > m_decorr["ctx_cov"].item()


def test_jepa_loss_backward(model, batch_and_masks):
    """Loss should be differentiable w.r.t. encoder and predictor parameters."""
    batch, masks = batch_and_masks
    model.train()
    z_hat, z_tgt, ctx_embs = model(batch, masks)
    total, _ = jepa_loss(z_hat, z_tgt, ctx_embs)
    total.backward()
    # At least some gradient should be non-zero
    grads = [p.grad for p in model.context_encoder.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)


# ------------------------------------------------------------------
# Batched vs per-subject equivalence
# ------------------------------------------------------------------

def test_batched_forward_matches_per_subject(model, batch_and_masks, dataset):
    """The batched forward must reproduce per-subject computation exactly."""
    import torch.nn.functional as TF

    from brain_jepa.masking.subnetwork_masking import extract_subgraph

    batch, masks = batch_and_masks
    model.eval()
    with torch.no_grad():
        z_hat, z_tgt, ctx_embs = model(batch, masks)

        # Reference: encode and predict one subject at a time
        z_hat_ref, z_tgt_ref = [], []
        for b, data_b in enumerate(batch.to_data_list()):
            ctx_mask = masks.context_node_masks[b]
            tgt_mask = masks.target_node_masks[b]
            ctx_emb = model.context_encoder(extract_subgraph(data_b, ctx_mask))
            tgt_emb = model.target_encoder(data_b)[tgt_mask]
            tgt_emb = TF.layer_norm(tgt_emb, (tgt_emb.shape[-1],))
            pred = model.predictor(
                ctx_emb,
                data_b.rsn_ids[ctx_mask],
                ctx_mask.nonzero(as_tuple=True)[0],
                data_b.rsn_ids[tgt_mask],
                tgt_mask.nonzero(as_tuple=True)[0],
            )
            z_hat_ref.append(pred)
            z_tgt_ref.append(tgt_emb)

    assert torch.allclose(z_hat, torch.cat(z_hat_ref), atol=1e-4)
    assert torch.allclose(z_tgt, torch.cat(z_tgt_ref), atol=1e-4)
