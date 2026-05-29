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
        pooling_mode="mean",
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
    B, K_c, M, D_enc, P = 4, 11, 1, D, 32
    pred = SubnetworkPredictor(encoder_dim=D_enc, predictor_dim=P, num_rsns=K, depth=2, num_heads=2)
    pred.eval()
    context_tokens = torch.randn(B, K_c, D_enc)
    context_rsn_ids = torch.stack([torch.randperm(K)[:K_c] for _ in range(B)])
    target_rsn_ids = torch.stack([torch.randperm(K)[K_c : K_c + M] for _ in range(B)])
    with torch.no_grad():
        out = pred(context_tokens, context_rsn_ids, target_rsn_ids)
    assert out.shape == (B, M, D_enc)


# ------------------------------------------------------------------
# Full model forward pass
# ------------------------------------------------------------------

def test_bsjepa_forward_shapes(model, batch_and_masks):
    batch, masks = batch_and_masks
    model.eval()
    with torch.no_grad():
        z_hat, z_tgt = model(batch, masks)
    B = len(batch.to_data_list())
    M = masks.target_rsn_ids.shape[1]
    assert z_hat.shape == (B, M, D)
    assert z_tgt.shape == (B, M, D)


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

def test_jepa_loss_zero():
    x = torch.randn(4, 1, D)
    assert jepa_loss(x, x.detach()) == pytest.approx(0.0, abs=1e-6)


def test_jepa_loss_nonnegative():
    z_hat = torch.randn(4, 1, D)
    z_tgt = torch.randn(4, 1, D)
    loss = jepa_loss(z_hat, z_tgt.detach())
    assert loss.item() >= 0.0


def test_jepa_loss_backward(model, batch_and_masks):
    """Loss should be differentiable w.r.t. encoder and predictor parameters."""
    batch, masks = batch_and_masks
    model.train()
    z_hat, z_tgt = model(batch, masks)
    loss = jepa_loss(z_hat, z_tgt)
    loss.backward()
    # At least some gradient should be non-zero
    grads = [p.grad for p in model.context_encoder.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)
