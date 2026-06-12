"""Representation-health diagnostics for monitoring BS-JEPA pretraining.

These are lightweight, label-free checks run on a held-out loader during
pretraining to detect representation *collapse* — the failure mode where the
JEPA similarity loss is minimised by the encoder emitting near-constant or
low-rank embeddings. None of these touch the optimiser, EMA, or the loss
definition; they only read the frozen target encoder's outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def effective_rank(features: torch.Tensor) -> float:
    """Effective rank (Roy & Vetterli, 2007) of a (N, d) embedding matrix.

    Defined as ``exp(H)`` where ``H`` is the Shannon entropy of the normalised
    singular-value spectrum of the *centred* feature matrix. Ranges from 1
    (all variance in one direction → dimensional collapse) up to ``min(N, d)``
    (variance spread evenly across directions). Returns 0.0 if undefined.
    """
    if features.ndim != 2 or features.shape[0] < 2:
        return 0.0
    x = (features - features.mean(dim=0, keepdim=True)).float()
    try:
        sv = torch.linalg.svdvals(x)
    except RuntimeError:  # pragma: no cover - numerical edge case
        return 0.0
    sv = sv[sv > 1e-12]
    if sv.numel() == 0:
        return 0.0
    p = sv / sv.sum()
    entropy = -(p * p.log()).sum()
    return float(entropy.exp())


@torch.no_grad()
def representation_health(features: torch.Tensor) -> dict[str, float]:
    """Compute collapse-sensitive statistics of subject-level embeddings.

    Args:
        features: (N_subjects, d) tensor of per-subject embeddings (e.g. the
            mean-pooled node embeddings produced by ``extract_representations``).

    Returns:
        dict with:
          - ``embedding_std_mean``:   mean over dimensions of the per-dimension
            std across subjects (low → embeddings barely vary between subjects).
          - ``embedding_std_min``:    smallest per-dimension std (a single dead
            dimension shows up here before it moves the mean).
          - ``mean_pairwise_cosine``: mean off-diagonal cosine similarity between
            subject embeddings (→1 → all subjects map to the same direction).
          - ``effective_rank``:       effective rank of the embedding matrix.
    """
    features = features.detach().float()
    n = features.shape[0]
    std = features.std(dim=0)
    metrics = {
        "embedding_std_mean": float(std.mean()),
        "embedding_std_min": float(std.min()),
        "effective_rank": effective_rank(features),
    }
    if n >= 2:
        normed = F.normalize(features, dim=1)
        sim = normed @ normed.T
        off_diag_sum = sim.sum() - torch.diagonal(sim).sum()
        metrics["mean_pairwise_cosine"] = float(off_diag_sum / (n * (n - 1)))
    else:
        metrics["mean_pairwise_cosine"] = float("nan")
    return metrics


@dataclass
class CollapseThresholds:
    """Configurable thresholds for collapse / instability warnings.

    Defaults are deliberately permissive — they fire only on clearly pathological
    values so the warnings stay meaningful rather than noisy.

    Attributes:
        tgt_std_min:          warn if the (train) target-embedding std drops below this.
        embedding_std_min:    warn if ``embedding_std_mean`` drops below this.
        pairwise_cosine_max:  warn if ``mean_pairwise_cosine`` exceeds this.
        grad_norm_max:        warn if the gradient norm exceeds this (or is 0/NaN/Inf).
    """

    tgt_std_min: float = 1e-2
    embedding_std_min: float = 1e-3
    pairwise_cosine_max: float = 0.95
    grad_norm_max: float = 1e3


def collapse_warnings(metrics: dict[str, float], thr: CollapseThresholds) -> list[str]:
    """Return a list of human-readable warning strings for *metrics*.

    Only keys present in *metrics* are checked, so this works whether or not the
    representation diagnostics were run this epoch.
    """
    warnings: list[str] = []

    tgt_std = metrics.get("train_tgt_std")
    if tgt_std is not None and tgt_std < thr.tgt_std_min:
        warnings.append(
            f"train_tgt_std={tgt_std:.4g} < {thr.tgt_std_min:g} — target embeddings "
            "barely vary; possible collapse."
        )

    emb_std = metrics.get("embedding_std_mean")
    if emb_std is not None and emb_std < thr.embedding_std_min:
        warnings.append(
            f"embedding_std_mean={emb_std:.4g} < {thr.embedding_std_min:g} — held-out "
            "embeddings are nearly constant across subjects; likely collapse."
        )

    cos = metrics.get("mean_pairwise_cosine")
    if cos is not None and cos == cos and cos > thr.pairwise_cosine_max:  # cos==cos rejects NaN
        warnings.append(
            f"mean_pairwise_cosine={cos:.4g} > {thr.pairwise_cosine_max:g} — subjects map "
            "to almost the same direction; likely collapse."
        )

    grad = metrics.get("grad_norm")
    if grad is not None:
        if grad != grad or grad in (float("inf"), float("-inf")):
            warnings.append(f"grad_norm={grad} — non-finite gradients; training is diverging.")
        elif grad == 0.0:
            warnings.append("grad_norm=0 — no gradient signal reaching trained parameters.")
        elif grad > thr.grad_norm_max:
            warnings.append(
                f"grad_norm={grad:.4g} > {thr.grad_norm_max:g} — exploding gradients."
            )

    return warnings
