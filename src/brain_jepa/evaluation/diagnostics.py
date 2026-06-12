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


@torch.no_grad()
def pooled_embeddings(
    model,
    loader,
    device: torch.device,
    num_rsns: int | None = None,
) -> dict:
    """Extract per-subject embeddings under several pooling strategies (one pass).

    Pools each subject's node embeddings (from the frozen target encoder via
    ``model.encode``) into:
      - ``"mean"``:     global mean over nodes                → (N_subj, d)
      - ``"rsn"``:      mean within each RSN, then flattened  → (N_subj, K*d)
      - ``"mean_std"``: concat of node mean and node std      → (N_subj, 2d)

    Also returns node-level statistics that localise *where* any collapse occurs:
      - ``node_embedding_std_mean``:      std across ALL nodes (every subject,
        every region), averaged over dims — collapse *inside the encoder*.
      - ``within_subject_node_std_mean``: per-subject std across its own nodes,
        averaged over subjects — whether nodes vary within a subject.

    Returns a dict with the three (N_subj, ·) tensors, the two node-level floats,
    and ``subject_ids``.
    """
    model.eval()
    if num_rsns is None:
        num_rsns = getattr(getattr(model, "atlas", None), "num_rsns", None)

    mean_pool: list[torch.Tensor] = []
    rsn_pool: list[torch.Tensor] = []
    meanstd_pool: list[torch.Tensor] = []
    subject_ids: list[str] = []
    within_std: list[float] = []

    # Streaming accumulators for the global (all-node) std.
    sum_d: torch.Tensor | None = None
    sumsq_d: torch.Tensor | None = None
    n_nodes = 0

    for batch in loader:
        data_list = batch if isinstance(batch, list) else [batch]
        for data in data_list:
            data = data.to(device)
            node_emb = model.encode(data)  # (N, d)
            d = node_emb.shape[-1]

            mean_pool.append(node_emb.mean(dim=0).cpu())
            meanstd_pool.append(
                torch.cat([node_emb.mean(dim=0), node_emb.std(dim=0)]).cpu()
            )
            within_std.append(float(node_emb.std(dim=0).mean()))

            if num_rsns is not None and hasattr(data, "rsn_ids"):
                rsn_ids = data.rsn_ids
                pooled = node_emb.new_zeros(num_rsns, d)
                for k in range(num_rsns):
                    mask = rsn_ids == k
                    if mask.any():
                        pooled[k] = node_emb[mask].mean(dim=0)
                rsn_pool.append(pooled.flatten().cpu())

            subject_ids.append(str(getattr(data, "subject_id", "")))

            node_sum = node_emb.sum(dim=0)
            node_sumsq = (node_emb ** 2).sum(dim=0)
            sum_d = node_sum if sum_d is None else sum_d + node_sum
            sumsq_d = node_sumsq if sumsq_d is None else sumsq_d + node_sumsq
            n_nodes += node_emb.shape[0]

    out: dict = {
        "mean": torch.stack(mean_pool),
        "mean_std": torch.stack(meanstd_pool),
        "subject_ids": subject_ids,
        "within_subject_node_std_mean": float(sum(within_std) / max(len(within_std), 1)),
    }
    if rsn_pool:
        out["rsn"] = torch.stack(rsn_pool)
    if sum_d is not None and n_nodes > 0:
        mean_d = sum_d / n_nodes
        var_d = (sumsq_d / n_nodes) - mean_d ** 2
        out["node_embedding_std_mean"] = float(var_d.clamp(min=0).sqrt().mean().cpu())
    else:
        out["node_embedding_std_mean"] = float("nan")
    return out


def pooling_comparison(pooled: dict) -> dict[str, dict[str, float]]:
    """Run :func:`representation_health` on each pooling strategy in *pooled*."""
    results: dict[str, dict[str, float]] = {}
    for key in ("mean", "rsn", "mean_std"):
        if key in pooled:
            results[key] = representation_health(pooled[key])
    return results


@torch.no_grad()
def debug_report(model, loader, device: torch.device) -> dict:
    """Run one diagnostic pass and return (and log) a detailed report.

    Intended for a manual smoke run on a (possibly loaded) checkpoint. Reports
    subject count, embedding shapes, sample subject IDs / norms, pairwise-cosine
    spread, ranks, node-level stats, and per-pooling health metrics.
    """
    pooled = pooled_embeddings(model, loader, device)
    mean_emb = pooled["mean"]
    n = mean_emb.shape[0]

    normed = F.normalize(mean_emb, dim=1)
    sim = normed @ normed.T
    off = sim[~torch.eye(n, dtype=torch.bool)] if n > 1 else torch.tensor([float("nan")])

    report = {
        "num_subjects": n,
        "num_unique_subject_ids": len(set(pooled["subject_ids"])),
        "embedding_shapes": {k: list(pooled[k].shape) for k in ("mean", "rsn", "mean_std") if k in pooled},
        "first_subject_ids": pooled["subject_ids"][:5],
        "first_embedding_norms": [round(float(mean_emb[i].norm()), 6) for i in range(min(5, n))],
        "pairwise_cosine_min": float(off.min()) if n > 1 else float("nan"),
        "pairwise_cosine_mean": float(off.mean()) if n > 1 else float("nan"),
        "pairwise_cosine_max": float(off.max()) if n > 1 else float("nan"),
        "node_embedding_std_mean": pooled["node_embedding_std_mean"],
        "within_subject_node_std_mean": pooled["within_subject_node_std_mean"],
        "pooling": pooling_comparison(pooled),
    }

    logger.info("=== diagnostics debug report ===")
    logger.info("subjects: %d (unique ids: %d)", report["num_subjects"], report["num_unique_subject_ids"])
    logger.info("embedding shapes: %s", report["embedding_shapes"])
    logger.info("first subject ids: %s", report["first_subject_ids"])
    logger.info("first embedding norms: %s", report["first_embedding_norms"])
    logger.info(
        "pairwise cosine  min=%.4f mean=%.4f max=%.4f",
        report["pairwise_cosine_min"], report["pairwise_cosine_mean"], report["pairwise_cosine_max"],
    )
    logger.info(
        "node_embedding_std_mean=%.3e | within_subject_node_std_mean=%.3e",
        report["node_embedding_std_mean"], report["within_subject_node_std_mean"],
    )
    for name, m in report["pooling"].items():
        logger.info(
            "pool=%-8s | emb_std_mean=%.3e | emb_std_min=%.3e | mean_pairwise_cos=%.4f | eff_rank=%.3f",
            name, m["embedding_std_mean"], m["embedding_std_min"],
            m["mean_pairwise_cosine"], m["effective_rank"],
        )
    return report


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
