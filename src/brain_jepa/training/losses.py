"""BS-JEPA loss — cosine prediction loss + variance/covariance regularization."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _off_diagonal(mat: torch.Tensor) -> torch.Tensor:
    """Return a flat view of all off-diagonal elements of a square matrix."""
    n, m = mat.shape
    assert n == m
    return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def jepa_loss(
    z_hat: torch.Tensor,
    z_tgt: torch.Tensor,
    ctx_embs: torch.Tensor,
    var_weight: float = 0.5,
    ctx_var_weight: float = 1.0,
    cov_weight: float = 0.1,
    var_gamma: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cosine prediction loss + variance/covariance regularisation.

    Loss = sim_loss
         + var_weight     * hat_var_loss
         + ctx_var_weight * ctx_var_loss
         + cov_weight     * ctx_cov_loss

    - sim_loss:      cosine distance between predicted and target node embeddings.
    - hat_var_loss:  variance regularisation on predictor output (z_hat).
    - ctx_var_loss:  variance regularisation DIRECTLY on context encoder outputs.
                     Undiluted gradient to the context encoder, preventing
                     collapse without propagating through the predictor.
    - ctx_cov_loss:  VICReg-style covariance penalty on context encoder outputs.
                     Per-dimension variance alone can be satisfied by embeddings
                     living in a low-rank subspace (dimensional collapse); the
                     covariance term decorrelates dimensions.

    Args:
        z_hat:          Predicted node embeddings (N_tgt_total, d).
        z_tgt:          Target node embeddings (N_tgt_total, d). Detached.
        ctx_embs:       Context encoder outputs (N_ctx_total, d). Has gradients.
        var_weight:     Weight for hat variance term.
        ctx_var_weight: Weight for context encoder variance term.
        cov_weight:     Weight for context encoder covariance term.
        var_gamma:      Target std; dims below this are penalised.

    Returns:
        (total, metrics) — *total* is the differentiable scalar loss; *metrics*
        is a dict of detached scalar diagnostics: ``sim``, ``ctx_var``,
        ``hat_var``, ``ctx_cov``, ``tgt_std``.
    """
    # Cosine similarity loss
    z_hat_n = F.normalize(z_hat, dim=-1)
    z_tgt_n = F.normalize(z_tgt, dim=-1)
    sim_loss = 2 - 2 * (z_hat_n * z_tgt_n).sum(dim=-1).mean()

    # Variance on predictor output
    hat_var_loss = F.relu(var_gamma - z_hat.std(dim=0)).mean()

    # Variance directly on context encoder outputs — undiluted gradient signal
    ctx_var_loss = F.relu(var_gamma - ctx_embs.std(dim=0)).mean()

    # Covariance penalty on context encoder outputs (off-diagonal decorrelation)
    n, d = ctx_embs.shape
    x = ctx_embs - ctx_embs.mean(dim=0)
    cov = (x.T @ x) / max(n - 1, 1)
    ctx_cov_loss = _off_diagonal(cov).pow(2).sum() / d

    # Diagnostic: target std (detached, no gradient)
    tgt_std = z_tgt.std(dim=0).mean()

    total = (
        sim_loss
        + var_weight * hat_var_loss
        + ctx_var_weight * ctx_var_loss
        + cov_weight * ctx_cov_loss
    )
    metrics = {
        "sim": sim_loss.detach(),
        "ctx_var": ctx_var_loss.detach(),
        "hat_var": hat_var_loss.detach(),
        "ctx_cov": ctx_cov_loss.detach(),
        "tgt_std": tgt_std.detach(),
    }
    return total, metrics
