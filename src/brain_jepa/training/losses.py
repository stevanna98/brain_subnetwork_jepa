"""BS-JEPA loss — cosine prediction loss + variance regularization."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def jepa_loss(
    z_hat: torch.Tensor,
    z_tgt: torch.Tensor,
    ctx_embs: torch.Tensor,
    var_weight: float = 0.5,
    ctx_var_weight: float = 1.0,
    var_gamma: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cosine prediction loss + direct context encoder variance regularisation.

    Loss = sim_loss + var_weight * hat_var_loss + ctx_var_weight * ctx_var_loss

    - sim_loss:      cosine distance between predicted and target node embeddings.
    - hat_var_loss:  variance regularisation on predictor output (z_hat).
    - ctx_var_loss:  variance regularisation DIRECTLY on context encoder outputs.
                     This is the key term: it gives an undiluted gradient to the
                     context encoder, preventing it from collapsing without having
                     to propagate through the predictor's many layers.

    Args:
        z_hat:         Predicted node embeddings (N_tgt_total, d).
        z_tgt:         Target node embeddings (N_tgt_total, d). Detached.
        ctx_embs:      Context encoder outputs (N_ctx_total, d). Has gradients.
        var_weight:    Weight for hat variance term.
        ctx_var_weight:Weight for context encoder variance term.
        var_gamma:     Target std; dims below this are penalised.

    Returns:
        (total, sim_loss, ctx_var_loss, hat_var_loss, tgt_std) — all scalars.
    """
    # Cosine similarity loss
    z_hat_n = F.normalize(z_hat, dim=-1)
    z_tgt_n = F.normalize(z_tgt, dim=-1)
    sim_loss = 2 - 2 * (z_hat_n * z_tgt_n).sum(dim=-1).mean()

    # Variance on predictor output
    hat_var_loss = F.relu(var_gamma - z_hat.std(dim=0)).mean()

    # Variance directly on context encoder outputs — undiluted gradient signal
    ctx_var_loss = F.relu(var_gamma - ctx_embs.std(dim=0)).mean()

    # Diagnostic: target std (detached, no gradient)
    tgt_std = z_tgt.std(dim=0).mean()

    total = sim_loss + var_weight * hat_var_loss + ctx_var_weight * ctx_var_loss
    return total, sim_loss.detach(), ctx_var_loss.detach(), hat_var_loss.detach(), tgt_std.detach()
