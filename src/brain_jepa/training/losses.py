"""BS-JEPA loss — cosine prediction loss + variance regularization."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def jepa_loss(
    z_hat: torch.Tensor,
    z_tgt: torch.Tensor,
    var_weight: float = 0.5,
    var_gamma: float = 0.1,
) -> torch.Tensor:
    """Cosine similarity loss with variance regularization to prevent collapse.

    Both tensors are node-level: shape (N_total, d) where N_total is the
    total number of target nodes across all subjects in the batch.

    Loss = cosine_loss + var_weight * var_loss

    where:
      cosine_loss = mean(2 - 2 * cos(z_hat_i, z_tgt_i))  per node, in [0, 4]
      var_loss    = mean(relu(var_gamma - std(z_tgt, dim=0)))  across all nodes

    Args:
        z_hat: Predicted node embeddings, shape (N_total, d).
        z_tgt: Target node embeddings, shape (N_total, d). Should be detached.
        var_weight: Weight for the variance regularization term.
        var_gamma: Target std; dims with std < gamma are penalised.

    Returns:
        Scalar loss value.
    """
    # Cosine similarity loss — one term per target node
    z_hat_n = F.normalize(z_hat, dim=-1)
    z_tgt_n = F.normalize(z_tgt, dim=-1)
    sim_loss = 2 - 2 * (z_hat_n * z_tgt_n).sum(dim=-1).mean()

    # Variance regularization on z_hat (has gradients through predictor +
    # context encoder).  Applying it to z_tgt would be a no-op because z_tgt
    # is detached.  Pushing z_hat to be diverse forces the context encoder to
    # produce diverse outputs; via EMA the target encoder follows.
    std_hat = z_hat.std(dim=0)                        # (d,)
    var_loss = F.relu(var_gamma - std_hat).mean()

    # Log target std for monitoring collapse (no gradient, diagnostic only)
    mean_std = z_tgt.std(dim=0).mean()

    total = sim_loss + var_weight * var_loss
    return total, sim_loss.detach(), var_loss.detach(), mean_std.detach()
