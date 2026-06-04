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

    # Variance regularization: std across all N_total target node embeddings
    # per feature dimension. Penalises any dim with std < var_gamma.
    std = z_tgt.std(dim=0)               # (d,)
    mean_std = std.mean()
    var_loss = F.relu(var_gamma - std).mean()

    total = sim_loss + var_weight * var_loss
    return total, sim_loss.detach(), var_loss.detach(), mean_std.detach()
