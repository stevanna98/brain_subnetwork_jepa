"""BS-JEPA loss — cosine prediction loss + variance regularization."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def jepa_loss(
    z_hat: torch.Tensor,
    z_tgt: torch.Tensor,
    var_weight: float = 0.5,
    var_gamma: float = 1.0,
) -> torch.Tensor:
    """Cosine similarity loss with variance regularization to prevent collapse.

    The cosine term removes the constant-magnitude shortcut of plain MSE.
    The variance term (from VICReg) penalises low spread of target tokens
    across the batch, preventing the model from mapping all subjects to the
    same representation direction.

    Loss = cosine_loss + var_weight * var_loss

    where:
      cosine_loss = mean(2 - 2 * cos(z_hat, z_tgt))  in [0, 4]
      var_loss    = mean(relu(var_gamma - std(z_tgt, dim=batch)))

    Args:
        z_hat: Predicted subnetwork tokens, shape (B, M, d).
        z_tgt: Target subnetwork tokens, shape (B, M, d). Should be detached.
        var_weight: Weight for the variance regularization term.
        var_gamma: Target standard deviation; penalises dims with std < gamma.

    Returns:
        Scalar loss value.
    """
    # Cosine similarity loss
    z_hat_n = F.normalize(z_hat, dim=-1)
    z_tgt_n = F.normalize(z_tgt, dim=-1)
    sim_loss = 2 - 2 * (z_hat_n * z_tgt_n).sum(dim=-1).mean()

    # Variance regularization: measure std of target tokens across the batch.
    # Reshape (B, M, d) → (B, M*d) so we measure diversity over all tokens
    # jointly, then penalise any dimension with std < var_gamma.
    B = z_tgt.shape[0]
    z_flat = z_tgt.reshape(B, -1)                    # (B, M*d)
    std = z_flat.std(dim=0)                           # (M*d,)
    var_loss = F.relu(var_gamma - std).mean()

    return sim_loss + var_weight * var_loss
