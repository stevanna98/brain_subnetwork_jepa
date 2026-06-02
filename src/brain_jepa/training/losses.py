"""BS-JEPA loss — cosine prediction loss in representation space."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def jepa_loss(z_hat: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
    """Cosine similarity loss between predictions and targets.

    Both tensors are L2-normalised before comparison, removing the trivial
    constant-magnitude shortcut that causes representation collapse under
    plain MSE with LayerNorm-normalised targets.

    Loss = 2 - 2 * cosine_similarity, which is 0 when perfectly aligned
    and 4 when opposite.  Gradients are well-scaled regardless of magnitude.

    The target tensor should already have stop-gradient applied (detached)
    by the caller.

    Args:
        z_hat: Predicted subnetwork tokens, shape (B, M, d).
        z_tgt: Target subnetwork tokens, shape (B, M, d). Should be detached.

    Returns:
        Scalar loss value.
    """
    z_hat = F.normalize(z_hat, dim=-1)
    z_tgt = F.normalize(z_tgt, dim=-1)
    return 2 - 2 * (z_hat * z_tgt).sum(dim=-1).mean()
