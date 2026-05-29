"""BS-JEPA loss — L2 prediction loss in representation space."""

from __future__ import annotations

import torch


def jepa_loss(z_hat: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
    """Compute the mean squared L2 loss between predictions and targets.

    The target tensor should already have stop-gradient applied (detached)
    by the caller.  This function does NOT call ``.detach()`` itself so that
    gradient inspection is straightforward.

    Args:
        z_hat: Predicted subnetwork tokens, shape (B, M, d).
        z_tgt: Target subnetwork tokens, shape (B, M, d). Should be detached.

    Returns:
        Scalar loss value.
    """
    diff = z_hat - z_tgt
    # Mean over batch, targets, and feature dimensions
    return (diff ** 2).mean()
