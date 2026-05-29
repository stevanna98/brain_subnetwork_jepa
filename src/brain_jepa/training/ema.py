"""Exponential Moving Average (EMA) target-encoder update.

The target encoder is never updated by backprop; it tracks the context
encoder via an EMA with a momentum schedule that ramps from tau_start → 1.0
over the course of pretraining (matching I-JEPA's convention).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EMAUpdater:
    """Updates the target encoder as an EMA of the context encoder.

    Usage::

        updater = EMAUpdater(tau_start=0.996, tau_end=1.0, total_steps=n)
        for step in range(n):
            updater.step(context_encoder, target_encoder)

    Args:
        tau_start: Initial momentum value (close to 1).
        tau_end: Final momentum value (1.0 → target freezes).
        total_steps: Total number of update steps (= epochs × iters_per_epoch).
    """

    def __init__(
        self,
        tau_start: float = 0.996,
        tau_end: float = 1.0,
        total_steps: int = 1,
    ) -> None:
        assert 0.0 < tau_start <= tau_end <= 1.0
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_steps = max(total_steps, 1)
        self._step = 0

    @property
    def current_tau(self) -> float:
        progress = self._step / self.total_steps
        return self.tau_start + progress * (self.tau_end - self.tau_start)

    @torch.no_grad()
    def step(self, context: nn.Module, target: nn.Module) -> float:
        """Update *target* parameters with EMA of *context* parameters.

        Returns the momentum value used for this step.
        """
        tau = self.current_tau
        for p_ctx, p_tgt in zip(context.parameters(), target.parameters(), strict=True):
            p_tgt.data.mul_(tau).add_((1.0 - tau) * p_ctx.detach().data)
        self._step += 1
        return tau
