"""Optimizer, LR schedule, and weight-decay schedule for BS-JEPA pretraining.

Follows I-JEPA's conventions:
- AdamW optimizer.
- Cosine LR schedule with linear warmup.
- Linear weight-decay schedule (0.04 → 0.4).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class WarmupCosineSchedule:
    """Cosine LR schedule with a linear warm-up phase."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        start_lr: float,
        ref_lr: float,
        total_steps: int,
        final_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.cosine_steps = max(total_steps - warmup_steps, 1)
        self._step = 0

    def step(self) -> float:
        self._step += 1
        if self._step <= self.warmup_steps:
            progress = self._step / max(1, self.warmup_steps)
            lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = (self._step - self.warmup_steps) / self.cosine_steps
            lr = self.final_lr + 0.5 * (self.ref_lr - self.final_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


class LinearWDSchedule:
    """Linear weight-decay schedule: wd_start → wd_end over *total_steps*."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        wd_start: float = 0.04,
        wd_end: float = 0.4,
        total_steps: int = 1,
    ) -> None:
        self.optimizer = optimizer
        self.wd_start = wd_start
        self.wd_end = wd_end
        self.total_steps = max(total_steps, 1)
        self._step = 0

    def step(self) -> float:
        self._step += 1
        progress = min(self._step / self.total_steps, 1.0)
        wd = self.wd_start + progress * (self.wd_end - self.wd_start)
        for pg in self.optimizer.param_groups:
            if not pg.get("no_wd", False):
                pg["weight_decay"] = wd
        return wd


def build_optimizer(
    *modules: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.04,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Create AdamW optimizer over any number of modules; bias/norm params exempt from WD."""
    no_wd_params: list[torch.Tensor] = []
    wd_params: list[torch.Tensor] = []

    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or "norm" in name.lower():
                no_wd_params.append(param)
            else:
                wd_params.append(param)

    param_groups = [
        {"params": wd_params, "weight_decay": weight_decay},
        {"params": no_wd_params, "weight_decay": 0.0, "no_wd": True},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
