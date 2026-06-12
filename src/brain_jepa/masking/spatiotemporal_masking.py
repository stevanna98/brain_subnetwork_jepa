"""Spatiotemporal masking over the RSN-time grid for ST-BS-JEPA.

Samples target blocks in the (P windows x K subnetworks) grid and returns
boolean context/target masks. Modes:
  - "spatial":  mask whole RSNs across all windows.
  - "temporal": mask all RSNs across a contiguous span of windows.
  - "block":    mask rectangular RSN x time blocks (default).
Context is the complement of the target; the two never overlap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..data.windowing import STSample


@dataclass
class STBatch:
    """Collated spatiotemporal batch.

    Attributes:
        x_win: (B, P, N, L) windowed BOLD.
        subject_ids / ages / genders: per-subject metadata (length B).
    """

    x_win: torch.Tensor
    subject_ids: list[str]
    ages: list
    genders: list

    def to(self, device: torch.device) -> "STBatch":
        self.x_win = self.x_win.to(device)
        return self


@dataclass
class STMaskOutput:
    """Context/target masks over the RSN-time grid.

    Attributes:
        context_mask / target_mask: (B, P, K) bool, mutually exclusive.
        target_rsn_ids / target_time_ids: length-B lists of 1-D tensors giving
            the RSN and window id of each target token per subject.
    """

    context_mask: torch.Tensor
    target_mask: torch.Tensor
    target_rsn_ids: list[torch.Tensor]
    target_time_ids: list[torch.Tensor]

    def to(self, device: torch.device) -> "STMaskOutput":
        self.context_mask = self.context_mask.to(device)
        self.target_mask = self.target_mask.to(device)
        self.target_rsn_ids = [t.to(device) for t in self.target_rsn_ids]
        self.target_time_ids = [t.to(device) for t in self.target_time_ids]
        return self


class SpatioTemporalMaskCollator:
    """Collate :class:`STSample` objects and sample RSN-time target blocks.

    Args:
        num_rsns: K, number of subnetworks.
        mode: "spatial" | "temporal" | "block".
        target_rsn_ratio: fraction of RSNs targeted (spatial/block).
        target_time_ratio: fraction of windows targeted (temporal).
        num_target_blocks: number of rectangular blocks (block mode).
        block_time_length: window span per block (block mode).
        seed: optional RNG seed for reproducible masks.
    """

    def __init__(
        self,
        num_rsns: int = 12,
        mode: str = "block",
        target_rsn_ratio: float = 0.5,
        target_time_ratio: float = 0.5,
        num_target_blocks: int = 1,
        block_time_length: int = 2,
        seed: int | None = None,
    ) -> None:
        if mode not in ("spatial", "temporal", "block"):
            raise ValueError(f"Unknown mask mode: {mode!r}")
        self.num_rsns = num_rsns
        self.mode = mode
        self.target_rsn_ratio = target_rsn_ratio
        self.target_time_ratio = target_time_ratio
        self.num_target_blocks = num_target_blocks
        self.block_time_length = block_time_length
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(seed)

    def __call__(self, samples: list[STSample]) -> tuple[STBatch, STMaskOutput]:
        b = len(samples)
        p = samples[0].num_windows
        k = self.num_rsns

        x_win = torch.stack([s.x_win for s in samples])  # (B, P, N, L)
        target_mask = torch.zeros(b, p, k, dtype=torch.bool)
        for i in range(b):
            self._sample_one(target_mask[i], p, k)

        context_mask = ~target_mask
        target_rsn_ids: list[torch.Tensor] = []
        target_time_ids: list[torch.Tensor] = []
        for i in range(b):
            t_idx, r_idx = target_mask[i].nonzero(as_tuple=True)  # (·,), (·,)
            target_time_ids.append(t_idx)
            target_rsn_ids.append(r_idx)

        batch = STBatch(
            x_win=x_win,
            subject_ids=[s.subject_id for s in samples],
            ages=[s.age for s in samples],
            genders=[s.gender for s in samples],
        )
        masks = STMaskOutput(context_mask, target_mask, target_rsn_ids, target_time_ids)
        return batch, masks

    def _sample_one(self, mask: torch.Tensor, p: int, k: int) -> None:
        """Fill one subject's (P, K) target mask in-place; guarantee >=1 each."""
        if self.mode == "spatial":
            n_rsn = max(1, min(k - 1, round(self.target_rsn_ratio * k)))
            rsns = torch.randperm(k, generator=self._gen)[:n_rsn]
            mask[:, rsns] = True
        elif self.mode == "temporal":
            span = max(1, min(p - 1, round(self.target_time_ratio * p)))
            start = int(torch.randint(0, p - span + 1, (1,), generator=self._gen))
            mask[start : start + span, :] = True
        else:  # block
            n_rsn = max(1, min(k - 1, round(self.target_rsn_ratio * k)))
            span = max(1, min(p, self.block_time_length))
            for _ in range(self.num_target_blocks):
                start = int(torch.randint(0, max(1, p - span + 1), (1,), generator=self._gen))
                rsns = torch.randperm(k, generator=self._gen)[:n_rsn]
                t_slice = slice(start, start + span)
                for r in rsns:
                    mask[t_slice, r] = True

        # Safety: never mask everything, never mask nothing.
        if not mask.any():
            mask[0, 0] = True
        if mask.all():
            mask[0, 0] = False
