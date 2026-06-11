"""Optional feature-extraction transforms applied to region time-series."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

FeatureMode = Literal["passthrough", "conv1d"]


def build_feature_module(mode: FeatureMode, in_channels: int, out_channels: int) -> nn.Module:
    """Factory for time-series → feature-vector transforms.

    All modules accept (N, T) and return (N, out_channels).
    """
    if mode == "passthrough":
        return PassthroughFeatures(in_channels, out_channels)
    if mode == "conv1d":
        return Conv1dFeatures(in_channels, out_channels)
    raise ValueError(f"Unknown feature mode: {mode!r}")


class PassthroughFeatures(nn.Module):
    """Linear projection of mean time-series features (N, T) → (N, F)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T) — treat T as the feature dimension after mean
        return self.proj(x)


class Conv1dFeatures(nn.Module):
    """Two-layer temporal CNN extracting (N, F) features from (N, T) time series.

    Input time series are z-scored (zero mean), so a nonlinearity must precede
    any temporal pooling: the time-average of a purely linear conv response of
    a zero-mean signal collapses to the bias term. Mean and max pooling are
    concatenated so the features capture both sustained spectral content and
    transient events.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        mid = max(out_channels // 2, 1)
        self.net = nn.Sequential(
            nn.Conv1d(1, mid, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(mid, out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.GELU(),
        )
        self.proj = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.unsqueeze(1))                       # (N, F, T')
        h = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)  # (N, 2F)
        return self.proj(h)


