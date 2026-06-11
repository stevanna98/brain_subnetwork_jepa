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
    """Temporal convolution to extract (N, F) features from (N, T) time series."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, out_channels, kernel_size=kernel_size, stride=kernel_size // 2)
        # Input time series are z-scored (zero mean): averaging a *linear* conv
        # response over time would collapse to the bias term, so a nonlinearity
        # must precede the temporal pooling for the features to carry signal.
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv(x.unsqueeze(1)))  # (N, F, T')
        h = self.pool(h).squeeze(-1)             # (N, F)
        return self.proj(h)


