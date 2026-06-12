"""Tokenizer and Transformer encoder for spatiotemporal BS-JEPA.

``RSNTimeTokenizer`` turns windowed BOLD ``(B, P, N, L)`` into RSN-time tokens
``(B, P, K, d)`` via a per-region-window patch embedder followed by mean pooling
within each subnetwork. ``SpatioTemporalEncoder`` is a Transformer over the
flattened ``(P*K)`` token sequence with learned RSN and time positional
embeddings; it accepts a key-padding mask so the context branch can encode an
arbitrary subset of visible tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.transforms import FeatureMode, build_feature_module


class RSNTimeTokenizer(nn.Module):
    """Patch-embed each region-window, then mean-pool regions into RSN tokens.

    Args:
        window_length: L, timepoints per window (patch embedder input size).
        embed_dim: d, token dimension.
        rsn_ids: (N,) region→RSN assignment (0-indexed), from the atlas.
        num_rsns: K.
        feature_mode: "passthrough" | "conv1d" — reuses the existing patch
            embedders that map (M, L) → (M, d).
    """

    def __init__(
        self,
        window_length: int,
        embed_dim: int,
        rsn_ids: torch.Tensor,
        num_rsns: int,
        feature_mode: FeatureMode = "conv1d",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_rsns = num_rsns
        self.patch_embed = build_feature_module(feature_mode, window_length, embed_dim)
        # (K, N) row-normalised pooling matrix: pooled[k] = mean of regions in RSN k.
        n = rsn_ids.shape[0]
        onehot = torch.zeros(num_rsns, n)
        onehot[rsn_ids, torch.arange(n)] = 1.0
        onehot = onehot / onehot.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.register_buffer("rsn_pool", onehot)  # (K, N)

    def forward(self, x_win: torch.Tensor) -> torch.Tensor:
        """``(B, P, N, L)`` → ``(B, P, K, d)``."""
        b, p, n, l = x_win.shape
        h = self.patch_embed(x_win.reshape(b * p * n, l))   # (B*P*N, d)
        h = h.reshape(b, p, n, self.embed_dim)
        # RSN mean pool over regions: (K, N) x (B, P, N, d) -> (B, P, K, d)
        tokens = torch.einsum("kn,bpnd->bpkd", self.rsn_pool.to(h.dtype), h)
        return tokens


class SpatioTemporalEncoder(nn.Module):
    """Transformer over flattened RSN-time tokens with RSN + time positions.

    Args:
        embed_dim: d (input == output dim).
        num_rsns: K, for the RSN positional embedding.
        time_max_windows: upper bound on P, for the time positional embedding.
        depth / num_heads / mlp_ratio / dropout: Transformer config.
    """

    def __init__(
        self,
        embed_dim: int,
        num_rsns: int,
        time_max_windows: int,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.embed_dim = embed_dim
        self.rsn_embed = nn.Embedding(num_rsns, embed_dim)
        self.time_embed = nn.Embedding(time_max_windows, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.rsn_embed.weight, std=0.02)
        nn.init.normal_(self.time_embed.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,        # (B, S, d)
        rsn_ids: torch.Tensor,       # (B, S)
        time_ids: torch.Tensor,      # (B, S)
        key_padding_mask: torch.Tensor | None = None,  # (B, S) True = pad
    ) -> torch.Tensor:
        x = tokens + self.rsn_embed(rsn_ids) + self.time_embed(time_ids)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.norm(x)
