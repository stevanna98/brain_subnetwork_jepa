"""Narrow Transformer predictor for spatiotemporal BS-JEPA.

Mirrors the static :class:`SubnetworkPredictor`: context tokens (visible encoder
outputs) and target mask queries are concatenated into one sequence, run through
a narrow pre-norm Transformer, and the target positions are read out as the
predicted latents. Target queries carry detached RSN + time identity so each
query is unique without giving the predictor a gradient shortcut.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatioTemporalPredictor(nn.Module):
    """Predict target RSN-time latents from visible context tokens.

    Args:
        encoder_dim: d, dimension of encoder outputs / prediction targets.
        predictor_dim: internal width (< encoder_dim recommended; bottleneck).
        num_rsns: K.
        time_max_windows: upper bound on P.
        depth / num_heads / mlp_ratio / dropout: Transformer config.
    """

    def __init__(
        self,
        encoder_dim: int,
        predictor_dim: int = 128,
        num_rsns: int = 12,
        time_max_windows: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if predictor_dim % num_heads != 0:
            raise ValueError(
                f"predictor_dim ({predictor_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.predictor_dim = predictor_dim
        self.disable_target_identity = False

        self.input_proj = nn.Linear(encoder_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, predictor_dim))
        self.rsn_embed = nn.Embedding(num_rsns, predictor_dim)
        self.time_embed = nn.Embedding(time_max_windows, predictor_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=int(predictor_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.rsn_embed.weight, std=0.02)
        nn.init.normal_(self.time_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        ctx_tokens: torch.Tensor,       # (N_ctx, encoder_dim)
        ctx_rsn_ids: torch.Tensor,      # (N_ctx,)
        ctx_time_ids: torch.Tensor,     # (N_ctx,)
        tgt_rsn_ids: torch.Tensor,      # (N_tgt,)
        tgt_time_ids: torch.Tensor,     # (N_tgt,)
    ) -> torch.Tensor:
        """Predict latents for the target tokens of one subject → ``(N_tgt, encoder_dim)``."""
        ctx = self.input_proj(ctx_tokens)
        ctx = ctx + self.rsn_embed(ctx_rsn_ids) + self.time_embed(ctx_time_ids)

        n_tgt = tgt_rsn_ids.shape[0]
        tgt = self.mask_token.expand(n_tgt, -1)
        if not self.disable_target_identity:
            tgt = tgt + self.rsn_embed(tgt_rsn_ids).detach()
            tgt = tgt + self.time_embed(tgt_time_ids).detach()

        seq = torch.cat([ctx, tgt], dim=0).unsqueeze(0)   # (1, N_ctx+N_tgt, P)
        out = self.norm(self.transformer(seq)).squeeze(0)
        return self.output_proj(out[ctx.shape[0]:])       # (N_tgt, encoder_dim)
