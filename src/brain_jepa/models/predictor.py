"""Narrow Transformer predictor — analogous to I-JEPA's VisionTransformerPredictor.

Given context subnetwork tokens and mask tokens for target subnetworks,
predicts the target-encoder's representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SubnetworkPredictor(nn.Module):
    """Narrow Transformer that predicts target subnetwork representations.

    Design mirrors I-JEPA's predictor:
    - Context tokens are projected to a narrower predictor dimension.
    - Each target position is represented by a shared learnable mask token
      summed with a learned subnetwork-identity embedding.
    - Context and target tokens are concatenated and passed through Transformer
      blocks; only the target positions are returned and projected back to the
      encoder embedding dimension.

    Args:
        encoder_dim: Output dimension d of the encoder.
        predictor_dim: Internal predictor width (should be < encoder_dim).
        num_rsns: Number of distinct subnetwork identities K.
        depth: Number of Transformer encoder layers.
        num_heads: Attention heads.
        mlp_ratio: FFN hidden expansion factor.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        encoder_dim: int,
        predictor_dim: int = 384,
        num_rsns: int = 12,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.num_rsns = num_rsns

        # Project context tokens from encoder dim to predictor dim
        self.input_proj = nn.Linear(encoder_dim, predictor_dim)

        # One learnable mask token shared across target positions
        self.mask_token = nn.Parameter(torch.zeros(1, predictor_dim))

        # Per-subnetwork identity embeddings (used for both context and target)
        self.rsn_embed = nn.Embedding(num_rsns, predictor_dim)

        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=int(predictor_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(predictor_dim)

        # Project predictions back to encoder dim
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.rsn_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_rsn_ids: torch.Tensor,
        target_rsn_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target subnetwork representations.

        Args:
            context_tokens: (B, K_c, encoder_dim) context subnetwork tokens.
            context_rsn_ids: (B, K_c) RSN indices for context tokens (0-based).
            target_rsn_ids: (B, M) RSN indices for target tokens (0-based).

        Returns:
            Predicted target tokens, shape (B, M, encoder_dim).
        """
        B, K_c, _ = context_tokens.shape
        M = target_rsn_ids.shape[1]

        # Project context to predictor dimension + add identity embeddings
        ctx = self.input_proj(context_tokens)  # (B, K_c, P)
        ctx = ctx + self.rsn_embed(context_rsn_ids)  # (B, K_c, P)

        # Build target query tokens: mask_token + identity embedding.
        # The RSN embedding is detached so the shortcut gradient path
        # (predict a fixed per-RSN direction → loss=0) is cut off.
        # The predictor still knows which RSN to predict, but can't
        # optimise the embedding specifically for that shortcut.
        mask = self.mask_token.unsqueeze(0).expand(B, M, -1)          # (B, M, P)
        mask = mask + self.rsn_embed(target_rsn_ids).detach()          # (B, M, P)

        # Concatenate and run Transformer
        seq = torch.cat([ctx, mask], dim=1)  # (B, K_c + M, P)
        seq = self.transformer(seq)
        seq = self.norm(seq)

        # Extract target positions and project back to encoder dim
        pred = seq[:, K_c:]          # (B, M, P)
        return self.output_proj(pred)  # (B, M, encoder_dim)
