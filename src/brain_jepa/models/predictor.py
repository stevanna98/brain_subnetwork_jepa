"""Narrow Transformer predictor for node-level target prediction.

Processes one subject at a time.  Context tokens are pooled RSN representations
(K_c, d).  Target queries are one mask token per target NODE (N_tgt,) tagged
with the RSN identity of each node.  The predictor outputs one embedding per
target node (N_tgt, d), which is compared directly against the target encoder's
node embeddings — no pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SubnetworkPredictor(nn.Module):
    """Narrow Transformer that predicts target node representations.

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

        self.input_proj = nn.Linear(encoder_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, predictor_dim))
        # Per-RSN identity embedding used for both context and target tokens
        self.rsn_embed = nn.Embedding(num_rsns, predictor_dim)

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
        target_node_rsn_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict node-level target representations for one subject.

        Args:
            context_tokens:     (K_c, encoder_dim) pooled context RSN tokens.
            context_rsn_ids:    (K_c,) RSN index for each context token.
            target_node_rsn_ids:(N_tgt,) RSN index for each target NODE.

        Returns:
            Predicted node embeddings, shape (N_tgt, encoder_dim).
        """
        K_c = context_tokens.shape[0]
        N_tgt = target_node_rsn_ids.shape[0]

        # Context: project + add RSN identity
        ctx = self.input_proj(context_tokens)           # (K_c, P)
        ctx = ctx + self.rsn_embed(context_rsn_ids)     # (K_c, P)

        # Target: one mask token per node, tagged with its RSN identity.
        # RSN embedding is detached to cut the gradient path that would allow
        # the predictor to cheat by memorising a fixed per-RSN direction.
        mask = self.mask_token.expand(N_tgt, -1)                        # (N_tgt, P)
        mask = mask + self.rsn_embed(target_node_rsn_ids).detach()      # (N_tgt, P)

        # Run Transformer over context + target tokens (add batch dim)
        seq = torch.cat([ctx, mask], dim=0).unsqueeze(0)  # (1, K_c + N_tgt, P)
        seq = self.transformer(seq).squeeze(0)             # (K_c + N_tgt, P)
        seq = self.norm(seq)

        pred = seq[K_c:]                    # (N_tgt, P)
        return self.output_proj(pred)       # (N_tgt, encoder_dim)
