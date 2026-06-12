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
        num_regions: Number of atlas regions (one identity embedding per node).
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
        num_regions: int = 379,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if predictor_dim % num_heads != 0:
            raise ValueError(
                f"predictor_dim ({predictor_dim}) must be divisible by predictor_heads "
                f"({num_heads}); got remainder {predictor_dim % num_heads}."
            )
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.num_rsns = num_rsns

        self.input_proj = nn.Linear(encoder_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, predictor_dim))
        # Per-RSN identity embedding used for both context and target tokens
        self.rsn_embed = nn.Embedding(num_rsns, predictor_dim)
        # Per-region identity embedding (analog of I-JEPA's positional embedding
        # on mask tokens). Without it, all mask queries for one target RSN are
        # identical tokens and self-attention maps them to identical outputs —
        # the predictor can then only emit a single shared vector per RSN.
        self.region_embed = nn.Embedding(num_regions, predictor_dim)

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
        nn.init.normal_(self.region_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_tokens(
        self,
        context_tokens: torch.Tensor,
        context_rsn_ids: torch.Tensor,
        context_region_ids: torch.Tensor,
        target_node_rsn_ids: torch.Tensor,
        target_region_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the (N_ctx + N_tgt, P) token sequence for one subject."""
        # Context: project + add RSN and region identities
        ctx = self.input_proj(context_tokens)               # (N_ctx, P)
        ctx = ctx + self.rsn_embed(context_rsn_ids)         # (N_ctx, P)
        ctx = ctx + self.region_embed(context_region_ids)   # (N_ctx, P)

        # Target: one mask token per node, tagged with its RSN and region
        # identities so each query is unique. Identity embeddings are detached
        # on the target side to cut the gradient path that would let the
        # predictor tune them into per-node answer codes; they still train
        # through the context side.
        n_tgt = target_node_rsn_ids.shape[0]
        mask = self.mask_token.expand(n_tgt, -1)                        # (N_tgt, P)
        mask = mask + self.rsn_embed(target_node_rsn_ids).detach()      # (N_tgt, P)
        mask = mask + self.region_embed(target_region_ids).detach()     # (N_tgt, P)

        return torch.cat([ctx, mask], dim=0)

    def forward_batched(
        self,
        context_tokens: list[torch.Tensor],
        context_rsn_ids: list[torch.Tensor],
        context_region_ids: list[torch.Tensor],
        target_node_rsn_ids: list[torch.Tensor],
        target_region_ids: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Predict node-level target representations for a batch of subjects.

        Each argument is a length-B list of per-subject tensors (shapes as in
        :meth:`forward`). Sequences are padded to the batch maximum and run
        through the Transformer in a single call with a key-padding mask.

        Returns:
            Length-B list of predicted node embeddings, each (N_tgt_b, encoder_dim).
        """
        B = len(context_tokens)
        seqs = [
            self._build_tokens(
                context_tokens[b], context_rsn_ids[b], context_region_ids[b],
                target_node_rsn_ids[b], target_region_ids[b],
            )
            for b in range(B)
        ]
        n_ctx = [context_tokens[b].shape[0] for b in range(B)]
        n_tgt = [target_node_rsn_ids[b].shape[0] for b in range(B)]

        L = max(s.shape[0] for s in seqs)
        padded = seqs[0].new_zeros(B, L, self.predictor_dim)
        pad_mask = torch.ones(B, L, dtype=torch.bool, device=padded.device)
        for b, s in enumerate(seqs):
            padded[b, : s.shape[0]] = s
            pad_mask[b, : s.shape[0]] = False

        out = self.transformer(padded, src_key_padding_mask=pad_mask)
        out = self.norm(out)
        return [
            self.output_proj(out[b, n_ctx[b] : n_ctx[b] + n_tgt[b]])
            for b in range(B)
        ]

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_rsn_ids: torch.Tensor,
        context_region_ids: torch.Tensor,
        target_node_rsn_ids: torch.Tensor,
        target_region_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict node-level target representations for one subject.

        Args:
            context_tokens:      (N_ctx, encoder_dim) context node embeddings.
            context_rsn_ids:     (N_ctx,) RSN index for each context node.
            context_region_ids:  (N_ctx,) atlas region index for each context node.
            target_node_rsn_ids: (N_tgt,) RSN index for each target node.
            target_region_ids:   (N_tgt,) atlas region index for each target node.

        Returns:
            Predicted node embeddings, shape (N_tgt, encoder_dim).
        """
        return self.forward_batched(
            [context_tokens], [context_rsn_ids], [context_region_ids],
            [target_node_rsn_ids], [target_region_ids],
        )[0]
