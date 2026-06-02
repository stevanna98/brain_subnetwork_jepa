"""Top-level BS-JEPA model: context encoder + target encoder (EMA) + predictor.

Forward pass overview
---------------------
1. For each subject in the batch, extract the context subgraph (induced on
   context-RSN nodes) and run the *context encoder* → node embeddings.
2. Pool context node embeddings per context RSN → context tokens z_ctx.
3. Run the *target encoder* (EMA copy, no-grad) on the **full graph** → node
   embeddings for all N regions; index target-RSN nodes from this embedding.
4. Pool target node embeddings per target RSN → target tokens z_tgt.
5. Run the *predictor* on z_ctx + mask tokens → predicted z_hat.
6. Return (z_hat, z_tgt) for the loss.

The EMA update is performed externally by the :class:`~brain_jepa.training.ema.EMAUpdater`.
"""

from __future__ import annotations

import copy
import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from ..data.atlas import AtlasMapping
from ..masking.subnetwork_masking import MaskOutput, extract_subgraph
from .encoders import GCNEncoder, GraphTransformerEncoder
from .pooling import PoolingMode, build_pooling
from .predictor import SubnetworkPredictor

logger = logging.getLogger(__name__)

EncoderType = Literal["gcn", "graph_transformer"]


class BSJEPA(nn.Module):
    """Brain Subnetwork JEPA model.

    Args:
        context_encoder: GNN that processes the context subgraph.
        predictor: Narrow Transformer predicting target representations.
        pooling: Subnetwork pooling module (shared by both encoders).
        atlas: Atlas with region-to-RSN mapping.
        include_cross_edges: Passed to :func:`~brain_jepa.masking.extract_subgraph`.
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        predictor: SubnetworkPredictor,
        pooling: nn.Module,
        atlas: AtlasMapping,
        include_cross_edges: bool = False,
    ) -> None:
        super().__init__()
        self.context_encoder = context_encoder
        self.predictor = predictor
        self.pooling = pooling
        self.atlas = atlas
        self.include_cross_edges = include_cross_edges

        # Target encoder: EMA copy — no gradient flow through it
        self.target_encoder = copy.deepcopy(context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _encode_target(self, data: Data) -> torch.Tensor:
        # Full graph — matches I-JEPA: target encoder sees all nodes so target
        # representations capture inter-network context, not just intra-RSN structure.
        return self.target_encoder(data)

    def _encode_context(self, data: Data, node_mask: torch.Tensor) -> tuple[torch.Tensor, Data]:
        sub = extract_subgraph(data, node_mask, include_cross_edges=self.include_cross_edges)
        return self.context_encoder(sub), sub

    def forward(
        self,
        batch: Batch,
        masks: MaskOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the full BS-JEPA forward pass.

        Args:
            batch: Collated PyG Batch of B subjects.
            masks: Subnetwork mask assignments from the collator.

        Returns:
            z_hat: Predicted target tokens, shape (B, M, d).
            z_tgt: Stop-gradient target tokens, shape (B, M, d).
        """
        data_list = batch.to_data_list()
        B = len(data_list)

        ctx_tokens_list: list[torch.Tensor] = []
        tgt_tokens_list: list[torch.Tensor] = []

        for b in range(B):
            data_b = data_list[b]
            ctx_mask = masks.context_node_masks[b]
            tgt_mask = masks.target_node_masks[b]
            ctx_rsns = masks.context_rsn_ids[b]
            tgt_rsns = masks.target_rsn_ids[b]

            # Context branch
            ctx_emb, ctx_sub = self._encode_context(data_b, ctx_mask)  # (N_ctx, d)
            ctx_rsn_ids = data_b.rsn_ids[ctx_sub.original_indices]
            ctx_tokens = self.pooling(ctx_emb, ctx_rsn_ids, ctx_rsns)  # (K_c, d)
            ctx_tokens_list.append(ctx_tokens)

            # Target branch (no grad, target encoder sees full graph)
            tgt_emb_full = self._encode_target(data_b)          # (N, d)
            tgt_emb = tgt_emb_full[tgt_mask]                    # (N_tgt, d)
            tgt_rsn_ids = data_b.rsn_ids[tgt_mask]
            # Layer-normalise target embeddings (mirrors I-JEPA)
            tgt_emb = F.layer_norm(tgt_emb, (tgt_emb.shape[-1],))
            tgt_tokens = self.pooling(tgt_emb, tgt_rsn_ids, tgt_rsns)  # (M, d)
            tgt_tokens_list.append(tgt_tokens)

        ctx_tokens_batch = torch.stack(ctx_tokens_list)  # (B, K_c, d)
        tgt_tokens_batch = torch.stack(tgt_tokens_list)  # (B, M, d)

        z_hat = self.predictor(
            ctx_tokens_batch,
            masks.context_rsn_ids,
            masks.target_rsn_ids,
        )  # (B, M, d)

        return z_hat, tgt_tokens_batch.detach()


def build_bsjepa(
    atlas: AtlasMapping,
    encoder_type: EncoderType = "gcn",
    in_channels: int = 64,
    encoder_hidden: int = 256,
    encoder_out: int = 512,
    encoder_layers: int = 4,
    encoder_heads: int = 4,
    encoder_dropout: float = 0.0,
    pooling_mode: PoolingMode = "mean",
    predictor_dim: int = 384,
    predictor_depth: int = 6,
    predictor_heads: int = 6,
    predictor_dropout: float = 0.0,
    include_cross_edges: bool = False,
    region_positional_encoding: bool = False,
) -> BSJEPA:
    """Factory function — constructs a :class:`BSJEPA` from config parameters."""
    num_regions = atlas.num_regions if region_positional_encoding else None
    if encoder_type == "gcn":
        encoder = GCNEncoder(
            in_channels=in_channels,
            hidden_channels=encoder_hidden,
            out_channels=encoder_out,
            num_layers=encoder_layers,
            dropout=encoder_dropout,
            num_regions=num_regions,
        )
    elif encoder_type == "graph_transformer":
        encoder = GraphTransformerEncoder(
            in_channels=in_channels,
            hidden_channels=encoder_hidden,
            out_channels=encoder_out,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            dropout=encoder_dropout,
            num_regions=num_regions,
        )
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type!r}")

    pooling = build_pooling(pooling_mode, encoder_out, atlas.num_rsns)
    predictor = SubnetworkPredictor(
        encoder_dim=encoder_out,
        predictor_dim=predictor_dim,
        num_rsns=atlas.num_rsns,
        depth=predictor_depth,
        num_heads=predictor_heads,
        dropout=predictor_dropout,
    )
    return BSJEPA(
        context_encoder=encoder,
        predictor=predictor,
        pooling=pooling,
        atlas=atlas,
        include_cross_edges=include_cross_edges,
    )
