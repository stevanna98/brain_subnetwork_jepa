"""Top-level BS-JEPA model: context encoder + target encoder (EMA) + predictor.

Forward pass overview
---------------------
1. For each subject in the batch, extract the context subgraph and run the
   *context encoder* → node embeddings, then pool per context RSN → (K_c, d).
2. Run the *target encoder* (EMA copy, no-grad) on the full graph → (N, d);
   index target-RSN nodes → (N_tgt, d).  No pooling — node-level targets.
3. Run the *predictor* with context tokens + one mask token per target NODE
   → (N_tgt, d) predicted node embeddings.
4. Concatenate across subjects and return (N_total, d) pairs for the loss.

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
        feature_extractor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.context_encoder = context_encoder
        self.predictor = predictor
        self.pooling = pooling
        self.atlas = atlas
        self.include_cross_edges = include_cross_edges
        # Trained end-to-end via context path; shared between context and target
        self.feature_extractor = feature_extractor

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

    @torch.no_grad()
    def encode(self, data: Data) -> torch.Tensor:
        """Feature extractor + target encoder on a single graph; used at eval time.

        Returns node embeddings of shape (N, d).
        """
        if self.feature_extractor is not None:
            data = data.clone()
            data.x = self.feature_extractor(data.x)
        return self.target_encoder(data)

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
            z_hat: Predicted node embeddings, shape (N_total, d).
            z_tgt: Stop-gradient target node embeddings, shape (N_total, d).
                   N_total = sum of target-RSN nodes across all subjects in batch.
        """
        data_list = batch.to_data_list()
        B = len(data_list)

        # Apply the feature extractor once per subject (with gradients — it is
        # trained via the context path; target path is stopped by .detach() later)
        if self.feature_extractor is not None:
            for data_b in data_list:
                data_b.x = self.feature_extractor(data_b.x)

        z_hat_list: list[torch.Tensor] = []
        z_tgt_list: list[torch.Tensor] = []

        for b in range(B):
            data_b = data_list[b]
            ctx_mask = masks.context_node_masks[b]
            tgt_mask = masks.target_node_masks[b]
            ctx_rsns = masks.context_rsn_ids[b]

            # Context branch: pool per RSN → (K_c, d) context tokens
            ctx_emb, ctx_sub = self._encode_context(data_b, ctx_mask)
            ctx_rsn_ids = data_b.rsn_ids[ctx_sub.original_indices]
            ctx_tokens = self.pooling(ctx_emb, ctx_rsn_ids, ctx_rsns)  # (K_c, d)

            # Target branch: node-level, no pooling → (N_tgt, d)
            tgt_emb_full = self._encode_target(data_b)      # (N, d)
            tgt_emb = tgt_emb_full[tgt_mask]                # (N_tgt, d)
            tgt_node_rsn_ids = data_b.rsn_ids[tgt_mask]    # (N_tgt,) RSN per node
            tgt_emb = F.layer_norm(tgt_emb, (tgt_emb.shape[-1],))

            # Predict one embedding per target node
            z_hat_b = self.predictor(ctx_tokens, ctx_rsns, tgt_node_rsn_ids)  # (N_tgt, d)

            z_hat_list.append(z_hat_b)
            z_tgt_list.append(tgt_emb)

        # Concatenate across subjects: (N_total, d)
        z_hat = torch.cat(z_hat_list, dim=0)
        z_tgt = torch.cat(z_tgt_list, dim=0)

        return z_hat, z_tgt.detach()


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
    feature_extractor: nn.Module | None = None,
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
        feature_extractor=feature_extractor,
    )
