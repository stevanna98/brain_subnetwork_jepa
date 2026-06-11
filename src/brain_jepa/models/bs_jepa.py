"""Top-level BS-JEPA model: context encoder + target encoder (EMA) + predictor.

Forward pass overview
---------------------
1. For each subject in the batch, extract the induced subgraph on context-RSN nodes
   (target nodes are dropped entirely) and run the *context encoder* → (N_ctx, d).
2. Run the *target encoder* (EMA copy, no-grad) on the full graph → (N, d);
   index target-RSN nodes → (N_tgt, d).  No pooling.
3. Run the *predictor* with all context node embeddings + one mask token per
   target node → (N_tgt, d) predicted node embeddings.
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
        atlas: AtlasMapping,
        feature_extractor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.context_encoder = context_encoder
        self.predictor = predictor
        self.atlas = atlas
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

    def _encode_context(self, data: Data, ctx_mask: torch.Tensor) -> torch.Tensor:
        """Extract the context subgraph (dropping target nodes) and encode it.

        Returns context node embeddings (N_ctx, d).
        """
        ctx_graph = extract_subgraph(data, ctx_mask)
        return self.context_encoder(ctx_graph)  # (N_ctx, d)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full BS-JEPA forward pass.

        Args:
            batch: Collated PyG Batch of B subjects.
            masks: Subnetwork mask assignments from the collator.

        Returns:
            z_hat:    Predicted node embeddings, shape (N_tgt_total, d).
            z_tgt:    Stop-gradient target node embeddings, shape (N_tgt_total, d).
            ctx_embs: Context node embeddings, shape (N_ctx_total, d).
                      Has gradients — used for direct variance regularisation on
                      the context encoder without going through the predictor.
        """
        if self.feature_extractor is not None:
            batch.x = self.feature_extractor(batch.x)

        # Region identity for every node of the collated batch (nodes are
        # atlas-ordered within each subject, so node index = region index).
        counts = batch.ptr.diff()
        batch.region_ids = torch.cat(
            [torch.arange(int(c), device=batch.x.device) for c in counts]
        )

        # ---- Target branch: one pass over the full collated batch ----
        tgt_emb_full = self._encode_target(batch)               # (N_total, d)
        global_tgt_mask = torch.cat(masks.target_node_masks)     # (N_total,)
        z_tgt = tgt_emb_full[global_tgt_mask]                    # (N_tgt_total, d)
        z_tgt = F.layer_norm(z_tgt, (z_tgt.shape[-1],))

        # ---- Context branch: per-subject subgraphs, encoded as one batch ----
        data_list = batch.to_data_list()
        ctx_graphs: list[Data] = []
        ctx_rsn_list: list[torch.Tensor] = []
        ctx_region_list: list[torch.Tensor] = []
        tgt_rsn_list: list[torch.Tensor] = []
        tgt_region_list: list[torch.Tensor] = []

        for b, data_b in enumerate(data_list):
            ctx_mask = masks.context_node_masks[b]
            tgt_mask = masks.target_node_masks[b]
            ctx_graphs.append(extract_subgraph(data_b, ctx_mask))
            ctx_rsn_list.append(data_b.rsn_ids[ctx_mask])
            ctx_region_list.append(ctx_mask.nonzero(as_tuple=True)[0])
            tgt_rsn_list.append(data_b.rsn_ids[tgt_mask])
            tgt_region_list.append(tgt_mask.nonzero(as_tuple=True)[0])

        ctx_batch = Batch.from_data_list(ctx_graphs)
        ctx_embs = self.context_encoder(ctx_batch)               # (N_ctx_total, d)
        ctx_sizes = [g.num_nodes for g in ctx_graphs]
        ctx_split = list(ctx_embs.split(ctx_sizes, dim=0))

        # ---- Predictor: single padded Transformer call over all subjects ----
        z_hat_list = self.predictor.forward_batched(
            ctx_split, ctx_rsn_list, ctx_region_list,
            tgt_rsn_list, tgt_region_list,
        )
        z_hat = torch.cat(z_hat_list, dim=0)       # (N_tgt_total, d)

        return z_hat, z_tgt.detach(), ctx_embs


def build_bsjepa(
    atlas: AtlasMapping,
    encoder_type: EncoderType = "gcn",
    in_channels: int = 64,
    encoder_hidden: int = 256,
    encoder_out: int = 512,
    encoder_layers: int = 4,
    encoder_heads: int = 4,
    encoder_dropout: float = 0.0,
    predictor_dim: int = 384,
    predictor_depth: int = 6,
    predictor_heads: int = 6,
    predictor_dropout: float = 0.0,
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

    predictor = SubnetworkPredictor(
        encoder_dim=encoder_out,
        predictor_dim=predictor_dim,
        num_rsns=atlas.num_rsns,
        num_regions=atlas.num_regions,
        depth=predictor_depth,
        num_heads=predictor_heads,
        dropout=predictor_dropout,
    )
    return BSJEPA(
        context_encoder=encoder,
        predictor=predictor,
        atlas=atlas,
        feature_extractor=feature_extractor,
    )
