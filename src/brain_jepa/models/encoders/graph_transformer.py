"""GPS-style Graph Transformer encoder backbone.

Each layer combines:
  1. Local MPNN (edge-weighted GCN message passing).
  2. Global multi-head self-attention over all nodes.
  3. Feed-forward sub-layer.

References:
    Rampášek et al., "Recipe for a General, Powerful, Scalable Graph Transformer"
    (NeurIPS 2022).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch


class GPSLayer(nn.Module):
    """Single GPS layer: local MPNN + global attention + FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # Local: edge-weighted message passing
        self.mpnn = GCNConv(dim, dim, add_self_loops=False)

        # Global: multi-head self-attention
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # Feed-forward
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Local branch
        h_local = self.mpnn(self.norm1(x), edge_index, edge_weight)
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)

        # Global branch — attention must stay within each graph: for a collated
        # multi-subject batch, the `batch` vector scopes attention per subject.
        h_norm = self.norm2(x)
        if batch is None:
            h_global, _ = self.attn(
                h_norm.unsqueeze(0), h_norm.unsqueeze(0), h_norm.unsqueeze(0)
            )
            h_global = h_global.squeeze(0)
        else:
            dense, node_mask = to_dense_batch(h_norm, batch)  # (B, L, dim), (B, L)
            attn_out, _ = self.attn(
                dense, dense, dense, key_padding_mask=~node_mask
            )
            h_global = attn_out[node_mask]                    # back to (N, dim)
        h_global = F.dropout(h_global, p=self.dropout, training=self.training)

        x = x + h_local + h_global

        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


class GraphTransformerEncoder(nn.Module):
    """GPS-style Graph Transformer producing node-level embeddings.

    Args:
        in_channels: Input feature dimension F.
        hidden_channels: Width of GPS layers.
        out_channels: Output embedding dimension d.
        num_layers: Number of GPS layers.
        num_heads: Attention heads in each GPS layer.
        dropout: Dropout rate.
        normalize: If True, L2-normalise output node embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
        normalize: bool = False,
        num_regions: int | None = None,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.region_embed = (
            nn.Embedding(num_regions, hidden_channels) if num_regions is not None else None
        )
        self.layers = nn.ModuleList(
            [GPSLayer(hidden_channels, num_heads, dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        self.norm_out = nn.LayerNorm(out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        """Encode a (sub)graph.

        Args:
            data: PyG Data with ``x`` (N, F), ``edge_index`` (2, E),
                  optionally ``edge_attr`` (E, 1), and optionally
                  ``original_indices`` (N,) for subgraphs.

        Returns:
            Node embeddings of shape (N, d).
        """
        x = self.input_proj(data.x)
        if self.region_embed is not None:
            # region_ids (not original_indices): PyG batching offsets any
            # attribute whose name contains "index", which would corrupt
            # atlas-region lookups for batched subgraphs.
            node_ids = getattr(data, "region_ids", None)
            if node_ids is None:
                node_ids = torch.arange(data.num_nodes, device=x.device)
            x = x + self.region_embed(node_ids)
        edge_index = data.edge_index
        edge_weight = data.edge_attr.squeeze(-1) if data.edge_attr is not None else None
        batch = getattr(data, "batch", None)

        for layer in self.layers:
            x = layer(x, edge_index, edge_weight, batch)

        x = self.norm_out(self.output_proj(x))
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x
