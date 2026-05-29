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
    ) -> torch.Tensor:
        # Local branch
        h_local = self.mpnn(self.norm1(x), edge_index, edge_weight)
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)

        # Global branch — treat the N nodes as a sequence (batch size = 1)
        h_norm = self.norm2(x).unsqueeze(0)  # (1, N, dim)
        h_global, _ = self.attn(h_norm, h_norm, h_norm)
        h_global = h_global.squeeze(0)
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
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList(
            [GPSLayer(hidden_channels, num_heads, dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        self.norm_out = nn.LayerNorm(out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        """Encode a (sub)graph.

        Args:
            data: PyG Data with ``x`` (N, F), ``edge_index`` (2, E),
                  and optionally ``edge_attr`` (E, 1).

        Returns:
            Node embeddings of shape (N, d).
        """
        x = self.input_proj(data.x)
        edge_index = data.edge_index
        edge_weight = data.edge_attr.squeeze(-1) if data.edge_attr is not None else None

        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)

        x = self.norm_out(self.output_proj(x))
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x
