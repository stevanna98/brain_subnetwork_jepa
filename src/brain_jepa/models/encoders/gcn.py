"""GCN / GraphSAGE encoder backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """Multi-layer GraphSAGE encoder producing node-level embeddings.

    Args:
        in_channels: Input feature dimension F.
        hidden_channels: Hidden layer width.
        out_channels: Output embedding dimension d.
        num_layers: Number of message-passing layers.
        dropout: Dropout rate applied between layers.
        normalize: If True, L2-normalise output node embeddings.
        num_regions: If set, adds a learnable anatomical embedding per brain
            region so the encoder knows which region each node is, not just
            what its features are.  Uses ``data.original_indices`` when
            processing subgraphs (set by the mask collator).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        dropout: float = 0.0,
        normalize: bool = False,
        num_regions: int | None = None,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.dropout = dropout

        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.region_embed = (
            nn.Embedding(num_regions, hidden_channels) if num_regions is not None else None
        )

        dims = [hidden_channels] * num_layers + [out_channels]
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1], add_self_loops=False) for i in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(dims[i + 1]) for i in range(num_layers)]
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Encode a (sub)graph.

        Args:
            data: PyG Data with ``x`` (N, F), ``edge_index`` (2, E), and
                  optionally ``original_indices`` (N,) for subgraphs.

        Returns:
            Node embeddings of shape (N, d).
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr.squeeze(-1) if data.edge_attr is not None else None
        x = self.input_proj(x)
        if self.region_embed is not None:
            node_ids = getattr(
                data, "original_indices",
                torch.arange(data.num_nodes, device=x.device),
            )
            x = x + self.region_embed(node_ids)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            x = conv(x, edge_index, edge_weight)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x
