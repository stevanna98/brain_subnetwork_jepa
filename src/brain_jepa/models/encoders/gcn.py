"""GCN / GraphSAGE encoder backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class GCNEncoder(nn.Module):
    """Multi-layer GraphSAGE encoder producing node-level embeddings.

    Args:
        in_channels: Input feature dimension F.
        hidden_channels: Hidden layer width.
        out_channels: Output embedding dimension d.
        num_layers: Number of message-passing layers.
        dropout: Dropout rate applied between layers.
        normalize: If True, L2-normalise output node embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        dropout: float = 0.0,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.dropout = dropout

        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(dims[i + 1]) for i in range(num_layers)]
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Encode a (sub)graph.

        Args:
            data: PyG Data with ``x`` (N, F), ``edge_index`` (2, E).

        Returns:
            Node embeddings of shape (N, d).
        """
        x, edge_index = data.x, data.edge_index
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x
