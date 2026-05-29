"""Functional connectivity → PyG graph construction."""

from __future__ import annotations

from typing import Literal

import torch
from torch_geometric.data import Data

FCStrategy = Literal["dense", "top_k", "absolute_threshold", "fisher_z_then_threshold"]


def pearson_correlation(time_series: torch.Tensor) -> torch.Tensor:
    """Compute Pearson FC matrix from a (N, T) time-series tensor.

    Returns an (N, N) correlation matrix with diagonal = 1.
    """
    x = time_series - time_series.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
    x = x / std
    return (x @ x.T) / time_series.shape[1]


def _top_k_adjacency(fc: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k strongest positive connections per node; result is symmetrised."""
    N = fc.shape[0]
    adj = torch.zeros_like(fc)
    _, indices = fc.abs().topk(min(k, N - 1), dim=1)
    for i in range(N):
        adj[i, indices[i]] = fc[i, indices[i]]
    # Discard negative edges before symmetrising
    adj = adj.clamp(min=0.0)
    adj = (adj + adj.T) / 2
    return adj


def _fisher_z(fc: torch.Tensor) -> torch.Tensor:
    """Apply the Fisher-z (atanh) transform, clamping to avoid infinities."""
    return torch.atanh(fc.clamp(-0.9999, 0.9999))


def build_graph(
    fc_matrix: torch.Tensor,
    strategy: FCStrategy = "top_k",
    top_k: int = 10,
    threshold: float = 0.2,
    self_loops: bool = False,
) -> Data:
    """Convert an (N, N) FC matrix to a :class:`~torch_geometric.data.Data` graph.

    Args:
        fc_matrix: Symmetric (N, N) functional connectivity matrix.
        strategy: Edge selection strategy.
            - ``"dense"``: keep all edges (weighted adjacency).
            - ``"top_k"``: keep the *top_k* strongest connections per node.
            - ``"absolute_threshold"``: keep edges where |FC| ≥ *threshold*.
            - ``"fisher_z_then_threshold"``: apply Fisher-z transform then threshold.
        top_k: Used only when *strategy* = ``"top_k"``.
        threshold: Used for threshold-based strategies.
        self_loops: If False, remove diagonal edges.

    Returns:
        A PyG ``Data`` object with ``edge_index`` (2, E) and ``edge_attr`` (E, 1).
    """
    N = fc_matrix.shape[0]
    assert fc_matrix.shape == (N, N), "fc_matrix must be square"

    fc = fc_matrix.clone().float()
    # Zero diagonal to avoid self-loops in threshold/top_k paths
    if not self_loops:
        fc.fill_diagonal_(0.0)

    if strategy == "dense":
        adj = fc
    elif strategy == "top_k":
        adj = _top_k_adjacency(fc, top_k)
    elif strategy == "absolute_threshold":
        adj = fc * (fc.abs() >= threshold)
    elif strategy == "fisher_z_then_threshold":
        fc_z = _fisher_z(fc)
        adj = fc_z * (fc_z.abs() >= threshold)
    else:
        raise ValueError(f"Unknown FC strategy: {strategy!r}")

    # Build sparse edge list from non-zero entries
    src, dst = adj.nonzero(as_tuple=True)
    edge_attr = adj[src, dst].unsqueeze(1)
    edge_index = torch.stack([src, dst], dim=0)

    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=N)
