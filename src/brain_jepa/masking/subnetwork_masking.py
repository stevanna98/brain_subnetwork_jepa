"""Subnetwork-level masking — analogous to I-JEPA's multi-block collator.

At each iteration we sample *M* target subnetworks uniformly and designate
the remaining K-M as context.  The masking collator is called on a collated
batch and returns index tensors that downstream modules use to extract
induced subgraphs from each subject's node-feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Batch, Data


@dataclass
class MaskOutput:
    """Per-batch masking output.

    Attributes:
        context_rsn_ids: (B, K-M) tensor of context RSN indices (0-based).
        target_rsn_ids: (B, M) tensor of target RSN indices (0-based).
        context_node_masks: list of length B; each is a boolean (N,) tensor
            selecting context nodes for that subject.
        target_node_masks: list of length B; each is a boolean (N,) tensor
            selecting target nodes for that subject.
    """

    context_rsn_ids: torch.Tensor
    target_rsn_ids: torch.Tensor
    context_node_masks: list[torch.Tensor]
    target_node_masks: list[torch.Tensor]


class SubnetworkMaskCollator:
    """Collates a list of PyG :class:`~torch_geometric.data.Data` graphs and
    produces subnetwork-level context/target splits.

    Args:
        num_rsns: Total number of subnetworks K (default 12).
        num_targets: Number of target subnetworks M to mask (default 1).
        include_cross_edges_in_context: If True, context subgraph retains
            edges originally incident to target nodes (dangling edges are
            dropped at the GNN level since target nodes are absent). False
            matches I-JEPA's strict no-overlap convention.
    """

    def __init__(
        self,
        num_rsns: int = 12,
        num_targets: int = 1,
        include_cross_edges_in_context: bool = False,
    ) -> None:
        assert 1 <= num_targets < num_rsns, (
            f"num_targets must be in [1, num_rsns-1], got {num_targets}"
        )
        self.num_rsns = num_rsns
        self.num_targets = num_targets
        self.include_cross_edges_in_context = include_cross_edges_in_context

    def __call__(self, batch: list[Data]) -> tuple[Batch, MaskOutput]:
        """Collate *batch* and generate masks.

        Returns:
            collated_batch: Standard PyG :class:`~torch_geometric.data.Batch`.
            mask_output: :class:`MaskOutput` with context/target splits.
        """
        B = len(batch)
        collated = Batch.from_data_list(batch)

        context_rsn_ids, target_rsn_ids = self._sample_splits(B, batch[0].rsn_ids.device)

        context_node_masks: list[torch.Tensor] = []
        target_node_masks: list[torch.Tensor] = []

        for b in range(B):
            rsn_ids = batch[b].rsn_ids  # (N,)
            ctx_mask = torch.isin(rsn_ids, context_rsn_ids[b])
            tgt_mask = torch.isin(rsn_ids, target_rsn_ids[b])
            context_node_masks.append(ctx_mask)
            target_node_masks.append(tgt_mask)

        return collated, MaskOutput(
            context_rsn_ids=context_rsn_ids,
            target_rsn_ids=target_rsn_ids,
            context_node_masks=context_node_masks,
            target_node_masks=target_node_masks,
        )

    def _sample_splits(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample target RSNs uniformly; context = complement.

        Returns:
            context_ids: (B, K-M) int64 tensor.
            target_ids:  (B, M)   int64 tensor.
        """
        all_rsns = torch.arange(self.num_rsns, device=device)
        context_list, target_list = [], []
        for _ in range(batch_size):
            perm = torch.randperm(self.num_rsns, device=device)
            tgt = all_rsns[perm[: self.num_targets]]
            ctx = all_rsns[perm[self.num_targets :]]
            target_list.append(tgt)
            context_list.append(ctx)
        return torch.stack(context_list), torch.stack(target_list)


def extract_subgraph(
    data: Data,
    node_mask: torch.Tensor,
    include_cross_edges: bool = False,
) -> Data:
    """Extract the induced subgraph for nodes selected by *node_mask*.

    Args:
        data: Full-subject PyG Data with ``x``, ``edge_index``, ``edge_attr``.
        node_mask: Boolean (N,) tensor; True for nodes to keep.
        include_cross_edges: If True, retain edges where only one endpoint is
            in the subgraph (the missing endpoint's features are excluded).

    Returns:
        A new :class:`~torch_geometric.data.Data` with re-indexed nodes and
        an ``original_indices`` attribute mapping new → old node positions.
    """
    original_indices = node_mask.nonzero(as_tuple=True)[0]
    x_sub = data.x[original_indices]

    old_to_new = torch.full((data.num_nodes,), -1, dtype=torch.long)
    old_to_new[original_indices] = torch.arange(len(original_indices))

    src, dst = data.edge_index
    if include_cross_edges:
        keep = node_mask[src] | node_mask[dst]
    else:
        keep = node_mask[src] & node_mask[dst]

    edge_index_sub = old_to_new[data.edge_index[:, keep]]
    edge_attr_sub = data.edge_attr[keep] if data.edge_attr is not None else None

    sub = Data(
        x=x_sub,
        edge_index=edge_index_sub,
        edge_attr=edge_attr_sub,
        num_nodes=len(original_indices),
    )
    sub.original_indices = original_indices
    return sub
