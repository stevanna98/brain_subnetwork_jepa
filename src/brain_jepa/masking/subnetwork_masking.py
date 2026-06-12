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

    def to(self, device: torch.device) -> "MaskOutput":
        """Move all mask tensors to *device* (returns self)."""
        self.context_rsn_ids = self.context_rsn_ids.to(device)
        self.target_rsn_ids = self.target_rsn_ids.to(device)
        self.context_node_masks = [m.to(device) for m in self.context_node_masks]
        self.target_node_masks = [m.to(device) for m in self.target_node_masks]
        return self


class SubnetworkMaskCollator:
    """Collates a list of PyG :class:`~torch_geometric.data.Data` graphs and
    produces subnetwork-level context/target splits.

    Args:
        num_rsns: Total number of subnetworks K (default 12).
        num_targets: Number of target subnetworks M to mask (default 1).
        extra_target_ratio: Fraction of the *remaining* (non-target-RSN) nodes
            that are additionally masked, sampled uniformly per subject. With
            only K possible RSN-level masks the prediction task has K variants
            and a large predictor can memorise them; random extra nodes make
            every sample a distinct task while keeping the subnetwork-level
            core. 0.0 disables (pure RSN masking).
        include_cross_edges_in_context: If True, context subgraph retains
            edges originally incident to target nodes (dangling edges are
            dropped at the GNN level since target nodes are absent). False
            matches I-JEPA's strict no-overlap convention.
    """

    def __init__(
        self,
        num_rsns: int = 12,
        num_targets: int = 1,
        extra_target_ratio: float = 0.0,
        include_cross_edges_in_context: bool = False,
    ) -> None:
        assert 1 <= num_targets < num_rsns, (
            f"num_targets must be in [1, num_rsns-1], got {num_targets}"
        )
        assert 0.0 <= extra_target_ratio < 1.0, (
            f"extra_target_ratio must be in [0, 1), got {extra_target_ratio}"
        )
        self.num_rsns = num_rsns
        self.num_targets = num_targets
        self.extra_target_ratio = extra_target_ratio
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
            tgt_mask = torch.isin(rsn_ids, target_rsn_ids[b])
            if self.extra_target_ratio > 0.0:
                candidates = (~tgt_mask).nonzero(as_tuple=True)[0]
                n_extra = int(round(self.extra_target_ratio * candidates.numel()))
                if n_extra > 0:
                    extra = candidates[torch.randperm(candidates.numel())[:n_extra]]
                    tgt_mask = tgt_mask.clone()
                    tgt_mask[extra] = True
            ctx_mask = ~tgt_mask
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
    device = node_mask.device
    original_indices = node_mask.nonzero(as_tuple=True)[0]
    x_sub = data.x[original_indices]

    old_to_new = torch.full((data.num_nodes,), -1, dtype=torch.long, device=device)
    old_to_new[original_indices] = torch.arange(len(original_indices), device=device)

    src, dst = data.edge_index
    # Always require both endpoints to be in the subgraph: cross-edges would
    # produce old_to_new=-1 for the absent endpoint, yielding invalid indices.
    # include_cross_edges is kept as a parameter for API compatibility but
    # the safe behaviour is identical in both cases.
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
    # Same values under a name PyG batching leaves untouched: attributes whose
    # name contains "index" (like original_indices) get offset per graph by
    # Batch.from_data_list, which would corrupt atlas-region lookups.
    sub.region_ids = original_indices
    return sub
