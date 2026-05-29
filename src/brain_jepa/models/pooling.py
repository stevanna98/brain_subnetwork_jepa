"""Subnetwork pooling — aggregates node embeddings into RSN-level tokens."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

PoolingMode = Literal["mean", "attention"]


def build_pooling(mode: PoolingMode, embed_dim: int, num_rsns: int) -> nn.Module:
    if mode == "mean":
        return MeanPooling()
    if mode == "attention":
        return AttentionPooling(embed_dim)
    raise ValueError(f"Unknown pooling mode: {mode!r}")


class MeanPooling(nn.Module):
    """Average node embeddings within each subnetwork.

    Input:
        node_emb: (N, d) node-level embeddings.
        rsn_ids:  (N,) 0-indexed RSN assignment for each node.
        target_rsns: sequence of RSN indices to pool.

    Returns:
        Tensor of shape (|target_rsns|, d) — one token per requested RSN.
    """

    def forward(
        self,
        node_emb: torch.Tensor,
        rsn_ids: torch.Tensor,
        target_rsns: torch.Tensor,
    ) -> torch.Tensor:
        tokens = []
        for k in target_rsns:
            mask = rsn_ids == k
            tokens.append(node_emb[mask].mean(dim=0))
        return torch.stack(tokens)


class AttentionPooling(nn.Module):
    """Soft attention pooling over nodes within each subnetwork.

    A single linear layer scores each node; scores are softmax-normalised
    within each subnetwork before weighting the node embeddings.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(
        self,
        node_emb: torch.Tensor,
        rsn_ids: torch.Tensor,
        target_rsns: torch.Tensor,
    ) -> torch.Tensor:
        tokens = []
        for k in target_rsns:
            mask = rsn_ids == k
            emb_k = node_emb[mask]             # (n_k, d)
            weights = F.softmax(self.score(emb_k), dim=0)  # (n_k, 1)
            tokens.append((weights * emb_k).sum(dim=0))    # (d,)
        return torch.stack(tokens)
