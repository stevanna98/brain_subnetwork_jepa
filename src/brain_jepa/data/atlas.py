"""Glasser 379-region atlas — loads region-to-subnetwork mapping."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class AtlasMapping:
    """Immutable container for the Glasser-379 → RSN-12 mapping.

    Attributes:
        region_ids: 1-indexed region identifiers, shape (N,).
        region_names: list of region name strings, length N.
        rsn_ids: 0-indexed RSN assignment for each region, shape (N,).
        rsn_names: list of RSN name strings, length K.
        num_regions: N = 379.
        num_rsns: K = 12.
    """

    region_ids: torch.Tensor
    region_names: list[str]
    rsn_ids: torch.Tensor
    rsn_names: list[str]
    num_regions: int
    num_rsns: int

    def regions_in_rsn(self, rsn_idx: int) -> torch.Tensor:
        """Return 0-indexed region indices belonging to *rsn_idx* (0-based)."""
        return (self.rsn_ids == rsn_idx).nonzero(as_tuple=True)[0]

    def rsn_mask_tensor(self) -> torch.Tensor:
        """Return a (K, N) boolean tensor; entry [k, i] is True iff region i ∈ RSN k."""
        K, N = self.num_rsns, self.num_regions
        mask = torch.zeros(K, N, dtype=torch.bool)
        for k in range(K):
            mask[k] = self.rsn_ids == k
        return mask


def load_atlas(csv_path: str | Path) -> AtlasMapping:
    """Parse *csv_path* and return an :class:`AtlasMapping`.

    The CSV must have columns: region_id, region_name, rsn_id, rsn_name.
    RSN ids in the file are 1-indexed; they are stored 0-indexed internally.
    """
    csv_path = Path(csv_path)
    region_ids: list[int] = []
    region_names: list[str] = []
    rsn_ids_raw: list[int] = []
    rsn_name_map: dict[int, str] = {}

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            region_ids.append(int(row["region_id"]))
            region_names.append(row["region_name"])
            raw_rid = int(row["rsn_id"])
            rsn_ids_raw.append(raw_rid)
            rsn_name_map[raw_rid] = row["rsn_name"]

    # Convert to 0-indexed RSN ids
    sorted_rsn_keys = sorted(rsn_name_map)
    raw_to_zero = {raw: idx for idx, raw in enumerate(sorted_rsn_keys)}
    rsn_ids_zero = [raw_to_zero[r] for r in rsn_ids_raw]
    rsn_names = [rsn_name_map[k] for k in sorted_rsn_keys]

    return AtlasMapping(
        region_ids=torch.tensor(region_ids, dtype=torch.long),
        region_names=region_names,
        rsn_ids=torch.tensor(rsn_ids_zero, dtype=torch.long),
        rsn_names=rsn_names,
        num_regions=len(region_ids),
        num_rsns=len(rsn_names),
    )
