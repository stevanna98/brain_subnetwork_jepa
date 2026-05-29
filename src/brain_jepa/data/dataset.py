"""Brain fMRI dataset — loads per-subject files and builds PyG graphs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .atlas import AtlasMapping
from .connectivity import FCStrategy, build_graph, pearson_correlation
from .transforms import FeatureMode, build_feature_module

logger = logging.getLogger(__name__)


class BrainDataset(Dataset):
    """Dataset of resting-state fMRI subjects.

    Each subject file is a ``.npz`` or ``.pt`` file containing at least one of:
    - ``time_series``: region-level BOLD time series, shape (N, T). FC is computed
      from this and a feature module reduces it to (N, F).
    - ``X``: time-series matrix stored as node features, shape (N, T). FC is
      computed directly from it via Pearson correlation. An explicit
      ``fc_matrix`` key (N, N) may also be provided to override FC computation.

    When ``in_channels`` is known at construction time the feature module is
    initialised eagerly, which is required for correctness when ``num_workers>0``
    in the DataLoader (lazy init would produce different random weights in each
    worker process).
    """

    def __init__(
        self,
        subject_files: list[Path | str],
        atlas: AtlasMapping,
        feature_mode: FeatureMode = "passthrough",
        feature_dim: int = 64,
        fc_strategy: FCStrategy = "top_k",
        top_k: int = 10,
        threshold: float = 0.2,
        in_channels: int | None = None,
    ) -> None:
        self.files = [Path(p) for p in subject_files]
        self.atlas = atlas
        self.fc_strategy = fc_strategy
        self.top_k = top_k
        self.threshold = threshold
        self.feature_dim = feature_dim
        self.feature_mode = feature_mode
        # Eagerly init when in_channels is known so all DataLoader workers share
        # the same weights rather than each creating their own random copy.
        self._feature_module: torch.nn.Module | None = (
            build_feature_module(feature_mode, in_channels, feature_dim)
            if in_channels is not None
            else None
        )

    def _get_feature_module(self, in_channels: int) -> torch.nn.Module:
        if self._feature_module is None:
            # Deterministic seed so workers that fork after this point all
            # produce identical weights; avoids per-worker random divergence.
            with torch.random.fork_rng():
                torch.manual_seed(0)
                self._feature_module = build_feature_module(
                    self.feature_mode, in_channels, self.feature_dim
                )
        return self._feature_module

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        path = self.files[idx]
        raw = _load_subject(path)

        if "time_series" in raw:
            ts = torch.as_tensor(raw["time_series"], dtype=torch.float32)
            fc_matrix = pearson_correlation(ts)
            feat_module = self._get_feature_module(ts.shape[1])
            with torch.no_grad():
                x = feat_module(ts)
        elif "X" in raw:
            x = torch.as_tensor(raw["X"], dtype=torch.float32)
            if "fc_matrix" in raw:
                fc_matrix = torch.as_tensor(raw["fc_matrix"], dtype=torch.float32)
            else:
                # Treat X as (N, T) — pearson_correlation expects (N, T) → (N, N)
                fc_matrix = pearson_correlation(x) if x.shape[1] > 1 else torch.eye(x.shape[0])
        else:
            raise KeyError(f"Subject file {path} must contain 'time_series' or 'X'")

        graph = build_graph(
            fc_matrix,
            strategy=self.fc_strategy,
            top_k=self.top_k,
            threshold=self.threshold,
        )
        graph.x = x
        graph.rsn_ids = self.atlas.rsn_ids.clone()

        metadata = raw.get("metadata", {})
        if metadata:
            graph.subject_id = str(metadata.get("subject_id", idx))

        return graph


def _load_subject(path: Path) -> dict:
    if path.suffix == ".npz":
        raw = np.load(path, allow_pickle=True)
        return {k: raw[k] for k in raw.files}
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu")
    raise ValueError(f"Unsupported file format: {path.suffix!r}. Use .npz or .pt")


class SyntheticBrainDataset(Dataset):
    """Generates synthetic brain graphs for smoke tests and development.

    No real fMRI data required. Each sample has random node features and
    a random FC matrix used to build a graph with the specified strategy.
    """

    def __init__(
        self,
        atlas: AtlasMapping,
        num_subjects: int = 32,
        feature_dim: int = 64,
        time_points: int = 200,
        fc_strategy: FCStrategy = "top_k",
        top_k: int = 10,
        seed: int = 0,
    ) -> None:
        self.atlas = atlas
        self.num_subjects = num_subjects
        self.feature_dim = feature_dim
        self.time_points = time_points
        self.fc_strategy = fc_strategy
        self.top_k = top_k
        self.seed = seed

    def __len__(self) -> int:
        return self.num_subjects

    def __getitem__(self, idx: int) -> Data:
        rng = torch.Generator()
        rng.manual_seed(self.seed + idx)
        N = self.atlas.num_regions

        x = torch.randn(N, self.feature_dim, generator=rng)
        ts = torch.randn(N, self.time_points, generator=rng)
        fc_matrix = pearson_correlation(ts)

        graph = build_graph(fc_matrix, strategy=self.fc_strategy, top_k=self.top_k)
        graph.x = x
        graph.rsn_ids = self.atlas.rsn_ids.clone()
        return graph
