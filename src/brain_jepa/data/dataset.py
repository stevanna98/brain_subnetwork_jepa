"""Brain fMRI dataset — loads per-subject files and builds PyG graphs."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .atlas import AtlasMapping
from .connectivity import FCStrategy, build_graph, pearson_correlation
from .transforms import FeatureMode, build_feature_module

NodeFeatureType = Literal["bold", "fc_row", "ones"]

logger = logging.getLogger(__name__)


def _zscore(ts: torch.Tensor) -> torch.Tensor:
    """Z-score each region's time series independently (N, T) → (N, T)."""
    mean = ts.mean(dim=1, keepdim=True)
    std = ts.std(dim=1, keepdim=True).clamp(min=1e-8)
    return (ts - mean) / std


class BrainDataset(Dataset):
    """Dataset of resting-state fMRI subjects.

    Each subject file is a ``.npz`` or ``.pt`` file containing at least one of:
    - ``time_series``: region-level BOLD time series, shape (N, T).
    - ``X``: time-series matrix stored as node features, shape (N, T). An explicit
      ``fc_matrix`` key (N, N) may also be provided to override FC computation.

    Args:
        node_feature_type: What to use as node features.
            - ``"bold"``: BOLD time series projected through the feature module → (N, F).
            - ``"fc_row"``: each node's row of the FC matrix → (N, N).
            - ``"ones"``: constant all-ones vector → (N, 1), graph-structure-only baseline.

    When ``in_channels`` is known at construction time the feature module is
    initialised eagerly, which is required for correctness when ``num_workers>0``
    in the DataLoader (lazy init would produce different random weights in each
    worker process).
    """

    def __init__(
        self,
        subject_files: list[Path | str],
        atlas: AtlasMapping,
        node_feature_type: NodeFeatureType = "bold",
        feature_mode: FeatureMode = "passthrough",
        feature_dim: int = 64,
        fc_strategy: FCStrategy = "top_k",
        top_k: int = 10,
        threshold: float = 0.2,
        in_channels: int | None = None,
    ) -> None:
        self.files = [Path(p) for p in subject_files]
        self.atlas = atlas
        self.node_feature_type = node_feature_type
        self.fc_strategy = fc_strategy
        self.top_k = top_k
        self.threshold = threshold
        self.feature_dim = feature_dim
        self.feature_mode = feature_mode
        self._feature_module: torch.nn.Module | None = (
            build_feature_module(feature_mode, in_channels, feature_dim)
            if in_channels is not None and node_feature_type == "bold"
            else None
        )

    def _get_feature_module(self, in_channels: int) -> torch.nn.Module:
        if self._feature_module is None:
            with torch.random.fork_rng(devices=[]):
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
            ts = _zscore(torch.as_tensor(raw["time_series"], dtype=torch.float32))
            fc_matrix = pearson_correlation(ts)
        elif "X" in raw:
            ts = _zscore(torch.as_tensor(raw["X"], dtype=torch.float32))
            if "fc_matrix" in raw:
                fc_matrix = torch.as_tensor(raw["fc_matrix"], dtype=torch.float32)
            else:
                fc_matrix = pearson_correlation(ts) if ts.shape[1] > 1 else torch.eye(ts.shape[0])
        else:
            raise KeyError(f"Subject file {path} must contain 'time_series' or 'X'")

        N = fc_matrix.shape[0]
        if self.node_feature_type == "bold":
            feat_module = self._get_feature_module(ts.shape[1])
            with torch.no_grad():
                x = feat_module(ts)
        elif self.node_feature_type == "fc_row":
            x = fc_matrix  # (N, N)
        elif self.node_feature_type == "ones":
            x = torch.ones(N, 1)
        else:
            raise ValueError(f"Unknown node_feature_type: {self.node_feature_type!r}")

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


def _load_dict_file(path: Path) -> dict:
    """Load a subject-dictionary file regardless of format (.pkl, .pt, .npz)."""
    if path.suffix == ".pkl":
        with path.open("rb") as fh:
            return pickle.load(fh)
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=False)
    if path.suffix == ".npz":
        raw = np.load(path, allow_pickle=True)
        return {k: raw[k] for k in raw.files}
    raise ValueError(f"Unsupported dict file format: {path.suffix!r}. Use .pkl, .pt, or .npz")


def _load_subject(path: Path) -> dict:
    if path.suffix == ".npz":
        raw = np.load(path, allow_pickle=True)
        return {k: raw[k] for k in raw.files}
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu")
    raise ValueError(f"Unsupported file format: {path.suffix!r}. Use .npz or .pt")


class FCDictDataset(Dataset):
    """Dataset backed by a single .pt dictionary file.

    Expected file structure (produced by ``torch.save``)::

        {
            "sub-001": {"BOLD": Tensor(N,T), "FC": Tensor(N,N), "gender": ..., "age": ...},
            "sub-002": {...},
            ...
        }

    Args:
        node_feature_type: What to use as node features.
            - ``"bold"``: BOLD time series projected through the feature module → (N, F).
            - ``"fc_row"``: each node's row of the FC matrix → (N, N).
            - ``"ones"``: constant all-ones vector → (N, 1), graph-structure-only baseline.
        dict_path: Path to the .pt file.
        atlas: Atlas with region-to-RSN mapping.
        feature_mode: How to reduce BOLD to node features (only used when node_feature_type="bold").
        feature_dim: Output feature dimension F (only used when node_feature_type="bold").
        fc_strategy: Edge selection strategy applied to the FC matrix.
        top_k: Used when fc_strategy="top_k".
        threshold: Used for threshold-based strategies.
        bold_key: Key for the time-series tensor in each subject dict.
        fc_key: Key for the FC matrix in each subject dict.
        transpose_bold: Set True if BOLD is stored as (T, N); it will be
            transposed to (N, T) before feature extraction.
    """

    def __init__(
        self,
        dict_path: str | Path,
        atlas: AtlasMapping,
        node_feature_type: NodeFeatureType = "bold",
        feature_mode: FeatureMode = "passthrough",
        feature_dim: int = 64,
        fc_strategy: FCStrategy = "top_k",
        top_k: int = 10,
        threshold: float = 0.2,
        bold_key: str = "BOLD",
        fc_key: str = "FC",
        transpose_bold: bool = False,
    ) -> None:
        self._data: dict = _load_dict_file(Path(dict_path))
        self._subject_ids: list = list(self._data.keys())
        self.atlas = atlas
        self.node_feature_type = node_feature_type
        self.feature_mode = feature_mode
        self.feature_dim = feature_dim
        self.fc_strategy = fc_strategy
        self.top_k = top_k
        self.threshold = threshold
        self.bold_key = bold_key
        self.fc_key = fc_key
        self.transpose_bold = transpose_bold
        self._feature_module: torch.nn.Module | None = None

    def _get_feature_module(self, in_channels: int) -> torch.nn.Module:
        if self._feature_module is None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                self._feature_module = build_feature_module(
                    self.feature_mode, in_channels, self.feature_dim
                )
        return self._feature_module

    def __len__(self) -> int:
        return len(self._subject_ids)

    def __getitem__(self, idx: int) -> Data:
        subject_id = self._subject_ids[idx]
        subject = self._data[subject_id]

        fc = torch.as_tensor(subject[self.fc_key], dtype=torch.float32)
        graph = build_graph(
            fc,
            strategy=self.fc_strategy,
            top_k=self.top_k,
            threshold=self.threshold,
        )

        N = fc.shape[0]
        if self.node_feature_type == "bold":
            bold = torch.as_tensor(subject[self.bold_key], dtype=torch.float32)
            if self.transpose_bold:
                bold = bold.T  # (T, N) → (N, T)
            bold = _zscore(bold)
            feat_module = self._get_feature_module(bold.shape[1])
            with torch.no_grad():
                x = feat_module(bold)  # (N, F)
        elif self.node_feature_type == "fc_row":
            x = fc  # (N, N)
        elif self.node_feature_type == "ones":
            x = torch.ones(N, 1)
        else:
            raise ValueError(f"Unknown node_feature_type: {self.node_feature_type!r}")

        graph.x = x
        graph.rsn_ids = self.atlas.rsn_ids.clone()
        graph.subject_id = str(subject_id)

        if "age" in subject:
            graph.age = float(subject["age"])
        if "gender" in subject:
            graph.gender = subject["gender"]

        return graph


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
