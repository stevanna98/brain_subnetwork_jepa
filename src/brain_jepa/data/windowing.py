"""Temporal windowing for spatiotemporal BS-JEPA.

Converts a subject's region BOLD ``(N, T)`` into a stack of time windows
``(P, N, L)`` and packages it as an :class:`STSample` carrying the per-token
RSN/time grid ids used by both the model and the diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .atlas import AtlasMapping
from .dataset import _load_dict_file, _zscore


def window_bold(
    bold: torch.Tensor,
    window_length: int,
    window_stride: int,
    drop_last: bool = True,
    pad_last: bool = False,
) -> torch.Tensor:
    """Slice ``(N, T)`` BOLD into windows ``(P, N, L)``.

    Args:
        bold: region time series, shape ``(N, T)``.
        window_length: L, timepoints per window.
        window_stride: hop between consecutive window starts.
        drop_last: drop a trailing partial window (default).
        pad_last: zero-pad a trailing partial window to length L instead of
            dropping it. Ignored if ``drop_last`` is True.
    """
    n, t = bold.shape
    windows: list[torch.Tensor] = []
    start = 0
    while start < t:
        seg = bold[:, start : start + window_length]
        if seg.shape[1] < window_length:
            if drop_last or not pad_last:
                break
            seg = torch.nn.functional.pad(seg, (0, window_length - seg.shape[1]))
        windows.append(seg)
        start += window_stride
    if not windows:
        raise ValueError(
            f"No windows produced: T={t}, window_length={window_length}. "
            "Reduce window_length or enable pad_last."
        )
    return torch.stack(windows)  # (P, N, L)


@dataclass
class STSample:
    """One subject's windowed sample plus the per-token RSN/time grid.

    ``rsn_ids`` / ``time_ids`` are length ``P*K`` (flattened RSN-time grid,
    row-major ``p*K + k``) so the generic diagnostics (which expect a
    ``rsn_ids`` attribute aligned with the encoder output) work unchanged.
    """

    x_win: torch.Tensor          # (P, N, L)
    rsn_ids: torch.Tensor        # (P*K,) per-token RSN id
    time_ids: torch.Tensor       # (P*K,) per-token window id
    num_windows: int
    subject_id: str = ""
    age: float | None = None
    gender: Any = None

    def to(self, device: torch.device) -> "STSample":
        self.x_win = self.x_win.to(device)
        self.rsn_ids = self.rsn_ids.to(device)
        self.time_ids = self.time_ids.to(device)
        return self


def make_grid_ids(num_windows: int, num_rsns: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(rsn_ids, time_ids)`` for a flattened ``(P, K)`` grid (row-major)."""
    p, k = num_windows, num_rsns
    rsn_ids = torch.arange(k).repeat(p)                # k cycles fastest: p*K+k -> k
    time_ids = torch.arange(p).repeat_interleave(k)    # window id
    return rsn_ids, time_ids


class WindowedBOLDDataset(Dataset):
    """Spatiotemporal dataset: yields :class:`STSample` per subject.

    Loads the same ``.pkl/.pt`` subject dict as :class:`FCDictDataset` but,
    instead of building a static FC graph, z-scores each region's full time
    series and slices it into windows. Subjects are assumed to share T (so P is
    constant across the batch); subjects whose T yields a different P are
    skipped with the window count of the first usable subject.

    Args:
        dict_path: path to the subject dict file.
        atlas: region→RSN mapping.
        window_length / window_stride / drop_last / pad_last: windowing params.
        bold_key: key for the time-series tensor in each subject dict.
        transpose_bold: set True if BOLD is stored as (T, N).
    """

    def __init__(
        self,
        dict_path: str | Path,
        atlas: AtlasMapping,
        window_length: int = 40,
        window_stride: int = 20,
        drop_last: bool = True,
        pad_last: bool = False,
        bold_key: str = "BOLD",
        transpose_bold: bool = False,
    ) -> None:
        self._data: dict = _load_dict_file(Path(dict_path))
        self._subject_ids: list = list(self._data.keys())
        self.atlas = atlas
        self.window_length = window_length
        self.window_stride = window_stride
        self.drop_last = drop_last
        self.pad_last = pad_last
        self.bold_key = bold_key
        self.transpose_bold = transpose_bold

    def __len__(self) -> int:
        return len(self._subject_ids)

    def __getitem__(self, idx: int) -> STSample:
        subject_id = self._subject_ids[idx]
        subject = self._data[subject_id]

        bold = torch.as_tensor(subject[self.bold_key], dtype=torch.float32)
        if self.transpose_bold:
            bold = bold.T  # (T, N) -> (N, T)
        bold = _zscore(bold)  # z-score per region over the full session

        x_win = window_bold(
            bold, self.window_length, self.window_stride, self.drop_last, self.pad_last
        )  # (P, N, L)
        p = x_win.shape[0]
        rsn_ids, time_ids = make_grid_ids(p, self.atlas.num_rsns)

        return STSample(
            x_win=x_win,
            rsn_ids=rsn_ids,
            time_ids=time_ids,
            num_windows=p,
            subject_id=str(subject_id),
            age=float(subject["age"]) if "age" in subject else None,
            gender=subject.get("gender") if isinstance(subject, dict) else None,
        )
