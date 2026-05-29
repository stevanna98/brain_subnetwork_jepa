#!/usr/bin/env python
"""Prepare per-subject .npz files from raw fMRI data.

This script is a starting point; adapt it to your specific data format.

Example
-------
::

    python scripts/prepare_data.py \\
        --input_dir /data/raw/HCP \\
        --output_dir data/subjects \\
        --atlas_csv data/atlas/glasser379_to_rsn12.csv \\
        --feature_dim 64

The script generates one ``.npz`` file per subject containing:
- ``X``           : (379, feature_dim) pre-computed node features (mean over T).
- ``time_series`` : (379, T) BOLD time series.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare brain fMRI data files")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--atlas_csv", type=Path, default=Path("data/atlas/glasser379_to_rsn12.csv"))
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--num_regions", type=int, default=379)
    return parser.parse_args()


def prepare_subject(input_path: Path, output_path: Path, num_regions: int) -> None:
    """Convert a raw subject file to the expected .npz format.

    Adapt this function for your data layout (HCP, OpenNeuro, etc.).
    """
    # --- Placeholder: replace with your actual loading logic ---
    # Example: load a text/csv/nifti time-series and save as npz
    logger.warning("prepare_subject is a placeholder — adapt for your data format.")

    # Synthetic example
    T = 200
    ts = np.random.randn(num_regions, T).astype(np.float32)
    X = ts.mean(axis=1, keepdims=True).repeat(64, axis=1).astype(np.float32)

    np.savez_compressed(output_path, time_series=ts, X=X)
    logger.info("Saved %s", output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = list(args.input_dir.glob("**/*.nii.gz")) + list(args.input_dir.glob("**/*.csv"))
    if not raw_files:
        logger.error("No raw files found in %s. Exiting.", args.input_dir)
        sys.exit(1)

    for fp in raw_files:
        subject_id = fp.stem
        out_path = args.output_dir / f"{subject_id}.npz"
        if out_path.exists():
            logger.info("Skipping %s (already exists)", out_path)
            continue
        prepare_subject(fp, out_path, args.num_regions)

    logger.info("Done. %d subjects processed.", len(raw_files))


if __name__ == "__main__":
    main()
