#!/usr/bin/env python
"""Linear probe evaluation on frozen BS-JEPA representations.

Example (PMAT regression)::

    python scripts/linear_probe.py \\
        --config configs/eval/linear_probe.yaml \\
        --override model.checkpoint=outputs/pretrain/ckpt_epoch0100.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brain_jepa.data import FCDictDataset
from brain_jepa.data.atlas import load_atlas
from brain_jepa.evaluation import RegressionProbe, extract_representations
from brain_jepa.models import build_bsjepa
from brain_jepa.utils import load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear probe evaluation")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--override", action="append", default=[], dest="overrides", metavar="KEY=VALUE")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    setup_logging()
    set_seed(cfg.meta.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    atlas = load_atlas(Path(cfg.data.atlas_csv))

    dataset = FCDictDataset(
        dict_path=Path(cfg.data.dict_file),
        atlas=atlas,
        feature_mode=cfg.data.get("feature_mode", "passthrough"),
        feature_dim=int(cfg.data.get("feature_dim", 64)),
        fc_strategy=cfg.data.fc_strategy,
        top_k=int(cfg.data.top_k),
        bold_key=cfg.data.get("bold_key", "BOLD"),
        fc_key=cfg.data.get("fc_key", "FC"),
        transpose_bold=bool(cfg.data.get("transpose_bold", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        collate_fn=list,
    )

    model = build_bsjepa(
        atlas=atlas,
        encoder_type=cfg.model.encoder_type,
        in_channels=int(cfg.data.get("feature_dim", 64)),
        encoder_hidden=int(cfg.model.encoder_hidden),
        encoder_out=int(cfg.model.encoder_out),
        encoder_layers=int(cfg.model.encoder_layers),
    ).to(device)

    ckpt = torch.load(cfg.model.checkpoint, map_location="cpu", weights_only=False)
    model.target_encoder.load_state_dict(ckpt["target_encoder"])
    if model.feature_extractor is not None and "feature_extractor" in ckpt:
        model.feature_extractor.load_state_dict(ckpt["feature_extractor"])
    logger.info("Loaded checkpoint from %s (epoch %d)", cfg.model.checkpoint, ckpt.get("epoch", "?"))

    # Extract representations and subject IDs
    features, subject_ids = extract_representations(model, loader, device)
    logger.info("Extracted features: %s for %d subjects", tuple(features.shape), len(subject_ids))

    # Load PMAT scores and match to subjects
    import pandas as pd
    label_col = cfg.data.get("label_col", "PMAT24_A_CR")
    subject_col = cfg.data.get("subject_col", "Subject")
    df = pd.read_csv(cfg.data.label_csv)
    df[subject_col] = df[subject_col].astype(str)
    score_map = df.dropna(subset=[label_col]).set_index(subject_col)[label_col].to_dict()

    matched_features, matched_labels = [], []
    skipped = 0
    for feat, sid in zip(features, subject_ids):
        if sid in score_map:
            matched_features.append(feat)
            matched_labels.append(float(score_map[sid]))
        else:
            skipped += 1

    if not matched_features:
        logger.error("No subjects matched between dataset and label CSV. Check subject IDs.")
        sys.exit(1)

    features = torch.stack(matched_features)
    labels = torch.tensor(matched_labels, dtype=torch.float32)
    logger.info("Matched %d subjects (%d skipped — no PMAT score)", len(features), skipped)

    # 80/20 train/test split
    n = len(features)
    split = int(0.8 * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.meta.seed))
    train_features, test_features = features[perm[:split]], features[perm[split:]]
    train_labels, test_labels = labels[perm[:split]], labels[perm[split:]]
    logger.info("Train: %d | Test: %d", split, n - split)

    # Fit regression probe
    probe = RegressionProbe(in_features=features.shape[1])
    probe.fit(
        train_features, train_labels,
        num_epochs=int(cfg.probe.num_epochs),
        lr=float(cfg.probe.lr),
        weight_decay=float(cfg.probe.weight_decay),
        device=device,
    )

    metrics = probe.evaluate(test_features.to(device), test_labels.to(device))
    logger.info(
        "Test | MSE=%.4f | MAE=%.4f | R²=%.4f | Pearson r=%.4f",
        metrics["mse"], metrics["mae"], metrics["r2"], metrics["pearson_r"],
    )

    out_dir = Path(cfg.logging.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"probe": probe.state_dict(), "metrics": metrics}, out_dir / "regression_probe.pt")
    logger.info("Saved probe and metrics → %s", out_dir / "regression_probe.pt")


if __name__ == "__main__":
    main()
