#!/usr/bin/env python
"""Linear probe evaluation on frozen BS-JEPA representations.

Example::

    python scripts/linear_probe.py \\
        --config configs/eval/linear_probe.yaml \\
        --override model.checkpoint=outputs/pretrain/ckpt_epoch0010.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brain_jepa.data import BrainDataset
from brain_jepa.data.atlas import load_atlas
from brain_jepa.evaluation import LinearProbe, extract_representations
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

    subject_files = sorted(Path(cfg.data.subject_dir).glob("*.npz"))
    dataset = BrainDataset(
        subject_files=subject_files,
        atlas=atlas,
        fc_strategy=cfg.data.fc_strategy,
        top_k=int(cfg.data.top_k),
    )
    loader = DataLoader(dataset, batch_size=int(cfg.data.batch_size), num_workers=int(cfg.data.num_workers))

    model = build_bsjepa(
        atlas=atlas,
        encoder_type=cfg.model.encoder_type,
        encoder_hidden=int(cfg.model.encoder_hidden),
        encoder_out=int(cfg.model.encoder_out),
        encoder_layers=int(cfg.model.encoder_layers),
    ).to(device)

    ckpt = torch.load(cfg.model.checkpoint, map_location="cpu")
    model.context_encoder.load_state_dict(ckpt["context_encoder"])
    logger.info("Loaded checkpoint from %s (epoch %d)", cfg.model.checkpoint, ckpt.get("epoch", "?"))

    features, labels = extract_representations(model, loader, device)
    logger.info("Features: %s, Labels: %s", tuple(features.shape), tuple(labels.shape))

    # 80/20 train/test split so evaluation is on held-out subjects
    n = len(features)
    split = int(0.8 * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.meta.seed))
    train_features, test_features = features[perm[:split]], features[perm[split:]]
    train_labels, test_labels = labels[perm[:split]], labels[perm[split:]]
    logger.info("Train subjects: %d | Test subjects: %d", split, n - split)

    probe = LinearProbe(in_features=train_features.shape[1], num_classes=int(cfg.probe.num_classes))
    probe.fit(
        train_features, train_labels,
        num_epochs=int(cfg.probe.num_epochs),
        lr=float(cfg.probe.lr),
        weight_decay=float(cfg.probe.weight_decay),
        device=device,
    )
    metrics = probe.evaluate(test_features.to(device), test_labels.to(device))
    logger.info("Linear probe accuracy (test): %.4f", metrics["accuracy"])

    out_dir = Path(cfg.logging.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"probe": probe.state_dict(), "metrics": metrics}, out_dir / "linear_probe.pt")


if __name__ == "__main__":
    main()
