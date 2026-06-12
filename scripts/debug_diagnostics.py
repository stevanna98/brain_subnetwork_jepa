#!/usr/bin/env python
"""One-shot representation-health debug pass on a pretrained checkpoint.

Loads the frozen target encoder, extracts subject embeddings under several
pooling strategies, and prints a detailed report (subject count, embedding
shapes, sample IDs/norms, pairwise-cosine spread, ranks, node-level stats, and
per-pooling collapse metrics). Diagnostic-only: never updates the model.

Example::

    python scripts/debug_diagnostics.py \\
        --config configs/eval/linear_probe.yaml \\
        --override model.checkpoint=outputs/pretrain/ckpt_epoch0030.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brain_jepa.data import FCDictDataset
from brain_jepa.data.atlas import load_atlas
from brain_jepa.evaluation import debug_report
from brain_jepa.models import build_bsjepa
from brain_jepa.utils import load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Representation-health debug pass")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--override", action="append", default=[], dest="overrides", metavar="KEY=VALUE")
    p.add_argument("--num-subjects", type=int, default=None, help="Cap subjects for a quick pass.")
    p.add_argument(
        "--disable-target-identity",
        action="store_true",
        help="Zero out the predictor's target identity embeddings (predictor shortcut test).",
    )
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
        node_feature_type=cfg.data.get("node_feature_type", "bold"),
        feature_mode=cfg.data.get("feature_mode", "passthrough"),
        feature_dim=int(cfg.data.get("feature_dim", 64)),
        fc_strategy=cfg.data.fc_strategy,
        top_k=int(cfg.data.top_k),
        bold_key=cfg.data.get("bold_key", "BOLD"),
        fc_key=cfg.data.get("fc_key", "FC"),
        transpose_bold=bool(cfg.data.get("transpose_bold", False)),
    )
    if args.num_subjects is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(min(args.num_subjects, len(dataset)))))
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
    logger.info("Loaded checkpoint %s (epoch %s)", cfg.model.checkpoint, ckpt.get("epoch", "?"))

    if args.disable_target_identity:
        model.predictor.disable_target_identity = True
        logger.info("Predictor target identity embeddings DISABLED for this pass.")

    report = debug_report(model, loader, device)

    out_dir = Path(cfg.logging.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diagnostics_debug.json"
    with out_path.open("w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Saved debug report → %s", out_path)


if __name__ == "__main__":
    main()
