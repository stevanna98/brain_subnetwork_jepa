#!/usr/bin/env python
"""BS-JEPA pretraining entry point.

Example (real data)::

    python scripts/pretrain.py --config configs/pretrain/gcn_base.yaml

Example (synthetic data smoke test)::

    python scripts/pretrain.py \\
        --config configs/pretrain/default.yaml \\
        --override data.synthetic=true \\
        --override data.num_synthetic_subjects=32 \\
        --override training.num_epochs=3

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brain_jepa.data import BrainDataset, FCDictDataset, SyntheticBrainDataset
from brain_jepa.data.atlas import load_atlas
from brain_jepa.masking import SubnetworkMaskCollator
from brain_jepa.models import build_bsjepa
from brain_jepa.training import (
    EMAUpdater,
    LinearWDSchedule,
    Trainer,
    WarmupCosineSchedule,
    build_optimizer,
)
from brain_jepa.utils import load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BS-JEPA pretraining")
    p.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="OmegaConf dot-path overrides (repeatable)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    setup_logging()
    set_seed(cfg.meta.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Atlas
    atlas_csv = Path(cfg.data.atlas_csv)
    atlas = load_atlas(atlas_csv)
    logger.info("Atlas: %d regions, %d RSNs", atlas.num_regions, atlas.num_rsns)

    # Dataset
    if cfg.data.get("synthetic", False):
        dataset = SyntheticBrainDataset(
            atlas=atlas,
            num_subjects=int(cfg.data.num_synthetic_subjects),
            feature_dim=int(cfg.data.feature_dim),
            fc_strategy=cfg.data.fc_strategy,
            top_k=int(cfg.data.top_k),
            seed=cfg.meta.seed,
        )
        logger.info("Using synthetic dataset (%d subjects)", len(dataset))
    elif cfg.data.get("dict_file"):
        dataset = FCDictDataset(
            dict_path=Path(cfg.data.dict_file),
            atlas=atlas,
            feature_mode=cfg.data.feature_mode,
            feature_dim=int(cfg.data.feature_dim),
            fc_strategy=cfg.data.fc_strategy,
            top_k=int(cfg.data.top_k),
            threshold=float(cfg.data.threshold),
            bold_key=cfg.data.get("bold_key", "BOLD"),
            fc_key=cfg.data.get("fc_key", "FC"),
            transpose_bold=bool(cfg.data.get("transpose_bold", False)),
        )
        logger.info("FCDictDataset: %d subjects from %s", len(dataset), cfg.data.dict_file)
    else:
        subject_dir = Path(cfg.data.subject_dir)
        subject_files = sorted(subject_dir.glob("*.npz")) + sorted(subject_dir.glob("*.pt"))
        if not subject_files:
            logger.error("No subject files found in %s", subject_dir)
            sys.exit(1)
        dataset = BrainDataset(
            subject_files=subject_files,
            atlas=atlas,
            feature_mode=cfg.data.feature_mode,
            feature_dim=int(cfg.data.feature_dim),
            fc_strategy=cfg.data.fc_strategy,
            top_k=int(cfg.data.top_k),
            threshold=float(cfg.data.threshold),
        )
        logger.info("Dataset: %d subjects", len(dataset))

    # Mask collator (used in DataLoader's collate_fn via the Trainer)
    mask_collator = SubnetworkMaskCollator(
        num_rsns=atlas.num_rsns,
        num_targets=int(cfg.masking.num_targets),
        include_cross_edges_in_context=bool(cfg.masking.include_cross_edges),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        collate_fn=list,  # return list of Data; Trainer applies mask_collator
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # Model
    model = build_bsjepa(
        atlas=atlas,
        encoder_type=cfg.model.encoder_type,
        in_channels=int(cfg.data.feature_dim),
        encoder_hidden=int(cfg.model.encoder_hidden),
        encoder_out=int(cfg.model.encoder_out),
        encoder_layers=int(cfg.model.encoder_layers),
        encoder_heads=int(cfg.model.encoder_heads),
        encoder_dropout=float(cfg.model.encoder_dropout),
        pooling_mode=cfg.model.pooling_mode,
        predictor_dim=int(cfg.model.predictor_dim),
        predictor_depth=int(cfg.model.predictor_depth),
        predictor_heads=int(cfg.model.predictor_heads),
        predictor_dropout=float(cfg.model.predictor_dropout),
        include_cross_edges=bool(cfg.masking.include_cross_edges),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %d", total_params)

    # Optimisation
    iters_per_epoch = len(loader)
    total_steps = iters_per_epoch * int(cfg.training.num_epochs)
    warmup_steps = iters_per_epoch * int(cfg.training.warmup_epochs)

    optimizer = build_optimizer(
        model.context_encoder,
        model.predictor,
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay_start),
    )
    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        start_lr=float(cfg.training.start_lr),
        ref_lr=float(cfg.training.lr),
        total_steps=total_steps,
        final_lr=float(cfg.training.final_lr),
    )
    wd_scheduler = LinearWDSchedule(
        optimizer,
        wd_start=float(cfg.training.weight_decay_start),
        wd_end=float(cfg.training.weight_decay_end),
        total_steps=total_steps,
    )
    ema_updater = EMAUpdater(
        tau_start=float(cfg.training.ema_tau_start),
        tau_end=float(cfg.training.ema_tau_end),
        total_steps=total_steps,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        wd_scheduler=wd_scheduler,
        ema_updater=ema_updater,
        mask_collator=mask_collator,
        device=device,
        clip_grad=float(cfg.training.clip_grad) if cfg.training.clip_grad else None,
        log_freq=int(cfg.logging.log_freq),
    )

    output_dir = Path(cfg.meta.output_dir)
    trainer.train(
        loader=loader,
        num_epochs=int(cfg.training.num_epochs),
        checkpoint_dir=output_dir,
        checkpoint_freq=int(cfg.logging.checkpoint_freq),
        plot_dir=Path(cfg.logging.plot_dir) if cfg.logging.get("plot_dir") else None,
    )


if __name__ == "__main__":
    main()
