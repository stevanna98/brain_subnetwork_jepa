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

from brain_jepa.data import (
    BrainDataset,
    FCDictDataset,
    SyntheticBrainDataset,
    WindowedBOLDDataset,
)
from brain_jepa.data.atlas import load_atlas
from brain_jepa.data.transforms import build_feature_module
from brain_jepa.evaluation import ProbeEvaluator
from brain_jepa.masking import SpatioTemporalMaskCollator, SubnetworkMaskCollator
from brain_jepa.models import build_bsjepa, build_st_bsjepa
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


def _held_out_split(dataset, cfg) -> tuple[list[int], int]:
    """Pick held-out probe/validation indices; returns (probe_indices, probe_freq)."""
    probe_cfg = cfg.get("probe", {})
    probe_freq = int(probe_cfg.get("freq", 10))
    if probe_freq <= 0:
        return [], probe_freq
    requested = int(probe_cfg.get("num_subjects", min(200, len(dataset))))
    probe_size = min(requested, len(dataset) // 2)
    g = torch.Generator().manual_seed(int(cfg.meta.seed))
    return torch.randperm(len(dataset), generator=g)[:probe_size].tolist(), probe_freq


def run_spatiotemporal(cfg, atlas, device: torch.device) -> None:
    """Spatiotemporal BS-JEPA pretraining (model.mode=spatiotemporal)."""
    if cfg.data.get("synthetic", False):
        logger.error("Spatiotemporal mode requires a real dict_file (no synthetic path).")
        sys.exit(1)

    dataset = WindowedBOLDDataset(
        dict_path=Path(cfg.data.dict_file),
        atlas=atlas,
        window_length=int(cfg.data.window_length),
        window_stride=int(cfg.data.window_stride),
        drop_last=bool(cfg.data.get("drop_last", True)),
        pad_last=bool(cfg.data.get("pad_last", False)),
        bold_key=cfg.data.get("bold_key", "BOLD"),
        transpose_bold=bool(cfg.data.get("transpose_bold", False)),
    )
    logger.info("WindowedBOLDDataset: %d subjects from %s", len(dataset), cfg.data.dict_file)

    mask_collator = SpatioTemporalMaskCollator(
        num_rsns=atlas.num_rsns,
        mode=cfg.masking.get("mode", "block"),
        target_rsn_ratio=float(cfg.masking.get("target_rsn_ratio", 0.5)),
        target_time_ratio=float(cfg.masking.get("target_time_ratio", 0.5)),
        num_target_blocks=int(cfg.masking.get("num_target_blocks", 1)),
        block_time_length=int(cfg.masking.get("block_time_length", 2)),
        seed=int(cfg.meta.seed),
    )

    # Held-out subjects for probe + diagnostics.
    probe_indices, probe_freq = _held_out_split(dataset, cfg)
    if probe_indices:
        probe_set = set(probe_indices)
        train_dataset = torch.utils.data.Subset(
            dataset, [i for i in range(len(dataset)) if i not in probe_set]
        )
        logger.info(
            "Subject split: %d pretraining / %d probe (held out)",
            len(train_dataset), len(probe_indices),
        )
    else:
        train_dataset = dataset

    loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        collate_fn=list,  # Trainer applies the ST mask collator
        drop_last=True,
    )

    model = build_st_bsjepa(
        atlas=atlas,
        window_length=int(cfg.data.window_length),
        embed_dim=int(cfg.model.get("st_embed_dim", 256)),
        feature_mode=cfg.data.get("feature_mode", "conv1d"),
        time_max_windows=int(cfg.model.get("time_max_windows", 64)),
        st_encoder_depth=int(cfg.model.get("st_encoder_depth", 4)),
        st_encoder_heads=int(cfg.model.get("st_encoder_heads", 8)),
        st_predictor_dim=int(cfg.model.get("st_predictor_dim", 128)),
        st_predictor_depth=int(cfg.model.get("st_predictor_depth", 4)),
        st_predictor_heads=int(cfg.model.get("st_predictor_heads", 4)),
        dropout=float(cfg.model.get("encoder_dropout", 0.0)),
    ).to(device)
    logger.info(
        "ST trainable params: %d",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    iters_per_epoch = len(loader)
    total_steps = iters_per_epoch * int(cfg.training.num_epochs)
    warmup_steps = iters_per_epoch * int(cfg.training.warmup_epochs)

    optimizer = build_optimizer(
        model.tokenizer, model.context_encoder, model.predictor,
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay_start),
    )
    lr_scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=warmup_steps, start_lr=float(cfg.training.start_lr),
        ref_lr=float(cfg.training.lr), total_steps=total_steps, final_lr=float(cfg.training.final_lr),
    )
    wd_scheduler = LinearWDSchedule(
        optimizer, wd_start=float(cfg.training.weight_decay_start),
        wd_end=float(cfg.training.weight_decay_end), total_steps=total_steps,
    )
    ema_updater = EMAUpdater(
        tau_start=float(cfg.training.ema_tau_start),
        tau_end=float(cfg.training.ema_tau_end), total_steps=total_steps,
    )

    probe_evaluator = None
    val_loader = None
    if probe_indices:
        probe_loader = DataLoader(
            torch.utils.data.Subset(dataset, probe_indices),
            batch_size=int(cfg.data.batch_size),
            shuffle=False, num_workers=int(cfg.data.num_workers), collate_fn=list,
        )
        val_loader = probe_loader
        probe_evaluator = ProbeEvaluator(loader=probe_loader, device=device, seed=int(cfg.meta.seed))

    trainer = Trainer(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, wd_scheduler=wd_scheduler,
        ema_updater=ema_updater, mask_collator=mask_collator, device=device,
        clip_grad=float(cfg.training.clip_grad) if cfg.training.clip_grad else None,
        log_freq=int(cfg.logging.log_freq), probe_evaluator=probe_evaluator,
        var_weight=float(cfg.training.get("var_weight", 0.5)),
        ctx_var_weight=float(cfg.training.get("ctx_var_weight", 1.0)),
        cov_weight=float(cfg.training.get("cov_weight", 0.1)),
        var_gamma=float(cfg.training.get("var_gamma", 0.1)),
        val_loader=val_loader, diag_freq=int(cfg.logging.get("diag_freq", 1)),
    )

    output_dir = Path(cfg.meta.output_dir)
    trainer.train(
        loader=loader, num_epochs=int(cfg.training.num_epochs), checkpoint_dir=output_dir,
        checkpoint_freq=int(cfg.logging.checkpoint_freq),
        plot_dir=Path(cfg.logging.plot_dir) if cfg.logging.get("plot_dir") else None,
        probe_freq=probe_freq,
    )


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

    # Spatiotemporal path is fully separate; static path continues below.
    if cfg.model.get("mode", "static") == "spatiotemporal":
        run_spatiotemporal(cfg, atlas, device)
        return

    # Dataset
    node_feature_type = cfg.data.get("node_feature_type", "bold")
    # in_channels depends on what we use as node features
    if node_feature_type == "bold":
        in_channels = int(cfg.data.feature_dim)
    elif node_feature_type == "fc_row":
        in_channels = atlas.num_regions
    elif node_feature_type == "ones":
        in_channels = 1
    else:
        logger.error("Unknown node_feature_type: %s", node_feature_type)
        sys.exit(1)

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
            node_feature_type=node_feature_type,
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
            node_feature_type=node_feature_type,
            feature_mode=cfg.data.feature_mode,
            feature_dim=int(cfg.data.feature_dim),
            fc_strategy=cfg.data.fc_strategy,
            top_k=int(cfg.data.top_k),
            threshold=float(cfg.data.threshold),
        )
        logger.info("Dataset: %d subjects", len(dataset))

    # For bold mode: detect T from first sample, build trainable feature extractor
    feature_extractor = None
    if node_feature_type == "bold" and not cfg.data.get("synthetic", False):
        time_points = dataset[0].x.shape[1]
        feature_extractor = build_feature_module(
            cfg.data.feature_mode,
            in_channels=time_points,
            out_channels=int(cfg.data.feature_dim),
        ).to(device)
        logger.info(
            "Feature extractor: %s | T=%d → F=%d",
            cfg.data.feature_mode, time_points, cfg.data.feature_dim,
        )

    logger.info("Node feature type: %s → in_channels=%d", node_feature_type, in_channels)

    # Mask collator (used in DataLoader's collate_fn via the Trainer)
    mask_collator = SubnetworkMaskCollator(
        num_rsns=atlas.num_rsns,
        num_targets=int(cfg.masking.num_targets),
        extra_target_ratio=float(cfg.masking.get("extra_target_ratio", 0.0)),
        include_cross_edges_in_context=bool(cfg.masking.include_cross_edges),
    )

    # Probe subjects are held out from pretraining so probe metrics measure
    # generalisation to unseen subjects, not memorisation.
    probe_cfg = cfg.get("probe", {})
    probe_freq = int(probe_cfg.get("freq", 10))
    probe_indices: list[int] = []
    if not cfg.data.get("synthetic", False) and probe_freq > 0:
        requested = int(probe_cfg.get("num_subjects", min(200, len(dataset))))
        probe_size = min(requested, len(dataset) // 2)
        if probe_size < requested:
            logger.warning(
                "Capping probe set at half the dataset: %d subjects", probe_size
            )
        g = torch.Generator().manual_seed(int(cfg.meta.seed))
        probe_indices = torch.randperm(len(dataset), generator=g)[:probe_size].tolist()

    if probe_indices:
        probe_set = set(probe_indices)
        train_indices = [i for i in range(len(dataset)) if i not in probe_set]
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        logger.info(
            "Subject split: %d pretraining / %d probe (held out)",
            len(train_indices), len(probe_indices),
        )
    else:
        train_dataset = dataset

    loader = DataLoader(
        train_dataset,
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
        in_channels=in_channels,
        encoder_hidden=int(cfg.model.encoder_hidden),
        encoder_out=int(cfg.model.encoder_out),
        encoder_layers=int(cfg.model.encoder_layers),
        encoder_heads=int(cfg.model.encoder_heads),
        encoder_dropout=float(cfg.model.encoder_dropout),
        predictor_dim=int(cfg.model.predictor_dim),
        predictor_depth=int(cfg.model.predictor_depth),
        predictor_heads=int(cfg.model.predictor_heads),
        predictor_dropout=float(cfg.model.predictor_dropout),
        region_positional_encoding=bool(cfg.model.get("region_positional_encoding", False)),
        feature_extractor=feature_extractor,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %d", total_params)

    # Optimisation
    iters_per_epoch = len(loader)
    total_steps = iters_per_epoch * int(cfg.training.num_epochs)
    warmup_steps = iters_per_epoch * int(cfg.training.warmup_epochs)

    modules_to_train = [model.context_encoder, model.predictor]
    if model.feature_extractor is not None:
        modules_to_train.append(model.feature_extractor)
    optimizer = build_optimizer(
        *modules_to_train,
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

    # Probe evaluator — uses the held-out probe subjects selected above
    probe_evaluator = None
    val_loader = None
    if probe_indices:
        probe_subset = torch.utils.data.Subset(dataset, probe_indices)
        probe_loader = DataLoader(
            probe_subset,
            batch_size=int(cfg.data.batch_size),
            shuffle=False,
            num_workers=int(cfg.data.num_workers),
            collate_fn=list,
        )
        # Same held-out subjects drive the per-epoch validation-loss and
        # representation-health diagnostics.
        val_loader = probe_loader
        probe_evaluator = ProbeEvaluator(
            loader=probe_loader,
            device=device,
            train_frac=float(probe_cfg.get("train_frac", 0.8)),
            num_epochs=int(probe_cfg.get("num_epochs", 50)),
            lr=float(probe_cfg.get("lr", 1e-3)),
            seed=int(cfg.meta.seed),
        )
        logger.info(
            "ProbeEvaluator: %d held-out subjects, every %d epochs",
            len(probe_indices), probe_freq,
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
        probe_evaluator=probe_evaluator,
        var_weight=float(cfg.training.get("var_weight", 0.5)),
        ctx_var_weight=float(cfg.training.get("ctx_var_weight", 1.0)),
        cov_weight=float(cfg.training.get("cov_weight", 0.1)),
        var_gamma=float(cfg.training.get("var_gamma", 0.1)),
        val_loader=val_loader,
        diag_freq=int(cfg.logging.get("diag_freq", 1)),
    )

    output_dir = Path(cfg.meta.output_dir)
    trainer.train(
        loader=loader,
        num_epochs=int(cfg.training.num_epochs),
        checkpoint_dir=output_dir,
        checkpoint_freq=int(cfg.logging.checkpoint_freq),
        plot_dir=Path(cfg.logging.plot_dir) if cfg.logging.get("plot_dir") else None,
        probe_freq=probe_freq,
    )


if __name__ == "__main__":
    main()
