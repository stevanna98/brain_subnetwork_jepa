#!/usr/bin/env python
"""Linear probe evaluation on frozen BS-JEPA representations.

Two modes:

- Default (single dataset): an 80/20 split of ``data.dict_file`` trains and
  scores an SGD ``RegressionProbe``.
- Independent test set (``--test-dict`` or ``data.test_dict_file``): the probe
  and its feature scaler are fit on the *full* ``data.dict_file`` cohort and
  evaluated exactly once on the held-out file. Uses a closed-form
  ``StandardScaler + RidgeCV`` with the scaler fit on train only (no leakage).

Example (PMAT regression, single dataset)::

    python scripts/linear_probe.py \\
        --config configs/eval/linear_probe.yaml \\
        --override model.checkpoint=outputs/pretrain/ckpt_epoch0100.pt

Example (independent test set)::

    python scripts/linear_probe.py \\
        --config configs/eval/linear_probe.yaml \\
        --test-dict /path/to/heldout_subjects.pkl \\
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
    p.add_argument(
        "--test-dict",
        type=Path,
        default=None,
        help=(
            "Path to an independent test .pkl/.pt/.npz subject dict. When given, the "
            "probe is fit on the full --config dataset and evaluated once on this set "
            "(no internal split, no scaler leakage). Overrides data.test_dict_file."
        ),
    )
    p.add_argument(
        "--test-label-csv",
        type=Path,
        default=None,
        help=(
            "Optional separate label CSV for the test set (use when the held-out cohort "
            "has its own label file). Defaults to data.test_label_csv, then data.label_csv. "
            "Assumes the same label_col / subject_col as the training labels."
        ),
    )
    return p.parse_args()


def _load_score_map(label_csv, subject_col: str, label_col: str) -> dict:
    """Build {subject_id: label} from a CSV, dropping rows with a missing label."""
    import pandas as pd

    df = pd.read_csv(label_csv)
    df[subject_col] = df[subject_col].astype(str)
    return df.dropna(subset=[label_col]).set_index(subject_col)[label_col].to_dict()


def _build_loader(dict_path: Path, atlas, cfg) -> DataLoader:
    """Build an FCDictDataset DataLoader for *dict_path* using the config's data settings."""
    dataset = FCDictDataset(
        dict_path=dict_path,
        atlas=atlas,
        feature_mode=cfg.data.get("feature_mode", "passthrough"),
        feature_dim=int(cfg.data.get("feature_dim", 64)),
        fc_strategy=cfg.data.fc_strategy,
        top_k=int(cfg.data.top_k),
        bold_key=cfg.data.get("bold_key", "BOLD"),
        fc_key=cfg.data.get("fc_key", "FC"),
        transpose_bold=bool(cfg.data.get("transpose_bold", False)),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        collate_fn=list,
    )


def _match_labels(features, subject_ids, score_map):
    """Keep only subjects present in *score_map*; return (features, labels) tensors and skip count."""
    matched_features, matched_labels = [], []
    skipped = 0
    for feat, sid in zip(features, subject_ids):
        if sid in score_map:
            matched_features.append(feat)
            matched_labels.append(float(score_map[sid]))
        else:
            skipped += 1
    if not matched_features:
        return None, None, skipped
    return (
        torch.stack(matched_features),
        torch.tensor(matched_labels, dtype=torch.float32),
        skipped,
    )


def _regression_metrics(preds, labels) -> dict[str, float]:
    """MSE, MAE, R² and Pearson r for 1-D numpy arrays."""
    import numpy as np

    preds = np.asarray(preds, dtype=float)
    labels = np.asarray(labels, dtype=float)
    mse = float(((preds - labels) ** 2).mean())
    mae = float(np.abs(preds - labels).mean())
    ss_res = float(((labels - preds) ** 2).sum())
    ss_tot = float(((labels - labels.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    pearson_r = float(np.corrcoef(preds, labels)[0, 1]) if len(labels) > 1 else 0.0
    return {"mse": mse, "mae": mae, "r2": r2, "pearson_r": pearson_r}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    setup_logging()
    set_seed(cfg.meta.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    atlas = load_atlas(Path(cfg.data.atlas_csv))

    # An independent test set may be supplied via --test-dict or data.test_dict_file.
    test_dict = args.test_dict or cfg.data.get("test_dict_file", None)
    test_dict = Path(test_dict) if test_dict else None

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

    # Build the label map for the training cohort.
    label_col = cfg.data.get("label_col", "PMAT24_A_CR")
    subject_col = cfg.data.get("subject_col", "Subject")
    score_map = _load_score_map(cfg.data.label_csv, subject_col, label_col)

    # Extract + label-match the primary (training) dataset.
    train_loader = _build_loader(Path(cfg.data.dict_file), atlas, cfg)
    train_features_all, train_ids = extract_representations(model, train_loader, device)
    logger.info(
        "Train set: extracted %s for %d subjects", tuple(train_features_all.shape), len(train_ids)
    )
    train_features, train_labels, train_skipped = _match_labels(
        train_features_all, train_ids, score_map
    )
    if train_features is None:
        logger.error("No training subjects matched the label CSV. Check subject IDs.")
        sys.exit(1)

    out_dir = Path(cfg.logging.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if test_dict is not None:
        # --- Independent test set: fit on ALL train subjects, evaluate once on test. ---
        logger.info(
            "Matched %d training subjects (%d skipped — no label)", len(train_features), train_skipped
        )
        # Test labels may live in a separate CSV; fall back to the training CSV.
        test_label_csv = (
            args.test_label_csv or cfg.data.get("test_label_csv", None) or cfg.data.label_csv
        )
        test_score_map = (
            score_map
            if str(test_label_csv) == str(cfg.data.label_csv)
            else _load_score_map(test_label_csv, subject_col, label_col)
        )

        test_loader = _build_loader(test_dict, atlas, cfg)
        test_features_all, test_ids = extract_representations(model, test_loader, device)
        logger.info(
            "Test set: extracted %s for %d subjects", tuple(test_features_all.shape), len(test_ids)
        )
        test_features, test_labels, test_skipped = _match_labels(
            test_features_all, test_ids, test_score_map
        )
        if test_features is None:
            logger.error("No test subjects matched the label CSV. Check subject IDs in %s.", test_dict)
            sys.exit(1)
        logger.info(
            "Matched %d test subjects (%d skipped — no label)", len(test_features), test_skipped
        )

        # Closed-form ridge with a scaler fit on TRAIN ONLY (no leakage), then applied to test.
        import numpy as np
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X_train, y_train = train_features.numpy(), train_labels.numpy()
        X_test, y_test = test_features.numpy(), test_labels.numpy()
        reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
        reg.fit(X_train, y_train)  # scaler + ridge fit on train only

        train_metrics = _regression_metrics(reg.predict(X_train), y_train)
        metrics = _regression_metrics(reg.predict(X_test), y_test)
        logger.info(
            "Train (fit)  | MSE=%.4f | MAE=%.4f | R²=%.4f | Pearson r=%.4f",
            train_metrics["mse"], train_metrics["mae"], train_metrics["r2"], train_metrics["pearson_r"],
        )
        logger.info(
            "Independent test | MSE=%.4f | MAE=%.4f | R²=%.4f | Pearson r=%.4f",
            metrics["mse"], metrics["mae"], metrics["r2"], metrics["pearson_r"],
        )

        import joblib
        joblib.dump(reg, out_dir / "regression_probe_sklearn.joblib")
        torch.save(
            {
                "metrics": metrics,
                "train_metrics": train_metrics,
                "n_train": len(train_features),
                "n_test": len(test_features),
                "test_dict": str(test_dict),
            },
            out_dir / "regression_probe_independent.pt",
        )
        logger.info("Saved probe and metrics → %s", out_dir / "regression_probe_independent.pt")
        return

    # --- No test set supplied: original single-dataset 80/20 split behaviour. ---
    features, labels = train_features, train_labels
    logger.info("Matched %d subjects (%d skipped — no label)", len(features), train_skipped)

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

    torch.save({"probe": probe.state_dict(), "metrics": metrics}, out_dir / "regression_probe.pt")
    logger.info("Saved probe and metrics → %s", out_dir / "regression_probe.pt")


if __name__ == "__main__":
    main()
