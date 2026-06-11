"""Linear probe evaluation on frozen BS-JEPA representations."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..models.bs_jepa import BSJEPA

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_representations(
    model: BSJEPA,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    """Run the target encoder (full graph) and mean-pool to one vector per subject.

    Returns:
        features:    (N_subjects, d) representation tensor.
        subject_ids: list of subject ID strings, length N_subjects.
    """
    model.eval()
    features_list: list[torch.Tensor] = []
    subject_ids: list[str] = []

    for batch in loader:
        # collate_fn=list → batch is already a list of Data objects
        data_list = batch if isinstance(batch, list) else [batch]
        for data in data_list:
            data = data.to(device)
            node_emb = model.encode(data)   # feature extractor + target encoder (N, d)
            features_list.append(node_emb.mean(dim=0).cpu())
            subject_ids.append(str(getattr(data, "subject_id", "")))

    return torch.stack(features_list), subject_ids


class LinearProbe(nn.Module):
    """Linear classifier for categorical labels (e.g. gender, diagnosis).

    Args:
        in_features: Representation dimensionality.
        num_classes: Number of target classes.
    """

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        num_epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: torch.device | None = None,
    ) -> list[float]:
        if device:
            self.to(device)
            features = features.to(device)
            labels = labels.to(device)

        loader = DataLoader(TensorDataset(features, labels), batch_size=256, shuffle=True)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        losses: list[float] = []

        self.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(self(x_batch), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))
            if (epoch + 1) % 10 == 0:
                logger.info("LinearProbe epoch %d | loss=%.4f", epoch + 1, losses[-1])
        return losses

    @torch.no_grad()
    def evaluate(self, features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        self.eval()
        preds = self(features).argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        return {"accuracy": acc}


class RegressionProbe(nn.Module):
    """Linear regression probe for continuous labels (e.g. PMAT score, age).

    Args:
        in_features: Representation dimensionality.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)  # (B,)

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        num_epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: torch.device | None = None,
    ) -> list[float]:
        if device:
            self.to(device)
            features = features.to(device)
            labels = labels.to(device)

        loader = DataLoader(TensorDataset(features, labels), batch_size=256, shuffle=True)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        losses: list[float] = []

        self.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(self(x_batch), y_batch.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))
            if (epoch + 1) % 10 == 0:
                logger.info("RegressionProbe epoch %d | mse=%.4f", epoch + 1, losses[-1])
        return losses

    @torch.no_grad()
    def evaluate(self, features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        """Return MSE, MAE, R² and Pearson r on the provided features/labels."""
        self.eval()
        preds = self(features)
        labels = labels.float()

        mse = nn.functional.mse_loss(preds, labels).item()
        mae = (preds - labels).abs().mean().item()

        # R² = 1 - SS_res / SS_tot
        ss_res = ((labels - preds) ** 2).sum()
        ss_tot = ((labels - labels.mean()) ** 2).sum()
        r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0

        # Pearson r
        vx = preds - preds.mean()
        vy = labels - labels.mean()
        denom = (vx.norm() * vy.norm())
        pearson_r = (vx @ vy / denom).item() if denom > 0 else 0.0

        return {"mse": mse, "mae": mae, "r2": r2, "pearson_r": pearson_r}


@torch.no_grad()
def extract_representations_with_labels(
    model: BSJEPA,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run the target encoder and collect per-subject embeddings, ages, and genders.

    Returns:
        features: (N, d) tensor of mean-pooled node embeddings.
        ages:     (N,) float tensor, or None if no subject has an age attribute.
        genders:  (N,) long tensor (0/1), or None if no subject has a gender attribute.
    """
    model.eval()
    features_list: list[torch.Tensor] = []
    ages_list: list[float] = []
    genders_raw: list[Any] = []

    for batch in loader:
        # collate_fn=list → batch is a list of Data objects
        data_list = batch if isinstance(batch, list) else [batch]
        for data in data_list:
            data = data.to(device)
            node_emb = model.encode(data)
            features_list.append(node_emb.mean(dim=0).cpu())
            if hasattr(data, "age"):
                ages_list.append(float(data.age))
            if hasattr(data, "gender"):
                genders_raw.append(data.gender)

    features = torch.stack(features_list)
    n = len(features_list)

    ages = torch.tensor(ages_list, dtype=torch.float32) if len(ages_list) == n else None

    genders: torch.Tensor | None = None
    if len(genders_raw) == n:
        # Encode gender strings (e.g. "M"/"F", "Male"/"Female") or pass-through ints
        unique = sorted(set(str(g) for g in genders_raw))
        label_map = {v: i for i, v in enumerate(unique)}
        genders = torch.tensor([label_map[str(g)] for g in genders_raw], dtype=torch.long)

    return features, ages, genders


class ProbeEvaluator:
    """Periodically evaluates representation quality via age regression and gender classification.

    Uses the standard frozen-feature linear-probe protocol from the SSL
    literature: features are standardised, and a *closed-form* regularised
    linear model is fitted with k-fold cross-validation. Cross-validated
    metrics on a small held-out set are far less noisy than a single
    SGD-trained probe on one train/test split (which is what produced
    R²=-10 / acc=0.5 earlier: unstandardised targets + an under-fit AdamW
    probe on ~40 subjects).

    Args:
        loader:    DataLoader over the held-out labeled subjects.
        device:    Torch device.
        n_splits:  Number of cross-validation folds.
        seed:      RNG seed for the CV split.
    """

    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        n_splits: int = 5,
        seed: int = 0,
        # accepted for backwards-compat with existing configs; unused
        train_frac: float = 0.8,
        num_epochs: int = 50,
        lr: float = 1e-3,
    ) -> None:
        self.loader = loader
        self.device = device
        self.n_splits = n_splits
        self.seed = seed

    def evaluate(self, model: BSJEPA) -> dict[str, float]:
        """Extract embeddings, fit cross-validated linear probes, return metrics."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression, RidgeCV
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        features, ages, genders = extract_representations_with_labels(
            model, self.loader, self.device
        )
        X = features.numpy()
        results: dict[str, float] = {}

        if ages is not None:
            y = ages.numpy()
            k = min(self.n_splits, len(y))
            reg = make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=np.logspace(-2, 4, 13)),
            )
            preds = cross_val_predict(reg, X, y, cv=k)
            ss_res = float(((y - preds) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            mae = float(np.abs(y - preds).mean())
            pearson = float(np.corrcoef(preds, y)[0, 1]) if len(y) > 1 else 0.0
            results.update({"age_r2": r2, "age_mae": mae, "age_pearson_r": pearson})
            logger.info(
                "ProbeEval | age R²=%.4f | Pearson r=%.4f | MAE=%.3f", r2, pearson, mae
            )

        if genders is not None:
            y = genders.numpy()
            classes, counts = np.unique(y, return_counts=True)
            majority = counts.max() / counts.sum()
            if len(classes) < 2 or counts.min() < 2:
                logger.warning(
                    "ProbeEval | gender: too few samples per class, skipping"
                )
            else:
                k = int(min(self.n_splits, counts.min()))
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=2000, C=1.0),
                )
                cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.seed)
                preds = cross_val_predict(clf, X, y, cv=cv)
                acc = float((preds == y).mean())
                results.update({"gender_accuracy": acc, "gender_majority": float(majority)})
                logger.info(
                    "ProbeEval | gender acc=%.4f (majority baseline=%.4f)", acc, majority
                )

        return results
