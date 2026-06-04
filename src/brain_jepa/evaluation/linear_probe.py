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

    A fresh probe is fitted from scratch at every call, so the frozen encoder
    is the only thing being measured.

    Args:
        loader:       DataLoader over a labeled subset (subjects with age/gender).
        device:       Torch device.
        train_frac:   Fraction of subjects used to fit the probe (rest is test).
        num_epochs:   Training epochs for each probe.
        lr:           Learning rate for the probe optimiser.
        seed:         RNG seed for the train/test split.
    """

    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        train_frac: float = 0.8,
        num_epochs: int = 50,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.loader = loader
        self.device = device
        self.train_frac = train_frac
        self.num_epochs = num_epochs
        self.lr = lr
        self.seed = seed

    def evaluate(self, model: BSJEPA) -> dict[str, float]:
        """Extract embeddings, fit probes, return metrics dict."""
        features, ages, genders = extract_representations_with_labels(
            model, self.loader, self.device
        )

        n = len(features)
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=g)
        split = max(1, int(self.train_frac * n))
        tr, te = perm[:split], perm[split:]

        results: dict[str, float] = {}

        if ages is not None:
            probe = RegressionProbe(features.shape[1])
            probe.fit(
                features[tr], ages[tr],
                num_epochs=self.num_epochs, lr=self.lr, device=self.device,
            )
            metrics = probe.evaluate(features[te].to(self.device), ages[te].to(self.device))
            results.update({f"age_{k}": v for k, v in metrics.items()})
            logger.info(
                "ProbeEval | age R²=%.4f | Pearson r=%.4f",
                metrics["r2"], metrics["pearson_r"],
            )

        if genders is not None:
            num_classes = int(genders.max().item()) + 1
            probe_cls = LinearProbe(features.shape[1], num_classes=num_classes)
            probe_cls.fit(
                features[tr], genders[tr],
                num_epochs=self.num_epochs, lr=self.lr, device=self.device,
            )
            metrics_cls = probe_cls.evaluate(
                features[te].to(self.device), genders[te].to(self.device)
            )
            results.update({f"gender_{k}": v for k, v in metrics_cls.items()})
            logger.info("ProbeEval | gender acc=%.4f", metrics_cls["accuracy"])

        return results
