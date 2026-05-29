"""Linear probe evaluation on frozen BS-JEPA representations."""

from __future__ import annotations

import logging

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
        graphs = batch[0] if isinstance(batch, (list, tuple)) else batch
        data_list = graphs.to_data_list() if hasattr(graphs, "to_data_list") else [graphs]
        for data in data_list:
            data = data.to(device)
            node_emb = model.target_encoder(data)   # (N, d) — full graph
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
