"""Linear probe evaluation on frozen BS-JEPA representations.

Representations are obtained by passing each subject through the
context encoder with no masking (all subnetworks visible) and then
mean-pooling all subnetwork tokens into a single subject-level vector.
"""

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the context encoder (all nodes visible) and pool to one vector per subject.

    Returns:
        features: (N_samples, d) representation tensor.
        labels: (N_samples,) label tensor (if loader provides them).
    """
    model.eval()
    features_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            graphs, labels = batch
            labels_list.append(labels)
        else:
            graphs = batch

        data_list = graphs.to_data_list() if hasattr(graphs, "to_data_list") else [graphs]
        for data in data_list:
            data = data.to(device)
            node_emb = model.context_encoder(data)    # (N, d)
            subj_repr = node_emb.mean(dim=0)          # (d,)
            features_list.append(subj_repr.cpu())

    features = torch.stack(features_list)
    labels = torch.cat(labels_list) if labels_list else torch.zeros(len(features), dtype=torch.long)
    return features, labels


class LinearProbe(nn.Module):
    """Single linear classifier on top of frozen representations.

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
        """Train the linear probe on pre-extracted features.

        Returns a list of per-epoch training losses.
        """
        if device:
            self.to(device)
            features = features.to(device)
            labels = labels.to(device)

        dataset = TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        losses: list[float] = []

        self.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))
            if (epoch + 1) % 10 == 0:
                logger.info("Linear probe epoch %d | loss=%.4f", epoch + 1, losses[-1])
        return losses

    @torch.no_grad()
    def evaluate(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, float]:
        """Return accuracy on the provided features/labels."""
        self.eval()
        logits = self(features)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        return {"accuracy": acc}
