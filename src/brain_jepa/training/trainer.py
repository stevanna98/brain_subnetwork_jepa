"""Pretraining trainer for BS-JEPA."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..masking.subnetwork_masking import SubnetworkMaskCollator
from ..models.bs_jepa import BSJEPA
from .ema import EMAUpdater
from .losses import jepa_loss
from .optim import LinearWDSchedule, WarmupCosineSchedule

logger = logging.getLogger(__name__)


class Trainer:
    """Single-GPU BS-JEPA pretraining loop.

    Designed so DDP is a small extension: wrap ``model.context_encoder`` and
    ``model.predictor`` with ``DistributedDataParallel`` before constructing
    this class, and pass a distributed sampler in the ``DataLoader``.

    Args:
        model: :class:`~brain_jepa.models.BSJEPA` instance.
        optimizer: AdamW optimizer over context encoder + predictor params.
        lr_scheduler: Cosine LR scheduler.
        wd_scheduler: Linear WD scheduler.
        ema_updater: EMA updater for the target encoder.
        mask_collator: Subnetwork mask collator.
        device: Torch device.
        clip_grad: If set, max gradient norm for clipping.
        log_freq: Log every this many iterations.
    """

    def __init__(
        self,
        model: BSJEPA,
        optimizer: torch.optim.AdamW,
        lr_scheduler: WarmupCosineSchedule,
        wd_scheduler: LinearWDSchedule,
        ema_updater: EMAUpdater,
        mask_collator: SubnetworkMaskCollator,
        device: torch.device,
        clip_grad: float | None = 1.0,
        log_freq: int = 10,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler
        self.ema_updater = ema_updater
        self.mask_collator = mask_collator
        self.device = device
        self.clip_grad = clip_grad
        self.log_freq = log_freq

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: Path | str | None = None,
        checkpoint_freq: int = 1,
    ) -> None:
        """Run the full pretraining loop."""
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            avg_loss = self._train_epoch(loader, epoch)
            logger.info("Epoch %d | avg_loss=%.4f", epoch, avg_loss)

            if checkpoint_dir and epoch % checkpoint_freq == 0:
                self._save_checkpoint(checkpoint_dir / f"ckpt_epoch{epoch:04d}.pt", epoch, avg_loss)

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for itr, raw_batch in enumerate(loader):
            t0 = time.time()

            # Collate with masking — raw_batch is a list of PyG Data objects
            batch, masks = self.mask_collator(raw_batch)
            batch = batch.to(self.device)
            masks.context_rsn_ids = masks.context_rsn_ids.to(self.device)
            masks.target_rsn_ids = masks.target_rsn_ids.to(self.device)
            masks.context_node_masks = [m.to(self.device) for m in masks.context_node_masks]
            masks.target_node_masks = [m.to(self.device) for m in masks.target_node_masks]

            # Forward
            self.optimizer.zero_grad()
            z_hat, z_tgt = self.model(batch, masks)
            loss = jepa_loss(z_hat, z_tgt)

            # Backward
            loss.backward()
            if self.clip_grad is not None:
                nn.utils.clip_grad_norm_(
                    list(self.model.context_encoder.parameters())
                    + list(self.model.predictor.parameters()),
                    self.clip_grad,
                )
            self.optimizer.step()

            # Schedules + EMA
            lr = self.lr_scheduler.step()
            wd = self.wd_scheduler.step()
            tau = self.ema_updater.step(self.model.context_encoder, self.model.target_encoder)

            total_loss += loss.item()

            if itr % self.log_freq == 0:
                elapsed = time.time() - t0
                logger.info(
                    "Epoch %d | itr %d | loss=%.4f | lr=%.2e | wd=%.2e | tau=%.5f | %.1f ms",
                    epoch, itr, loss.item(), lr, wd, tau, elapsed * 1000,
                )

        return total_loss / max(len(loader), 1)

    def _save_checkpoint(self, path: Path, epoch: int, loss: float) -> None:
        state = {
            "epoch": epoch,
            "loss": loss,
            "context_encoder": self.model.context_encoder.state_dict(),
            "target_encoder": self.model.target_encoder.state_dict(),
            "predictor": self.model.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)
        logger.info("Checkpoint saved → %s", path)

    @classmethod
    def load_checkpoint(cls, path: Path | str, model: BSJEPA, optimizer: torch.optim.AdamW) -> int:
        """Load a checkpoint into *model* and *optimizer*; returns the epoch."""
        state = torch.load(path, map_location="cpu")
        model.context_encoder.load_state_dict(state["context_encoder"])
        model.target_encoder.load_state_dict(state["target_encoder"])
        model.predictor.load_state_dict(state["predictor"])
        optimizer.load_state_dict(state["optimizer"])
        return state["epoch"]
