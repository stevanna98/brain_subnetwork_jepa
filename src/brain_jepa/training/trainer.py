"""Pretraining trainer for BS-JEPA."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, safe in all environments
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

from ..evaluation.linear_probe import ProbeEvaluator
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
        probe_evaluator: If provided, evaluated every ``probe_freq`` epochs.
        var_weight: Weight for the variance regularization term in the loss.
        var_gamma: Target std for variance regularization.
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
        probe_evaluator: ProbeEvaluator | None = None,
        var_weight: float = 0.5,
        ctx_var_weight: float = 1.0,
        cov_weight: float = 0.1,
        var_gamma: float = 0.1,
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
        self.probe_evaluator = probe_evaluator
        self._var_weight = var_weight
        self._ctx_var_weight = ctx_var_weight
        self._cov_weight = cov_weight
        self._var_gamma = var_gamma

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: Path | str | None = None,
        checkpoint_freq: int = 1,
        plot_dir: Path | str | None = None,
        probe_freq: int = 10,
    ) -> None:
        """Run the full pretraining loop."""
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        plot_dir = Path(plot_dir) if plot_dir else None
        if plot_dir:
            plot_dir.mkdir(parents=True, exist_ok=True)

        epoch_losses: list[float] = []
        epoch_taus: list[float] = []
        probe_history: list[dict] = []  # {epoch, **metrics}

        for epoch in range(1, num_epochs + 1):
            avg_loss = self._train_epoch(loader, epoch)
            tau = self.ema_updater.current_tau

            epoch_losses.append(avg_loss)
            epoch_taus.append(tau)

            logger.info("Epoch %d | avg_loss=%.4f | tau=%.5f", epoch, avg_loss, tau)

            is_last = epoch == num_epochs
            if checkpoint_dir and (epoch % checkpoint_freq == 0 or is_last):
                self._save_checkpoint(checkpoint_dir / f"ckpt_epoch{epoch:04d}.pt", epoch, avg_loss)

            if self.probe_evaluator is not None and (epoch % probe_freq == 0 or is_last):
                logger.info("Running probe evaluation at epoch %d…", epoch)
                metrics = self.probe_evaluator.evaluate(self.model)
                probe_history.append({"epoch": epoch, **metrics})
                self.model.train()  # restore train mode after probe eval

            if plot_dir:
                self._save_plots(plot_dir, epoch_losses, epoch_taus, probe_history)

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        running: dict[str, float] = {}

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
            z_hat, z_tgt, ctx_embs = self.model(batch, masks)
            loss, metrics = jepa_loss(
                z_hat, z_tgt, ctx_embs,
                var_weight=self._var_weight,
                ctx_var_weight=self._ctx_var_weight,
                cov_weight=self._cov_weight,
                var_gamma=self._var_gamma,
            )

            # Backward
            loss.backward()
            if self.clip_grad is not None:
                params = (
                    list(self.model.context_encoder.parameters())
                    + list(self.model.predictor.parameters())
                )
                if self.model.feature_extractor is not None:
                    params += list(self.model.feature_extractor.parameters())
                nn.utils.clip_grad_norm_(params, self.clip_grad)
            self.optimizer.step()

            # Schedules + EMA
            lr = self.lr_scheduler.step()
            wd = self.wd_scheduler.step()
            tau = self.ema_updater.step(self.model.context_encoder, self.model.target_encoder)

            total_loss += loss.item()
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v.item()

            if itr % self.log_freq == 0:
                elapsed = time.time() - t0
                metric_str = " | ".join(f"{k}={v.item():.4f}" for k, v in metrics.items())
                logger.info(
                    "Epoch %d | itr %d | loss=%.4f | %s | lr=%.2e | tau=%.5f | %.1f ms",
                    epoch, itr, loss.item(), metric_str, lr, tau, elapsed * 1000,
                )

        n = max(len(loader), 1)
        avg_str = " | ".join(f"avg_{k}={v / n:.4f}" for k, v in running.items())
        logger.info("Epoch %d done | avg_loss=%.4f | %s", epoch, total_loss / n, avg_str)
        return total_loss / n

    def _save_plots(
        self,
        plot_dir: Path,
        epoch_losses: list[float],
        epoch_taus: list[float],
        probe_history: list[dict] | None = None,
    ) -> None:
        if not _MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not installed — skipping plots.")
            return

        epochs = list(range(1, len(epoch_losses) + 1))

        # Loss plot
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, epoch_losses, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("BS-JEPA pretraining loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "loss.png", dpi=120)
        plt.close(fig)

        # Tau plot
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, epoch_taus, color="darkorange", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("EMA momentum τ")
        ax.set_title("Target encoder EMA momentum")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "tau.png", dpi=120)
        plt.close(fig)

        # Probe metrics plot
        if not probe_history:
            return
        probe_epochs = [h["epoch"] for h in probe_history]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        if "age_r2" in probe_history[0]:
            axes[0].plot(probe_epochs, [h["age_r2"] for h in probe_history],
                         marker="o", linewidth=1.5, label="R²")
            axes[0].plot(probe_epochs, [h["age_pearson_r"] for h in probe_history],
                         marker="s", linewidth=1.5, linestyle="--", label="Pearson r")
            axes[0].set_xlabel("Epoch")
            axes[0].set_title("Age regression probe")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        if "gender_accuracy" in probe_history[0]:
            axes[1].plot(probe_epochs, [h["gender_accuracy"] for h in probe_history],
                         marker="o", color="green", linewidth=1.5)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title("Gender classification probe")
            axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "probe_metrics.png", dpi=120)
        plt.close(fig)

    def _save_checkpoint(self, path: Path, epoch: int, loss: float) -> None:
        state = {
            "epoch": epoch,
            "loss": loss,
            "context_encoder": self.model.context_encoder.state_dict(),
            "target_encoder": self.model.target_encoder.state_dict(),
            "predictor": self.model.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.model.feature_extractor is not None:
            state["feature_extractor"] = self.model.feature_extractor.state_dict()
        torch.save(state, path)
        logger.info("Checkpoint saved → %s", path)

    @classmethod
    def load_checkpoint(cls, path: Path | str, model: BSJEPA, optimizer: torch.optim.AdamW) -> int:
        """Load a checkpoint into *model* and *optimizer*; returns the epoch."""
        state = torch.load(path, map_location="cpu")
        model.context_encoder.load_state_dict(state["context_encoder"])
        model.target_encoder.load_state_dict(state["target_encoder"])
        model.predictor.load_state_dict(state["predictor"])
        if model.feature_extractor is not None and "feature_extractor" in state:
            model.feature_extractor.load_state_dict(state["feature_extractor"])
        optimizer.load_state_dict(state["optimizer"])
        return state["epoch"]
