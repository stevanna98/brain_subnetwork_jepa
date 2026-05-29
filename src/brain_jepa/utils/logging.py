"""Logging setup with optional Weights & Biases integration."""

from __future__ import annotations

import logging
import sys
from typing import Any


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure root logger with a console handler (and optional file handler)."""
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt, handlers=handlers)


class WandbLogger:
    """Thin wrapper around wandb that no-ops gracefully when wandb is absent."""

    def __init__(self, project: str, config: dict[str, Any], enabled: bool = True) -> None:
        self._run = None
        if not enabled:
            return
        try:
            import wandb
            self._run = wandb.init(project=project, config=config)
        except ImportError:
            logging.getLogger(__name__).warning(
                "wandb not installed — logging to W&B is disabled."
            )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
