"""Configuration loading via OmegaConf / YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config file and apply optional CLI-style dot-path overrides.

    Example overrides: ``["model.encoder_type=graph_transformer", "training.lr=5e-4"]``
    """
    cfg = OmegaConf.load(Path(path))
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
