"""Configuration loading via OmegaConf / YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _load_with_defaults(path: Path) -> DictConfig:
    """Load a YAML config, resolving a Hydra-style ``defaults:`` list.

    Plain ``OmegaConf.load`` ignores ``defaults:`` (it's a Hydra feature), so a
    config like ``gcn_base.yaml`` with ``defaults: [default]`` would lose every
    key defined only in ``default.yaml``. This resolves each referenced config
    relative to *path*'s directory, merges them as the base, then lets the
    current file's own keys override — matching the intended inheritance.

    Supported ``defaults`` entries:
      - ``"name"``        → ``<path.parent>/name.yaml``
      - ``{group: name}`` → ``<path.parent>/group/name.yaml``
      - ``"_self_"``      → position of this file's own keys in the merge order
    """
    path = Path(path)
    raw = OmegaConf.load(path)
    defaults = raw.pop("defaults", None)
    if defaults is None:
        return raw  # type: ignore[return-value]

    merged = OmegaConf.create({})
    self_applied = False
    for entry in defaults:
        if isinstance(entry, str):
            if entry == "_self_":
                merged = OmegaConf.merge(merged, raw)
                self_applied = True
                continue
            base_path = path.parent / f"{entry}.yaml"
        else:  # dict / DictConfig: {group: name}
            (group, name), = dict(entry).items()
            base_path = path.parent / str(group) / f"{name}.yaml"

        if not base_path.exists():
            raise FileNotFoundError(
                f"Config {path} lists default {entry!r}, but {base_path} does not exist."
            )
        merged = OmegaConf.merge(merged, _load_with_defaults(base_path))

    # If _self_ wasn't explicit, the current file overrides its bases.
    if not self_applied:
        merged = OmegaConf.merge(merged, raw)
    return merged  # type: ignore[return-value]


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config file and apply optional CLI-style dot-path overrides.

    Resolves Hydra-style ``defaults:`` inheritance (e.g. ``gcn_base.yaml``
    inheriting from ``default.yaml``). CLI overrides are applied last.

    Example overrides: ``["model.encoder_type=graph_transformer", "training.lr=5e-4"]``
    """
    cfg = _load_with_defaults(Path(path))
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
