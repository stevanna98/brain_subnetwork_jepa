"""Reproducibility utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set all RNG seeds for reproducible runs.

    Args:
        seed: Integer seed value.
        deterministic: If True, enable CUDA deterministic algorithms
            (may reduce throughput).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
