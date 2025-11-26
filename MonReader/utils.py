from __future__ import annotations

import os
import random
from typing import Any

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

def set_seed(seed: int) -> None:
    """
    Set as many RNG seeds as possible for reproducibility.

    Call this once at the start of each training script:

        from monreader.utils import set_seed
        set_seed(cfg.seed)
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # some extra determinism; you can relax these if you want speed over reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """
    Return the best available device.

    Usage:
        device = get_device()
        model.to(device)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # for Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def get_model_size_mb(model: nn.Module) -> float:
    """
    Approximate model size in megabytes based on its state_dict.
    For F1 vs MB and Pareto plots.
    Usage:
        size_mb = get_model_size_mb(model)
    """
    total_bytes = 0
    state_dict = model.state_dict()
    for tensor in state_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 ** 2)

def ensure_dir(path: Path) -> Path:
    """
    Create a directory (and parents) if it does not exist.
    Returns the path so you can write:
        ckpt_dir = ensure_dir(cfg.output_dir)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path