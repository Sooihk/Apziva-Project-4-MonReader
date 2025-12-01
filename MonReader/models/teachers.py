from __future__ import annotations

from pathlib import Path
from typing import Optional
import os 

import torch
import torch.nn as nn

from MonReader.config import Config
from MonReader.utils import get_device
from MonReader.models.resnet18 import ResNet18Binary


def get_resnet18_teacher_ckpt_path(
    cfg: Config,
    teacher_tag: str = "resnet18_pageflip",
    pattern_substring: str = "best_full",   # look for *_best_full*.pt
) -> Path:
    """
    Return the path to the most recent *full* checkpoint for the ResNet-18 teacher.

    scans:
        cfg.models_dir / teacher_tag / "checkpoints"

    and picks the latest file whose name:
        - starts with `teacher_tag`
        - contains `pattern_substring` (default: 'best_full')

    mirrors the logic in restore_best_weights(), but for the teacher.
    """
    # If no explicit teacher_tag is given, default to cfg.tag.
    if teacher_tag is None:
        teacher_tag = cfg.tag
        
    ckpt_dir = (cfg.models_dir / teacher_tag / "checkpoints").resolve()

    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {ckpt_dir}. "
            f"Train and save the teacher model ({teacher_tag})"
        )

    # Collect candidate files like: resnet18_pageflip_best_full_....pt
    candidates = [
        p for p in ckpt_dir.iterdir()
        if p.is_file()
        and p.name.startswith(teacher_tag)
        and pattern_substring in p.name
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint files matching '{teacher_tag}*{pattern_substring}*' "
            f"found in {ckpt_dir}"
        )

    # Sort lexicographically; if your filenames include timestamps, newest is last
    candidates.sort(key=lambda p: p.name)

    # Pick the "latest" one
    return candidates[-1]

def load_resnet18_teacher(
    cfg: Config,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> nn.Module:
    if device is None:
        device = get_device()

    ckpt_path = get_resnet18_teacher_ckpt_path(cfg)  # now returns latest best_full
    print(f"Loading ResNet-18 teacher from: {ckpt_path}")

    teacher = ResNet18Binary(
        num_classes=cfg.num_classes,
        pretrained=False,
        dropout=0.4,
    )
    state = torch.load(ckpt_path, map_location=device)

    # support full checkpoint dict or plain state_dict
    if isinstance(state, dict) and "model_state" in state:
        state_dict = state["model_state"]
    else:
        state_dict = state

    teacher.load_state_dict(state_dict, strict=strict)
    teacher.to(device)
    teacher.eval()
    return teacher
