from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """
    Global configuration for the MonReader Project
    Source for:
    - Paths (data, models, outputs)
    - Data loading parameters
    - Optimization hyperparameters
    - Experiment tagging
    """
    # ---------------------------------------------------------------------
    # Paths
    # repo root
    root : Path = Path(__file__).resolve().parents[1]
    # data directory structure
    dataset_dir: Path = root / 'data'
    raw_dir: Path = dataset_dir / "raw"
    train_dir: Path = raw_dir / 'training'
    test_dir: Path = raw_dir / 'testing'

    # Where to save trained models / checkpoints
    models_dir: Path = root / 'models'

    # ---------------------------------------------------------------------
    # Data / dataloader params
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    # random seed for reproducibility
    seed: int = 42

    # ---------------------------------------------------------------------
    # Optimization hyperparameters
    # global lr
    lr: float = 1e-4
    # discriminative learning rates for fine tuning
    lr_backbone: float = 2e-5
    lr_head: float = 1e-4

    weight_decay: float = 1e-4
    # epochs for the two phases
    epochs_head: int = 5          # Phase A: linear probe (head only)
    epochs_fine_tune: int = 10    # Phase B: unfreeze part of backbone

    # ---------------------------------------------------------------------
    # Task / experiment metadata
    num_classes: int = 2

    # tag used to name output directories & checkpoint files
    # e.g. "resnet18_pageflip", "mobilenet_v2_pageflip_kd"
    tag: str = "resnet18_pageflip"

    # ---------------------------------------------------------------------
    # Knowledge distillation hyperparameters
    # alpha: weight on hard CE vs soft KL (0 => pure KD, 1 => pure CE).
    kd_alpha: float = 0.7
    # Softmax temperature for teacher/student in KD loss.
    kd_temperature: float = 4.0
    # Tag under which the teacher model's checkpoints are stored.
    # For ResNet-18 teacher, this would typically be "resnet18_pageflip".
    kd_teacher_tag: str = "resnet18_pageflip"

    @property
    def output_dir(self) -> Path:
        """
        Directory where checkpoints for this run should be saved.

        Example:
            models/resnet18_pageflip/checkpoints/
        """
        return (self.models_dir / self.tag / "checkpoints").resolve()