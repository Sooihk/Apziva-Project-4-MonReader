# monreader/training/train_mobilenet_v2_kd.py

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from MonReader.config import Config
from MonReader.utils import set_seed, get_device, get_model_size_mb
from MonReader.data.datamodules import DataModule
from MonReader.models.mobilenet_v2 import MobileNetV2Binary, create_mobilenetv2_student
from MonReader.models.utils import freeze_batchnorm_running_stats
from MonReader.models.teachers import load_resnet18_teacher
from MonReader.training.engine import evaluate, CheckpointManager
from MonReader.training.distillation import DistillationLoss, train_student_with_distillation




def train_mobilenet_v2_kd(cfg: Config) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Knowledge-distillation training for MobileNetV2 (single-phase).
        - Teacher: ResNet-18 loaded from its best checkpoint.
        - Student: MobileNetV2Binary initialized from ImageNet.
        - All student layers are trainable under KD loss (no Phase A/B).

    Training loop:
        * For each batch, teacher produces logits with no grad.
        * Student produces logits, and DistillationLoss combines
          hard CE with soft KL against teacher logits.
        * Train the *entire* student end-to-end.

    Config requirements (in addition to standard fields):
        - cfg.kd_alpha:       float in [0, 1], weight on hard CE vs soft KL.
        - cfg.kd_temperature: float > 0, softmax temperature for KD.
        - cfg.kd_teacher_tag: str, tag under which the teacher was trained
                               (e.g., "resnet18_pageflip").
        - cfg.tag:            student KD experiment tag
                               (e.g., "mobilenet_v2_kd_pageflip").

    Returns:
        Best validation metrics (dict) for the KD-trained student.
    """
    # ------------------------------------------------------------------
    # Setup: seed, device, dataloaders
    # ------------------------------------------------------------------
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    dm = DataModule(cfg)
    train_loader, val_loader = dm.build_dataloaders()

    # ------------------------------------------------------------------
    # Teacher: ResNet-18 loaded from its best checkpoint
    # ------------------------------------------------------------------
    # Construct a teacher Config by copying the current cfg but swapping
    # the tag to `cfg.kd_teacher_tag` so that teacher checkpoints are read
    # from the correct directory.
    teacher_cfg = replace(cfg, tag=cfg.kd_teacher_tag)
    teacher = load_resnet18_teacher(teacher_cfg, device=device, strict=True)

    # Ensure teacher parameters never receive gradients.
    for p in teacher.parameters():
        p.requires_grad = False

    # ------------------------------------------------------------------
    # Student: MobileNetV2 initialized from ImageNet
    # ------------------------------------------------------------------
    student = create_mobilenetv2_student(
        cfg=cfg,
        device=device,
        pretrained=True,
        dropout=0.2,
    )

    # KD loss function combining hard labels and teacher soft targets.
    kd_loss_fn = DistillationLoss(
        alpha=cfg.kd_alpha,
        temperature=cfg.kd_temperature,
    )

    # Validation uses plain CE loss on hard labels (no teacher needed).
    criterion_val = nn.CrossEntropyLoss()

    # AMP scaler.
    scaler = GradScaler()

    # Checkpoint manager keyed by the student KD tag.
    ckpt_mgr = CheckpointManager(output_dir=cfg.output_dir, tag=cfg.tag)

    # ------------------------------------------------------------------
    # Single-phase KD: train the entire student end-to-end
    print("=== MobileNetV2 KD: full-network distillation (no Phase A/B) ===")

    # All parameters are trainable; no freezing/unfreezing here.
    for param in student.model.parameters():
        param.requires_grad = True

    # Optimizer over all student parameters.
    optimizer = optim.AdamW(
        student.parameters(),
        lr=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )

    # Cosine schedule over the chosen KD epochs (use epochs_fine_tune).
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.epochs_fine_tune),
    )

    best_val_metrics: Dict[str, float] = {"f1": -1.0}

    for epoch in range(1, cfg.epochs_fine_tune + 1):
        # KD training step: teacher + student together.
        train_metrics = train_student_with_distillation(
            teacher=teacher,
            student=student,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            kd_loss_fn=kd_loss_fn,
            device=device,
            epoch=epoch,
            phase="MBV2-KD",
        )

        # Validation on hard labels only.
        val_metrics = evaluate(
            model=student,
            loader=val_loader,
            criterion=criterion_val,
            device=device,
            epoch=epoch,
            phase="MBV2-KD-Val",
        )

        scheduler.step()

        improved, _, _ = ckpt_mgr.maybe_save(
            model=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            val_metrics=val_metrics,
        )

        if improved:
            best_val_metrics = val_metrics

    print(f"[MobileNetV2-KD] Best val F1: {ckpt_mgr.best_f1:.4f}")

    # Restore best-validation checkpoint so this KD student can be used
    # for final test evaluation or deployment.
    ckpt_mgr.load_best_weights(student, device)
    if getattr(ckpt_mgr, "best_weights_path", None) is not None:
        model_size_mb = get_model_size_mb(ckpt_mgr.best_weights_path)
    else:
        model_size_mb = float("nan")

    best_val_metrics["model_size_mb"] = model_size_mb
    return student, best_val_metrics


if __name__ == "__main__":
    # Example entrypoint; assumes Config has KD-related fields.
    cfg = Config(
        tag="mobilenet_v2_kd_pageflip",
        # These fields must exist on Config; adjust names/defaults to match
        # your actual dataclass.
        kd_alpha=0.7,
        kd_temperature=4.0,
        kd_teacher_tag="resnet18_pageflip",
    )
    train_mobilenet_v2_kd(cfg)
