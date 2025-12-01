# monreader/training/distillation.py

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from loguru import logger

from MonReader.evaluation.metrics import classification_report_dict


# ---------------------------------------------------------------------
# Distillation loss: CE(targets, student) + KL(teacher || student)

class DistillationLoss(nn.Module):
    """
    Standard KD loss: hard CE + soft KL between teacher & student.

    L = alpha * CE(student_logits, targets)
        + (1 - alpha) * T^2 * KL(softmax(teacher/T) || softmax(student/T))
    T is the temperature.

    - Cross-entropy with true labels keeps the student grounded on the actual task.
    - KL with teacher soft targets encourages the student to mimic teacher's
      "dark knowledge" (relative probabilities across classes).
    - Temperature > 1 smooths the distributions so small logits differences
      matter more; the T^2 term keeps gradient magnitudes roughly comparable.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        temperature: float = 4.0,
        reduction: str = "batchmean",
    ) -> None:
        super().__init__()
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
        assert temperature > 0.0, "temperature must be positive"

        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        T = self.temperature

        # Hard-label loss (standard supervised CE).
        ce_loss = F.cross_entropy(student_logits, targets)

        # Soft target distributions from teacher and student at temperature T.
        # Note: teacher is assumed to be in eval/no-grad mode when called.
        log_p_student = F.log_softmax(student_logits / T, dim=1)
        p_teacher = F.softmax(teacher_logits / T, dim=1)

        # KL divergence between teacher and student distributions.
        # We use reduction="batchmean" (default in KD literature).
        kd_loss = F.kl_div(
            log_p_student,
            p_teacher,
            reduction=self.reduction,
        )

        # Scale KD term by T^2 as in Hinton et al.
        kd_loss = (T * T) * kd_loss

        # Combine hard and soft components.
        loss = self.alpha * ce_loss + (1.0 - self.alpha) * kd_loss
        return loss


# ---------------------------------------------------------------------
# KD training loop: one epoch over student with fixed teacher

def train_student_with_distillation(
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: Optional[GradScaler],
    kd_loss_fn: DistillationLoss,
    device: torch.device,
    epoch: int,
    phase: str = "KD-Train",
    use_amp: bool = True,
    log_interval: int = 20,
) -> Dict[str, float]:
    """
    Run one KD training epoch for the student using a fixed teacher.

    This mirrors `run_one_epoch`, but:
    - It always runs in *training* mode for the student.
    - It assumes the teacher is a frozen model (no grads, eval mode).
    - The loss is the KD combination of CE + KL, not just CE.

    Returns:
        metrics dict: {"loss", "accuracy", "precision", "recall", "f1"}.
    """
    teacher.eval()  # teacher should never train during KD
    student.train(True)

    running_losses: List[float] = []
    all_true: List[int] = []
    all_pred: List[int] = []

    for batch_idx, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Decide whether to use AMP for the student forward.
        if use_amp and device.type == "cuda":
            ctx = autocast(device_type="cuda")
        else:
            ctx = torch.enable_grad()

        # Teacher forward (no grad, can be outside autocast or inside; the
        # teacher is frozen, so precision/speed is less critical).
        with torch.no_grad():
            teacher_logits = teacher(inputs)

        with ctx:
            student_logits = student(inputs)
            loss = kd_loss_fn(student_logits, teacher_logits, targets)

        running_losses.append(loss.item())

        # Hard predictions from the student for metrics.
        preds = student_logits.argmax(dim=1)
        all_true.extend(targets.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())

        # Backprop + optimizer step with/without AMP.
        if use_amp and device.type == "cuda":
            assert scaler is not None, "GradScaler must be provided when use_amp=True"
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if batch_idx % log_interval == 0 or batch_idx == len(loader):
            logger.info(
                f"[{phase}] Epoch {epoch:03d} | "
                f"Batch {batch_idx:04d}/{len(loader):04d} | "
                f"KD Loss {loss.item():.4f}"
            )

    # Aggregate epoch metrics.
    metrics = classification_report_dict(all_true, all_pred)
    metrics["loss"] = float(sum(running_losses) / max(1, len(running_losses)))

    logger.info(
        f"[{phase}] Epoch {epoch:03d} | "
        f"Loss {metrics['loss']:.4f} "
        f"Acc {metrics['accuracy']:.4f} "
        f"Prec {metrics['precision']:.4f} "
        f"Rec {metrics['recall']:.4f} "
        f"F1 {metrics['f1']:.4f}"
    )

    return metrics
