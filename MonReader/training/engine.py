# monreader/training/engine.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from loguru import logger

from MonReader.evaluation.metrics import classification_report_dict
from MonReader.utils import ensure_dir


# ---------------------------------------------------------------------
# Core training / evaluation loops
# ---------------------------------------------------------------------

def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    scaler: Optional[GradScaler],
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    phase: str = "Train",
    use_amp: bool = True,
    log_interval: int = 20,
) -> Dict[str, float]:
    """
    * Run one epoch of training or evaluation over loader.
    Use function for both train and eval to avoid drift between code paths.
    Update weights is controlled by whether an optimizer is passed. 
    - If `optimizer` is not None : training mode (update weights).
    - If `optimizer` is None     : evaluation mode (no grads, no updates).

    * AMP (automatic mixed precision) is enabled only when requested and when
      running on CUDA.

    * Accumulate predictions and targets for the entire epoch and delegate
      metric computation to `classification_report_dict`, so the logic for
      accuracy / precision / recall / F1 is centralized in one place.
    Returns a dict with:
        {
          "loss": float,
          "accuracy": float,
          "precision": float,
          "recall": float,
          "f1": float
        }
    """
    # If we have an optimizer, we are in training mode; otherwise evaluation.
    is_train = optimizer is not None
    # `train(True/False)` toggles dropout / batchnorm behavior appropriately.
    model.train(is_train)
    # accumaltors for epoch stats
    running_losses: List[float] = []
    all_true: List[int] = []
    all_pred: List[int] = []

    # Iterate over training batches
    for batch_idx, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)

        if is_train:
            # clear gradients
            optimizer.zero_grad(set_to_none=True)

        # Choose grad / no-grad context, and AMP if on CUDA
        if is_train:
            # Training with AMP on CUDA: use autocast so most ops run in
            # float16 / bfloat16 while critical ones stay in float32.
            if use_amp and device.type == "cuda":
                ctx = autocast(device_type="cuda")
            else:
                # Training without AMP: we want gradients, so enable_grad.
                ctx = torch.enable_grad()
        else:
            # Evaluation: we never want gradients; this also saves memory.
            ctx = torch.no_grad()

        # Forward + loss inside the chosen context.
        with ctx:
            logits = model(inputs)
            loss = criterion(logits, targets)

        # Store scalar loss for epoch-level averaging.
        running_losses.append(loss.item())

        # hard predictions (argmax over class dimension).
        preds = logits.argmax(dim=1)
        all_true.extend(targets.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())

        # Backward pass and optimizer step (training only).
        if is_train:
            # With AMP we scale the loss to prevent underflow, then
            # unscale and step the optimizer via the GradScaler.
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
                f"Loss {loss.item():.4f}"
            )

    # End of epoch: compute aggregate metrics from all predictions.
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



@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
    phase: str = "Val",
) -> Dict[str, float]:
    """Evaluation-only pass over `loader`.
    - `@torch.no_grad()` to be explicit that this function is strictly for evaluation and 
    should never track gradients.
    - Internally just call `run_one_epoch` with `optimizer=None` and
      `use_amp=False` to reuse the same aggregation logic and logging.
    """
    return run_one_epoch(
        model=model,
        loader=loader,
        optimizer=None,   # signals evaluation mode
        scaler=None,
        criterion=criterion,
        device=device,
        epoch=epoch,
        phase=phase,
        use_amp=False,    # AMP usually unnecessary for eval metrics
    )


# ---------------------------------------------------------------------
# Checkpoint saving / loading helpers
# ---------------------------------------------------------------------


def save_weights(model: nn.Module, path: Union[str, Path]) -> Path:
    """Save only the model weights (state_dict) to `path`.
    Separate "weights-only" checkpoints from "full" checkpoints to  easily ship 
    the smaller weights file to production without any optimizer / scheduler baggage.
    """
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(model.state_dict(), path)
    logger.info(f"Saved model weights to: {path}")
    return path


def save_full_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[_LRScheduler],
    scaler: Optional[GradScaler],
    epoch: int,
    metrics: Dict[str, float],
    path: Union[str, Path],
) -> Path:
    """Save a full training checkpoint.
    - model_state: so we can restore the exact model parameters later.
    - optimizer_state: to resume training with correct momentum, etc.
    - scheduler_state: so LR schedule continues from the right point.
    - scaler_state: to resume AMP training without losing dynamic range.
    - epoch: for bookkeeping / potential resume logic.
    - metrics: to know what performance this checkpoint achieved.

    This is the artifact you want when you might resume training; for pure
    inference you usually only need the smaller weights-only file.
    """
    path = Path(path)
    ensure_dir(path.parent)

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
    }

    torch.save(checkpoint, path)
    logger.info(f"Saved full checkpoint to: {path}")
    return path


def restore_weights(
    model: nn.Module,
    path: Union[str, Path],
    device: torch.device,
    strict: bool = True,
) -> nn.Module:
    """Load weights into `model` from `path` and move to `device`.

    Support two common checkpoint formats:
    1) A raw state_dict (what `save_weights` writes).
    2) A full checkpoint dict with a "model_state" key (what
       `save_full_checkpoint` writes).

    Both formats alllow the reuse the same utility for both
    training-time restarts and deployment-time loading.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()  # put model in eval mode by default after restore
    logger.info(f"Restored model weights from: {path}")
    return model


def restore_best_weights(
    model: nn.Module,
    ckpt_mgr: "CheckpointManager",
    device: torch.device,
    strict: bool = True,
) -> nn.Module:
    """Restore the best weights tracked by a CheckpointManager.

    Thin wrapper around `restore_weights` that uses the
    `best_weights_path` stored in the manager.

        - During training, you call `maybe_save` each epoch.
        - At the end, `ckpt_mgr.best_weights_path` points at the checkpoint
          with the highest observed validation F1.
        - Before final test evaluation or export, you call this helper to
          ensure the model is using those best weights.
    """
    if ckpt_mgr.best_weights_path is None:
        raise ValueError(
            "CheckpointManager has no best_weights_path yet. "
            "Did you call maybe_save() at least once?"
        )
    return restore_weights(model, ckpt_mgr.best_weights_path, device, strict=strict)


# ---------------------------------------------------------------------
# Checkpoint manager (best-by-F1)
# ---------------------------------------------------------------------


@dataclass
class CheckpointManager:
    """Track and save the best checkpoints based on validation F1.

    Purpose:
    - Centralize "best checkpoint" logic instead of scattering file-name
      checks and comparisons throughout training scripts.
    - Keep only two files for the best model: a small weights-only file and
      a full training checkpoint.
    - Ease to restore the best model for final testing or export.

    """

    # Directory where checkpoints for this experiment live.
    output_dir: Path
    # Short tag for this experiment (e.g., "resnet18_pageflip").
    tag: str

    # Internal state tracking the best metric seen so far.
    best_f1: float = -1.0
    best_weights_path: Optional[Path] = None
    best_full_ckpt_path: Optional[Path] = None

    def __post_init__(self) -> None:
        # Make sure the directory exists so save operations don't fail later.
        self.output_dir = ensure_dir(Path(self.output_dir))

    def maybe_save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[_LRScheduler],
        scaler: Optional[GradScaler],
        epoch: int,
        val_metrics: Dict[str, float],
        pattern_suffix_weights: str = "best_weights",
        pattern_suffix_full: str = "best_full",
    ) -> Tuple[bool, Optional[Path], Optional[Path]]:
        """Save weights + full checkpoint if val F1 improved.

        Compare the current validation F1 to the best so far. If it is
        strictly better, we update the internal record and write both a
        weights-only file and a full checkpoint file.

        Returns:
            improved: Whether a new best F1 was observed.
            weights_path: Path to the saved weights file (if improved).
            full_ckpt_path: Path to the saved full checkpoint (if improved).
        """
        current_f1 = float(val_metrics.get("f1", -1.0))

        if current_f1 <= self.best_f1:
            # No improvement: do nothing and keep previous best.
            logger.debug(
                f"[CheckpointManager] F1 did not improve: "
                f"{current_f1:.4f} <= {self.best_f1:.4f} (no save)"
            )
            return False, None, None

        logger.info(
            f"[CheckpointManager] New best F1: {current_f1:.4f} "
            f"(prev {self.best_f1:.4f}) — saving checkpoints."
        )
        self.best_f1 = current_f1

        # Filenames are kept simple and deterministic. If you want multiple
        # best checkpoints (e.g. with timestamps), you can extend this naming
        # scheme later.
        weights_path = self.output_dir / f"{self.tag}_{pattern_suffix_weights}.pth"
        full_ckpt_path = self.output_dir / f"{self.tag}_{pattern_suffix_full}.pt"

        save_weights(model, weights_path)
        save_full_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            metrics=val_metrics,
            path=full_ckpt_path,
        )

        self.best_weights_path = weights_path
        self.best_full_ckpt_path = full_ckpt_path

        return True, weights_path, full_ckpt_path

    def load_best_weights(
        self,
        model: nn.Module,
        device: torch.device,
        strict: bool = True,
    ) -> nn.Module:
        """Method version of `restore_best_weights`.
        This makes the typical call site in training scripts a bit cleaner:
            ckpt_mgr.load_best_weights(model, device)
        """
        if self.best_weights_path is None:
            raise ValueError("No best_weights_path stored yet; cannot load best weights.")
        return restore_weights(model, self.best_weights_path, device, strict=strict)
