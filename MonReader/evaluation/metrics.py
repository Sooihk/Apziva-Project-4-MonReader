# monreader/evaluation/metrics.py

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from MonReader.config import Config
from MonReader.data.datamodules import DataModule
from MonReader.training.engine import evaluate
from MonReader.utils import get_device

# ---------------------------------------------------------------------
# Basic classification metrics helper
# ---------------------------------------------------------------------


def classification_report_dict(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    """Compute core classification metrics and return them as a dict.

    
    - Obtain accuracy / precision / recall / F1 computation for each model.
    - `run_one_epoch` calls this at the end of each epoch.
    - Keeping it here avoids subtle metric drift between different parts
      of the pipeline.

    Binary classification task (page flip vs not flip) and use
    the standard sklearn implementations underneath.
    """
    # Convert Python lists to NumPy arrays so we can feed them into sklearn.
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Core scalar metrics. zero_division=0 avoids crashes in degenerate cases
    # (e.g., a model that predicts only one class in early epochs).
    acc = accuracy_score(y_true_arr, y_pred_arr)
    prec = precision_score(y_true_arr, y_pred_arr, zero_division=0)
    rec = recall_score(y_true_arr, y_pred_arr, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)

    # Return Python floats so the result can be JSON-ified,
    # logged, tabulated, or plotted without worrying about NumPy types.
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


# ---------------------------------------------------------------------
# Test-set evaluation helper
# ---------------------------------------------------------------------


def evaluate_on_test(
    model: nn.Module,
    cfg: Config,
    device: Optional[torch.device] = None,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Evaluate a trained model on the *test* split.

    Design:
    - Reuse the same `evaluate` function that validation uses so the
      behavior of metrics is consistent across val/test.
    - The `Config` + `DataModule` combo is the single source of truth
      for how to construct the test DataLoader.
    - Criterion defaults to CrossEntropyLoss since that's what training
      uses.

    Returns:
        A metrics dict with keys: loss, accuracy, precision, recall, f1.
    """
    # Auto-detect device if one is not provided explicitly.
    if device is None:
        device = get_device()

    # Default loss mirrors the training setup (CE for classification).
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Build the test loader using the same transforms / batch size config
    # as the training DataModule.
    dm = DataModule(cfg)
    test_loader = dm.get_test_loader()

    # Move model to device and ensure eval() mode (disables dropout, etc.).
    model.to(device)
    model.eval()

    # Delegate the heavy lifting to the generic engine.evaluate.
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        epoch=0,
        phase="Test",
    )
    return test_metrics


# ---------------------------------------------------------------------
# Loss distribution helper
# ---------------------------------------------------------------------


def get_loss_distribution(
    model: nn.Module,
    cfg: Config,
    device: Optional[torch.device] = None,
    criterion: Optional[nn.Module] = None,
    split: str = "test",
) -> List[float]:
    """Compute per-example losses over a dataset split.

    - For knowledge distillation, calibration analysis, or error analysis,
      a single average loss isn't enough; you often want to see how loss is
      distributed across samples (e.g., are there long tails of hard pages?).
    - This helper runs the model in eval mode and returns a *list* of loss
      values (one per sample) to feed KDE plot

    Args:
        model: Trained model (already loaded with desired weights).
        cfg: Global configuration (used to find data paths / batch size).
        device: Torch device; if None we auto-detect.
        criterion: Loss function; defaults to CrossEntropyLoss.
        split: Which split to use: "test" (default) or "train"/"val".

    Returns:
        A list of per-example loss values (floats).
    """
    if device is None:
        device = get_device()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    dm = DataModule(cfg)

    # Choose which split to iterate over. For now we reuse DataModule's helper methods
    if split == "test":
        loader = dm.get_test_loader()
    else:
        train_loader, val_loader = dm.build_dataloaders()
        if split == "train":
            loader = train_loader
        elif split == "val":
            loader = val_loader
        else:
            raise ValueError(f"Unknown split '{split}'. Use 'train', 'val', or 'test'.")

    model.to(device)
    model.eval()

    losses: List[float] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass in eval mode.
            logits = model(inputs)
            batch_loss = criterion(logits, targets)

            # If batch_loss is a scalar tensor, it represents the *average*
            # loss over the mini-batch. In that case we broadcast the same
            # scalar to every item in the batch to get a list of the correct
            # length.
            if batch_loss.ndim == 0:
                losses.extend([batch_loss.item()] * inputs.size(0))
            else:
                # If the loss function already returns per-sample losses,
                # just convert them to Python floats.
                losses.extend(batch_loss.detach().cpu().tolist())

    return losses
