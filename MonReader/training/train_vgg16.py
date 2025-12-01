# monreader/training/train_vgg16.py

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from MonReader.config import Config
from MonReader.utils import set_seed, get_device, get_model_size_mb
from MonReader.data.datamodules import DataModule
from MonReader.models.vgg16 import VGG16Binary
from MonReader.training.engine import run_one_epoch, evaluate, CheckpointManager


def _unfreeze_vgg_block5_and_classifier(model: VGG16Binary) -> None:
    """Helper to unfreeze only VGG16's last conv block (block 5) + classifier.

    VGG16's features is a nn.Sequential with indices roughly grouped as:
        - 0-4  : block1
        - 5-9  : block2
        - 10-16: block3
        - 17-23: block4
        - 24-30: block5

    We freeze everything by default, then turn `requires_grad=True` only for:
        - `features[24:]` (block 5)
        - the entire `classifier` stack
    """
    # Start from all-frozen.
    for param in model.model.parameters():
        param.requires_grad = False

    for name, param in model.model.named_parameters():
        parts = name.split(".")
        if not parts:
            continue

        if parts[0] == "features":
            # Name pattern: "features.<idx>.<something>"
            try:
                idx = int(parts[1])
            except (IndexError, ValueError):
                continue
            # Unfreeze block 5 layers only.
            if idx >= 24:
                param.requires_grad = True

        elif parts[0] == "classifier":
            # Always train the classifier in Phase B.
            param.requires_grad = True


def train_vgg16(cfg: Config) -> Tuple[nn.Module, Dict[str, float], List[float]]:
    """Two-phase training for VGG16 on the page-flip task.

    Phase A (linear probe / head-only):
        - Freeze the convolutional `features` backbone.
        - Train only the classifier head for `cfg.epochs_head` epochs.

    Phase B (fine-tune):
        - Unfreeze VGG block 5 (highest-level conv block) + classifier.
        - Train with discriminative learning rates for backbone vs head.

    Returns:
        Best validation metrics (dict) observed across training.
    """
    # Setup: seed, device, dataloaders
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    dm = DataModule(cfg)
    train_loader, val_loader = dm.build_dataloaders()

    # Model: VGG16Binary wrapper around torchvision.vgg16
    model = VGG16Binary(
        num_classes=cfg.num_classes,
        pretrained=True,
        dropout=0.4,
    )
    model.to(device)

    # Phase A: train classifier head only (linear probe)
    print("=== VGG16 Phase A: training classifier head only ===")

    # Freeze all backbone parameters (features + classifier), then re-enable
    # gradients only for the classifier head.
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    # Optimizer for head-only training.
    optimizer_head = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )

    scheduler_head = CosineAnnealingLR(
        optimizer_head,
        T_max=max(1, cfg.epochs_head),
    )

    scaler = GradScaler()

    ckpt_mgr = CheckpointManager(output_dir=cfg.output_dir, tag=cfg.tag)
    val_f1_history: List[float] = []

    for epoch in range(1, cfg.epochs_head + 1):
        # Train classifier head with backbone frozen.
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer_head,
            scaler=scaler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            phase="VGG16-Head",
        )

        # Validation after each epoch of head training.
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            phase="VGG16-Val-Head",
        )
        # Track val F1 per epoch (Phase A)
        val_f1 = float(val_metrics.get("f1", 0.0))
        val_f1_history.append(val_f1)

        scheduler_head.step()

        ckpt_mgr.maybe_save(
            model=model,
            optimizer=optimizer_head,
            scheduler=scheduler_head,
            scaler=scaler,
            epoch=epoch,
            val_metrics=val_metrics,
        )

    print(f"[VGG16] Best F1 after Phase A: {ckpt_mgr.best_f1:.4f}")

    # Restore the best classifier head found so far before moving to Phase B.
    ckpt_mgr.load_best_weights(model, device)

    # Phase B: fine-tune block 5 + classifier with discriminative LRs
    print("=== VGG16 Phase B: fine-tuning block 5 + classifier ===")

    # Unfreeze only block 5 and the classifier.
    _unfreeze_vgg_block5_and_classifier(model)

    # Build parameter groups for discriminative LR:
    backbone_params = []  # block 5 conv layers
    head_params = []      # classifier layers

    for name, param in model.model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("features."):
            backbone_params.append(param)
        elif name.startswith("classifier."):
            head_params.append(param)

    optimizer_ft = optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr_backbone},
            {"params": head_params, "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )

    scheduler_ft = CosineAnnealingLR(
        optimizer_ft,
        T_max=max(1, cfg.epochs_fine_tune),
    )

    best_val_metrics = {"f1": ckpt_mgr.best_f1}

    for epoch in range(1, cfg.epochs_fine_tune + 1):
        # Use global epoch index for consistent logging across phases.
        epoch_idx = cfg.epochs_head + epoch

        # Fine-tuning step: block 5 + classifier.
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer_ft,
            scaler=scaler,
            criterion=criterion,
            device=device,
            epoch=epoch_idx,
            phase="VGG16-FineTune",
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch_idx,
            phase="VGG16-Val-FineTune",
        )
        # Track val F1 per epoch (Phase B)
        val_f1 = float(val_metrics.get("f1", 0.0))
        val_f1_history.append(val_f1)

        scheduler_ft.step()

        improved, _, _ = ckpt_mgr.maybe_save(
            model=model,
            optimizer=optimizer_ft,
            scheduler=scheduler_ft,
            scaler=scaler,
            epoch=epoch_idx,
            val_metrics=val_metrics,
        )

        if improved:
            best_val_metrics = val_metrics

    print(f"[VGG16] Best val F1 after Phase B: {ckpt_mgr.best_f1:.4f}")

    # At the end of training, restore the best-validation checkpoint so this
    # model can be used for downstream evaluation or comparison.
    ckpt_mgr.load_best_weights(model, device)
    # Attach model size (best weights file) in MB to metrics.
    if getattr(ckpt_mgr, "best_weights_path", None) is not None:
        model_size_mb = get_model_size_mb(ckpt_mgr.best_weights_path)
    else:
        model_size_mb = float("nan")
    best_val_metrics["model_size_mb"] = model_size_mb

    return model, best_val_metrics, val_f1_history



if __name__ == "__main__":
    # Basic entrypoint: adjust cfg fields here if you want to run VGG16
    # with different batch size / epochs / tag.
    cfg = Config(tag="vgg16_pageflip")
    train_vgg16(cfg)
