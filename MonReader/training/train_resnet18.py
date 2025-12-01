# monreader/training/train_resnet18.py

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
from MonReader.models.resnet18 import ResNet18Binary
from MonReader.models.utils import freeze_batchnorm_running_stats
from MonReader.training.engine import run_one_epoch, evaluate, CheckpointManager


def train_resnet18(cfg: Config) -> Tuple[nn.Module, Dict[str, float], List[float]]:
    """Two-phase training for ResNet-18.
        Phase A ("linear probe"):
            * Freeze the convolutional backbone.
            * Train only the final classifier head.
        Phase B ("fine-tuning"):
            * Unfreeze the top residual block(s) plus the classifier.
            * Use lower LR on the backbone and higher LR on the head.
            * Optionally stabilize BatchNorm to cope with small batch sizes.

    - Phase A lets the new head adapt to the target dataset while keeping
      the ImageNet features intact; this is cheap and stable.
    - Phase B then nudges the high-level features to better match 
      page-flip domain without wrecking the pretrained representation.

    Returns:
        The best validation metrics observed during training (based on F1).
    """
    # ------------------------------------------------------------------
    # Setup: seed, device, data
    # ------------------------------------------------------------------
    set_seed(cfg.seed)  # make runs reproducible across data splits and init
    device = get_device()
    print(f"Using device: {device}")

    # Build training/validation DataLoaders once; both phases reuse them.
    dm = DataModule(cfg)
    train_loader, val_loader = dm.build_dataloaders()

    # ------------------------------------------------------------------
    # Model: ResNet18 with binary head
    # ------------------------------------------------------------------
    model = ResNet18Binary(
        num_classes=cfg.num_classes,
        pretrained=True,   # start from ImageNet weights
        dropout=0.4,
    )
    model.to(device)

    # ---------------------- Phase A: head-only -------------------------
    print("=== Phase A: training classifier head only ===")

    # Strategy: freeze all backbone parameters, then unfreeze only the
    # classifier head (model.model.fc).
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.model.fc.parameters():
        param.requires_grad = True

    # freeze BatchNorm running stats in the backbone
    # so that Phase A behaves like a "pure" linear probe (backbone weights
    # *and* BN statistics remain fixed; only the classifier adapts).
    freeze_batchnorm_running_stats(model, freeze_affine=True)

    # Standard multinomial log-loss for classification.
    criterion = nn.CrossEntropyLoss()

    # Optimizer sees only the head parameters (requires_grad=True).
    optimizer_head = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )

    # CosineAnnealingLR is a smooth, widely used schedule; here we let it
    # anneal across the head-training epochs.
    scheduler_head = CosineAnnealingLR(
        optimizer_head,
        T_max=max(1, cfg.epochs_head),
    )

    # GradScaler handles dynamic loss scaling when AMP is enabled in
    # `run_one_epoch` (for CUDA), helping numerical stability.
    scaler = GradScaler()

    # CheckpointManager will track the best val F1 and save weights/full ckpts.
    ckpt_mgr = CheckpointManager(output_dir=cfg.output_dir, tag=cfg.tag)
    val_f1_history: List[float] = []

    for epoch in range(1, cfg.epochs_head + 1):
        # Train on the head only.
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer_head,
            scaler=scaler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            phase="Head",  # label for logging
        )

        # Evaluate current head on validation set.
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            phase="Val-Head",
        )
        # Track val F1 per epoch (Phase A)
        val_f1 = float(val_metrics.get("f1", 0.0))
        val_f1_history.append(val_f1)

        # Step LR schedule once per epoch.
        scheduler_head.step()

        # Save checkpoints only if val F1 improved.
        ckpt_mgr.maybe_save(
            model=model,
            optimizer=optimizer_head,
            scheduler=scheduler_head,
            scaler=scaler,
            epoch=epoch,
            val_metrics=val_metrics,
        )

    print(f"Best F1 after Phase A: {ckpt_mgr.best_f1:.4f}")

    # Restore the best head-only weights before fine-tuning so that Phase B
    # starts from the strongest classifier discovered in Phase A.
    ckpt_mgr.load_best_weights(model, device)

    # -------------------- Phase B: fine-tune backbone ------------------
    print("=== Phase B: fine-tuning last ResNet block(s) ===")

    # Fine-tuning pattern for ResNet18:
    #   - Keep early layers frozen (generic features like edges/texture).
    #   - Unfreeze only the last residual block (layer4) and the classifier.
    # This minimizes overfitting while still letting high-level features
    # adapt to the page-flip domain.
    for name, param in model.model.named_parameters():
        param.requires_grad = False  # start from all frozen
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True

    # BatchNorm stabilization: set BN layers to eval-like behavior for
    # running stats (but keep affine params trainable if freeze_affine=False).
    # This is important when your batch size is small (common in CV tasks).
    freeze_batchnorm_running_stats(model, freeze_affine=False)

    # Build parameter groups with discriminative learning rates:
    #   - backbone_params (layer4) get a lower LR (cfg.lr_backbone)
    #   - head_params (fc) get a higher LR (cfg.lr_head)
    backbone_params = []
    head_params = []
    for name, param in model.model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("layer4"):
            backbone_params.append(param)
        elif name.startswith("fc"):
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

    # Track best validation metrics across Phase B.
    best_val_metrics = {"f1": ckpt_mgr.best_f1}

    for epoch in range(1, cfg.epochs_fine_tune + 1):
        # Use a global epoch index for logging so that log lines reflect the
        # full training timeline (Phase A + Phase B).
        epoch_idx = cfg.epochs_head + epoch

        # Fine-tuning step (backbone + head).
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer_ft,
            scaler=scaler,
            criterion=criterion,
            device=device,
            epoch=epoch_idx,
            phase="FineTune",
        )

        # Validation after fine-tuning step.
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch_idx,
            phase="Val-FineTune",
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

    print(f"Best val F1 after Phase B: {ckpt_mgr.best_f1:.4f}")

    # ------------------------------------------------------------------
    # Restore best weights for downstream evaluation/export
    # ------------------------------------------------------------------
    # We restore the best validation checkpoint so that anything that
    # uses this model later (test evaluation, KD teacher, deployment)
    # starts from the best-seen weights. Actual test evaluation will be
    # done in a separate, explicit step to avoid leaking test metrics into
    # model selection.
    ckpt_mgr.load_best_weights(model, device)
    # Attach model size (best weights file) in MB to metrics.
    if getattr(ckpt_mgr, "best_weights_path", None) is not None:
        model_size_mb = get_model_size_mb(ckpt_mgr.best_weights_path)
    else:
        model_size_mb = float("nan")
    best_val_metrics["model_size_mb"] = model_size_mb

    return model, best_val_metrics, val_f1_history


if __name__ == "__main__":
    # Default run: you can override fields here if needed, e.g.:
    #   cfg = Config(tag="resnet18_pageflip", batch_size=64)
    cfg = Config(tag="resnet18_pageflip")
    train_resnet18(cfg)
