# monreader/training/train_efficientnet_b0.py

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
from MonReader.models.efficientnet_b0 import EfficientNetB0Binary
from MonReader.models.utils import freeze_batchnorm_running_stats
from MonReader.training.engine import run_one_epoch, evaluate, CheckpointManager

# Helper: unfreeze last EfficientNet-B0 stages + classifier
def _unfreeze_effnet_b0_last_stages_and_classifier(
    model: EfficientNetB0Binary,
    last_k_feature_blocks: int = 2,
) -> None:
    """Unfreeze only the last 2 EfficientNet-B0 feature blocks + classifier.

    EfficientNet-B0 :
        - model.features: nn.Sequential([...]) of MBConv blocks
        - model.classifier: Dropout -> Linear

    The exact number of feature blocks can vary with torchvision versions,
    so we use the length of `features` and unfreeze the *last* K blocks by
    index instead of hard-coding specific indices.

    Args:
        model: Our EfficientNetB0Binary wrapper.
        last_k_feature_blocks: How many of the final feature blocks to unfreeze.
    """
    # Start from all parameters frozen.
    for param in model.model.parameters():
        param.requires_grad = False

    # Determine how many feature blocks exist.
    num_blocks = len(model.model.features)
    k = max(1, min(last_k_feature_blocks, num_blocks))
    cutoff = num_blocks - k  # indices >= cutoff will be unfrozen

    for idx, block in enumerate(model.model.features):
        if idx >= cutoff:
            for param in block.parameters():
                param.requires_grad = True

    # Always train the classifier in Phase B.
    for param in model.model.classifier.parameters():
        param.requires_grad = True


def train_efficientnet_b0(cfg: Config) -> Tuple[nn.Module, Dict[str, float], List[float]]:
    """Two-phase training for EfficientNet-B0 on the page-flip task.

    Phase A (linear probe):
        - Freeze the entire EfficientNet-B0 backbone (features).
        - Train only the classifier head for cfg.epochs_head epochs.

    Phase B (fine-tune):
        - Unfreeze the last few MBConv stages + classifier.
        - Use discriminative learning rates (lower for backbone, higher for head).
        - Stabilize BatchNorm running stats to cope with smaller batch sizes.

    Returns:
        A dict of the best validation metrics (based on F1) observed.
    """
    # ------------------------------------------------------------------
    # Setup: seed, device, data
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    dm = DataModule(cfg)
    train_loader, val_loader = dm.build_dataloaders()

    # ------------------------------------------------------------------
    # Model: EfficientNet-B0 with binary classifier head
    model = EfficientNetB0Binary(
        num_classes=cfg.num_classes,
        pretrained=True,
        dropout=0.3,
    )
    model.to(device)

    # ------------------------------------------------------------------
    # Phase A: linear probe (classifier only)
    print("=== EfficientNet-B0 Phase A: training classifier head only ===")

    # Freeze entire backbone + classifier, then unfreeze only classifier.
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.model.classifier.parameters():
        param.requires_grad = True

    # linear probe freeze BN running stats so the backbone behaves exactly like the pretrained ImageNet encoder.
    freeze_batchnorm_running_stats(model, freeze_affine=True)

    criterion = nn.CrossEntropyLoss()

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
        # Train head-only.
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer_head,
            scaler=scaler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            phase="EffNet-Head",
        )

        # Validate.
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            phase="EffNet-Val-Head",
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

    print(f"[EfficientNet-B0] Best F1 after Phase A: {ckpt_mgr.best_f1:.4f}")

    # Restore best head-only weights before fine-tuning.
    ckpt_mgr.load_best_weights(model, device)

    # ------------------------------------------------------------------
    # Phase B: fine-tune last MBConv blocks + classifier
    print("=== EfficientNet-B0 Phase B: fine-tuning last stages + classifier ===")

    # Unfreeze last few stages of the feature extractor and the classifier.
    _unfreeze_effnet_b0_last_stages_and_classifier(model, last_k_feature_blocks=2)

    # After changing requires_grad flags, stabilize BN running stats.
    # Keep running stats frozen but allow affine params to update
    # in the unfrozen blocks for a balance between stability and flexibility.
    freeze_batchnorm_running_stats(model, freeze_affine=False)

    # Build parameter groups: backbone vs head.
    backbone_params = []  # last K feature blocks
    head_params = []      # classifier

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
        epoch_idx = cfg.epochs_head + epoch

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer_ft,
            scaler=scaler,
            criterion=criterion,
            device=device,
            epoch=epoch_idx,
            phase="EffNet-FineTune",
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch_idx,
            phase="EffNet-Val-FineTune",
        )

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

    print(f"[EfficientNet-B0] Best val F1 after Phase B: {ckpt_mgr.best_f1:.4f}")

    # Restore best-validation checkpoint for downstream evaluation/export.
    ckpt_mgr.load_best_weights(model, device)
    # Attach model size (best weights file) in MB to metrics.
    if getattr(ckpt_mgr, "best_weights_path", None) is not None:
        model_size_mb = get_model_size_mb(ckpt_mgr.best_weights_path)
    else:
        model_size_mb = float("nan")
    best_val_metrics["model_size_mb"] = model_size_mb

    return model, best_val_metrics, val_f1_history


if __name__ == "__main__":
    # Example entrypoint. You can adjust Config fields as needed.
    cfg = Config(tag="efficientnet_b0_pageflip")
    train_efficientnet_b0(cfg)
