from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from MonReader.config import Config
from MonReader.utils import get_device

# Training entrypoint for KD student
from MonReader.training.train_mobilenet_v2_kd import train_mobilenet_v2_kd

# Backbone model wrapers
from MonReader.models.resnet18 import ResNet18Binary
from MonReader.models.vgg16 import VGG16Binary
from MonReader.models.efficientnet_b0 import EfficientNetB0Binary
from MonReader.models.mobilenet_v2 import MobileNetV2Binary, create_mobilenetv2_student

# Test-set evaluation helper
from MonReader.evaluation.metrics import evaluate_on_test

# Plotting utilities for final KD report
from MonReader.evaluation.plots import display_academic_results_table, plot_kd_summary_figure

def _find_best_weights_path(cfg: Config, pattern_substring: str = "best_weights") -> Path:
    """Return the path to the most recent best-weights checkpoint.

    Scan cfg.output_dir (typically models/<tag>/checkpoints) and pick
    the lexicographically last file whose name starts with `cfg.tag` and
    contains `pattern_substring`.
    """
    ckpt_dir = cfg.output_dir

    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {ckpt_dir}. "
            f"Have you trained the model with tag={cfg.tag}?"
        )

    candidates = [
        p
        for p in ckpt_dir.iterdir()
        if p.is_file()
        and p.name.startswith(cfg.tag)
        and pattern_substring in p.name
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint files matching '{cfg.tag}*{pattern_substring}*' "
            f"found in {ckpt_dir}"
        )

    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def _load_model_from_best_weights(
    model: nn.Module,
    cfg: Config,
    device: torch.device,
    pattern_substring: str = "best_weights",
) -> nn.Module:
    """
    Load the best-validation weights into a freshly constructed model.
    """
    weights_path = _find_best_weights_path(cfg, pattern_substring=pattern_substring)
    print(f"Loading best weights for tag='{cfg.tag}' from: {weights_path}")

    state = torch.load(weights_path, map_location=device)

    # Support either a full checkpoint dict or a plain state_dict.
    if isinstance(state, dict) and "model_state" in state:
        state_dict = state["model_state"]
    elif isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------
# Main experiment: KD vs baseline models on the test set
# ---------------------------------------------------------------------


def run_kd_vs_baselines() -> Dict[str, Dict[str, float]]:
    """Train MobileNetV2-KD and compare against baselines on the test set.
        1. Train the MobileNetV2 KD student .
        2. Rebuild the KD student architecture and load its best weights.
        3. Rebuild VGG16, ResNet-18, EfficientNet-B0, and base MobileNetV2
           from their best-validation checkpoints.
        4. Evaluate all five models on the *test* dataset via `evaluate_on_test`.
        5. Aggregate the metrics into `model_results` and pass them to:
             - display_academic_results_table(model_results)
             - plot_kd_summary_figure(model_results)

    Returns:
        model_results: dict mapping model name -> test-set metrics dict.
    """
    # ------------------------------------------------------------------
    # 0) Setup and device
    base_cfg = Config()  
    device = get_device()
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1) Train MobileNetV2-KD student
    kd_cfg = Config(
        tag="mobilenet_v2_kd_pageflip",
        kd_alpha=base_cfg.kd_alpha,
        kd_temperature=base_cfg.kd_temperature,
        kd_teacher_tag="resnet18_pageflip",
    )

    print("=== Training MobileNetV2-KD student ===")
    kd_model, _ = train_mobilenet_v2_kd(kd_cfg)

    # Directly evaluate the returned KD model on the test set
    kd_test_metrics = evaluate_on_test(kd_model, kd_cfg, device=device)

    # ------------------------------------------------------------------
    # 3) Load baseline models from best-validation checkpoints
    # ResNet-18 baseline
    print("=== Loading ResNet-18 baseline for test evaluation ===")
    resnet_cfg = Config(tag="resnet18_pageflip")
    resnet_model = ResNet18Binary(
        num_classes=resnet_cfg.num_classes,
        pretrained=False,
        dropout=0.4,
    )
    resnet_model = _load_model_from_best_weights(resnet_model, resnet_cfg, device)
    resnet_test_metrics = evaluate_on_test(resnet_model, resnet_cfg, device=device)

    # VGG16 baseline
    print("=== Loading VGG16 baseline for test evaluation ===")
    vgg_cfg = Config(tag="vgg16_pageflip")
    vgg_model = VGG16Binary(
        num_classes=vgg_cfg.num_classes,
        pretrained=False,
        dropout=0.4,
    )
    vgg_model = _load_model_from_best_weights(vgg_model, vgg_cfg, device)
    vgg16_test_metrics = evaluate_on_test(vgg_model, vgg_cfg, device=device)

    # EfficientNet-B0 baseline
    print("=== Loading EfficientNet-B0 baseline for test evaluation ===")
    eff_cfg = Config(tag="efficientnet_b0_pageflip")
    eff_model = EfficientNetB0Binary(
        num_classes=eff_cfg.num_classes,
        pretrained=False,
        dropout=0.3,
    )
    eff_model = _load_model_from_best_weights(eff_model, eff_cfg, device)
    effb0_test_metrics = evaluate_on_test(eff_model, eff_cfg, device=device)

    # Base MobileNetV2 baseline
    print("=== Loading base MobileNetV2 baseline for test evaluation ===")
    mbv2_cfg = Config(tag="mobilenet_v2_pageflip")
    mbv2_model = MobileNetV2Binary(
        num_classes=mbv2_cfg.num_classes,
        pretrained=False,
        dropout=0.2,
    )
    mbv2_model = _load_model_from_best_weights(mbv2_model, mbv2_cfg, device)
    mbv2_test_metrics = evaluate_on_test(mbv2_model, mbv2_cfg, device=device)

    # ------------------------------------------------------------------
    # 4) Aggregate test metrics into a comparison dictionary
    # ------------------------------------------------------------------
    model_results: Dict[str, Dict[str, float]] = {
        "MobileNetV2_KD": kd_test_metrics,
        "MobileNetV2": mbv2_test_metrics,
        "ResNet18": resnet_test_metrics,
        "VGG16": vgg16_test_metrics,
        "EfficientNetB0": effb0_test_metrics,
    }

    # ------------------------------------------------------------------
    # 5) Display comparison table and KD summary figure
    # ------------------------------------------------------------------
    print("=== Test-set comparison: KD vs baselines ===")
    display_academic_results_table(model_results)
    plot_kd_summary_figure(model_results)


if __name__ == "__main__":
    run_kd_vs_baselines()
