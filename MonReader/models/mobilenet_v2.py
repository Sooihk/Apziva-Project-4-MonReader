from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from MonReader.config import Config
from MonReader.utils import get_device



class MobileNetV2Binary(nn.Module):
    """
    MobileNetV2 backbone with a binary classifier head.

    Exposes underlying torchvision model as `self.model`.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Load torchvision MobileNetV2
        base = models.mobilenet_v2(
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )

        # MobileNetV2 classifier: Dropout -> Linear
        in_features = base.classifier[-1].in_features

        base.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        self.model = base
    @property
    def features(self):
        return self.model.features
    @property
    def classifier(self):
        return self.model.classifier
    def forward(self, x):
        return self.model(x)


def create_mobilenetv2_student(
    cfg: Config,
    device: Optional[torch.device] = None,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Convenience factory for the KD student model.

    Usage:
        cfg = Config(tag="mobilenet_v2_pageflip_kd")
        device = get_device()
        student = create_mobilenetv2_student(cfg, device)
    """
    if device is None:
        device = get_device()

    student = MobileNetV2Binary(
        num_classes=cfg.num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
    student.to(device)
    return student
