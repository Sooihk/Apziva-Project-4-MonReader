from __future__ import annotations

import torch.nn as nn
from torchvision import models


class ResNet18Binary(nn.Module):
    """
    ResNet-18 backbone with a simple binary classifier head.

    The underlying torchvision model is in `self.model`.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        in_features = base.fc.in_features

        base.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        self.model = base

    def forward(self, x):
        return self.model(x)
