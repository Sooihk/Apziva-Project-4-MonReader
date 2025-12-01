# monreader/models/efficientnet_b0.py

from __future__ import annotations

import torch.nn as nn
from torchvision import models


class EfficientNetB0Binary(nn.Module):
    """
    EfficientNet-B0 backbone with a binary classifier head.

    - `self.model` is the underlying torchvision EfficientNet.
    - `self.features` exposes the convolutional backbone.
    - `self.classifier` exposes the classifier head.

    This makes it easier in training code to do things like:
        for p in model.features.parameters(): ...
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        # EfficientNet-B0 has classifier: Dropout -> Linear
        in_features = base.classifier[-1].in_features

        base.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        self.model = base

    @property
    def features(self) -> nn.Module:
        """Expose the convolutional backbone (all feature blocks)."""
        return self.model.features

    @property
    def classifier(self) -> nn.Module:
        """Expose the classifier head."""
        return self.model.classifier

    def forward(self, x):
        return self.model(x)
