from __future__ import annotations

import torch.nn as nn
from torchvision import models

class VGG16Binary(nn.Module):
    """
    VGG16 backbone with a lightweight binary classifier head.

    Exposes the underlying torchvision model as `self.model` to
    freeze/unfreeze blocks via `self.model.features[...]` in your training code.
    """
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        # Handle both new (weights=...) and old (pretrained=...) APIs
        if pretrained:
            try:
                weights = models.VGG16_Weights.IMAGENET1K_V1  # type: ignore[attr-defined]
                base = models.vgg16(weights=weights)
            except AttributeError:
                base = models.vgg16(pretrained=True)
        else:
            try:
                base = models.vgg16(weights=None)
            except TypeError:
                base = models.vgg16(pretrained=False)

        in_feats = base.classifier[6].in_features  # 4096
        base.classifier[6] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feats, num_classes),
        )
        self.model = base

        
    def forward(self, x):
        return self.model(x)