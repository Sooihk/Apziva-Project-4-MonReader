from __future__ import annotations
from typing import Tuple
from torchvision import transforms

# RGB mean/std used by ImageNet-pretrained backbones (VGG, ResNet, EfficientNet, MobileNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build training and evaluation transforms.

    Training:
      - Resize short side to 256
      - RandomResizedCrop to img_size (robustness to framing/zoom)
      - RandomRotation up to ~8 degrees
      - Light ColorJitter
      - Normalize with ImageNet stats

    Eval/Test:
      - Resize to 256
      - CenterCrop to img_size
      - Normalize with ImageNet stats
    """
    train_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomRotation(8),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_tfm, eval_tfm