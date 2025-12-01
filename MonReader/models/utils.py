from __future__ import annotations

import torch.nn as nn


def freeze_batchnorm_running_stats(
    model: nn.Module,
    freeze_affine: bool = False,
) -> None:
    """
    Freeze BatchNorm running stats (and optionally affine params) in a model.

    - Sets all BatchNorm layers to eval() so they stop updating running_mean/var.
    - Optionally sets requires_grad=False for weight/bias (gamma/beta).

    Call this at the start of fine-tuning Phase B when your batch size
    is small and you want stable BN behaviour.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Stop updating running_mean / running_var
            m.eval()
            m.track_running_stats = False

            if freeze_affine:
                if m.weight is not None:
                    m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)


