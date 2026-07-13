"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_al.pycls.core.config import cfg

# Supported loss functions
_loss_funs = {
    'cross_entropy': nn.CrossEntropyLoss,
}

def get_loss_fun():
    """Retrieves the loss function."""
    assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), \
        'Loss function \'{}\' not supported'.format(cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSS_FUN]().cuda()


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor


def distillation_loss(preds, soft_targets, mask, temperature=3.0):
    """
    KL divergence loss for distillation from soft targets with temperature scaling.

    Temperature scaling (Hinton et al.) softens probability distributions so the model
    can learn from "dark knowledge" -- the relative similarities between non-dominant
    classes. Gradients from soft targets scale as 1/T^2, so the result is multiplied by
    T^2 to keep them balanced with the hard-label CE loss.

    Args:
        preds: model predictions (logits), shape (batch_size, num_classes)
        soft_targets: soft probability targets (e.g. from prepare_distillation_targets),
            shape (batch_size, num_classes)
        mask: boolean mask indicating which samples to use for distillation, shape (batch_size,)
        temperature: temperature for softening distributions (default: 3.0). Higher = softer.

    Returns:
        Scalar loss value (mean KL divergence over valid samples, scaled by T^2).
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device)

    soft_preds = F.log_softmax(preds / temperature, dim=1)

    # Power-based temperature scaling of the soft targets.
    soft_targets_tempered = torch.pow(soft_targets.clamp(min=1e-8), 1.0 / temperature)
    soft_targets_tempered = soft_targets_tempered / soft_targets_tempered.sum(dim=1, keepdim=True)

    kl = F.kl_div(soft_preds, soft_targets_tempered, reduction='none').sum(dim=1)

    masked_kl = kl * mask.float()
    loss = masked_kl.sum() / mask.float().sum().clamp(min=1)

    # Multiply by T^2 to keep gradients on the same scale as CE loss (Hinton et al.)
    return loss * (temperature ** 2)
