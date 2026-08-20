"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_al.pycls.core.config import cfg as _DEFAULT_CFG

_RUNTIME_CFG = _DEFAULT_CFG


def set_runtime_cfg(active_cfg):
    """Set the active cfg used by loss constructors.

    This is needed because train_al.py calls losses.get_loss_fun() without
    passing cfg, while run_noise_al_pipeline.py builds a per-run cloned cfg.
    """
    global _RUNTIME_CFG
    _RUNTIME_CFG = active_cfg


def _cfg():
    return _RUNTIME_CFG if _RUNTIME_CFG is not None else _DEFAULT_CFG


class GeneralizedCrossEntropyLoss(nn.Module):
    """GCE loss: (1 - p_y^q) / q.

    q -> 0 behaves like cross entropy.
    q=1 behaves closer to MAE-style robustness.
    """
    def __init__(self):
        super().__init__()
        self.q = float(getattr(_cfg().MODEL, "GCE_Q", 0.7))

    def forward(self, logits, targets):
        if self.q <= 1e-8:
            return F.cross_entropy(logits, targets)

        probs = F.softmax(logits, dim=1)
        py = probs.gather(1, targets.view(-1, 1)).clamp_min(1e-8).squeeze(1)
        loss = (1.0 - py.pow(self.q)) / self.q
        return loss.mean()


class BootstrapSoftLoss(nn.Module):
    """Soft bootstrapping loss.

    target = beta * one_hot(label) + (1-beta) * model_prob.detach()
    """
    def __init__(self):
        super().__init__()
        self.beta = float(getattr(_cfg().MODEL, "BOOTSTRAP_BETA", 0.8))

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)
        model_probs = F.softmax(logits, dim=1)
        hard_targets = F.one_hot(targets, num_classes=num_classes).float()
        soft_targets = self.beta * hard_targets + (1.0 - self.beta) * model_probs
        return -(soft_targets * log_probs).sum(dim=1).mean()


class BootstrapHardLoss(nn.Module):
    """Hard bootstrapping loss.

    target = beta * one_hot(label) + (1-beta) * one_hot(model_pred)
    """
    def __init__(self):
        super().__init__()
        self.beta = float(getattr(_cfg().MODEL, "BOOTSTRAP_BETA", 0.8))

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)
        hard_targets = F.one_hot(targets, num_classes=num_classes).float()
        pred_targets = F.one_hot(logits.detach().argmax(dim=1), num_classes=num_classes).float()
        boot_targets = self.beta * hard_targets + (1.0 - self.beta) * pred_targets
        return -(boot_targets * log_probs).sum(dim=1).mean()


# Supported loss functions
_loss_funs = {
    "cross_entropy": nn.CrossEntropyLoss,
    "gce": GeneralizedCrossEntropyLoss,
    "bootstrap_soft": BootstrapSoftLoss,
    "bootstrap_hard": BootstrapHardLoss,
}


def get_loss_fun(active_cfg=None):
    """Retrieves the loss function."""
    if active_cfg is not None:
        set_runtime_cfg(active_cfg)

    active = _cfg()
    loss_name = active.MODEL.LOSS_FUN
    assert loss_name in _loss_funs.keys(), \
        "Loss function '{}' not supported. Supported: {}".format(loss_name, sorted(_loss_funs))
    return _loss_funs[loss_name]().cuda()


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor
