"""BYOL: Bootstrap Your Own Latent.

Paper: https://arxiv.org/abs/2006.07733
Reference implementation: vturrisi/solo-learn (solo/methods/byol.py).

Lean, plain-PyTorch port. BYOL learns without negative pairs. There are two
networks:

    online   = backbone -> projector -> predictor   (trained by backprop)
    momentum = backbone -> projector                (a slow EMA copy; no grad)

For a pair of views, the online branch predicts the momentum branch's
projection of the *other* view, and vice versa. The momentum network is never
trained directly: after each step its weights are nudged toward the online
weights by an exponential moving average (EMA). The asymmetry (only the online
side has a predictor) plus the stop-gradient on the target is what keeps the
representation from collapsing to a constant.

The negative-cosine loss lives in ..losses (byol_loss_func); the training loop
lives with the algorithm that uses this model.
"""

import copy

import torch
import torch.nn as nn

from ..losses import byol_loss_func


def _mlp(in_dim, hidden_dim, out_dim):
    """Linear -> BatchNorm -> ReLU -> Linear, as used by BYOL's projector and
    predictor. The batch-norm in the hidden layer is part of what stabilizes
    BYOL, so it is kept (unlike the SimCLR projector, which has none)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class BYOL(nn.Module):
    """BYOL online + momentum networks.

    Args:
        backbone: any nn.Module mapping images (N, C, H, W) to flat features
            (N, features_dim). A momentum copy is made internally via deepcopy,
            so pass a fresh backbone you are not sharing elsewhere.
        features_dim: output width of the backbone.
        proj_hidden_dim: hidden width of the projector MLP (solo-learn: 2048).
        proj_output_dim: projector/predictor output width (solo-learn: 256).
        pred_hidden_dim: hidden width of the predictor MLP (solo-learn: 4096).
        base_tau: starting EMA momentum for the target network (solo-learn:
            0.996). Closer to 1 means the target moves more slowly.
        final_tau: EMA momentum at the end of training (solo-learn: 1.0), used
            by the optional cosine schedule in momentum_tau().
    """

    def __init__(self, backbone, features_dim, proj_hidden_dim=2048,
                 proj_output_dim=256, pred_hidden_dim=4096,
                 base_tau=0.996, final_tau=1.0):
        super().__init__()
        self.features_dim = features_dim
        self.base_tau = base_tau
        self.final_tau = final_tau

        # Online network: trained by backprop.
        self.backbone = backbone
        self.projector = _mlp(features_dim, proj_hidden_dim, proj_output_dim)
        self.predictor = _mlp(proj_output_dim, pred_hidden_dim, proj_output_dim)

        # Momentum (target) network: a copy of the online backbone + projector
        # with gradients disabled. It is updated only by update_momentum().
        self.momentum_backbone = copy.deepcopy(backbone)
        self.momentum_projector = copy.deepcopy(self.projector)
        for param in self.momentum_backbone.parameters():
            param.requires_grad = False
        for param in self.momentum_projector.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Online branch: backbone -> projector -> predictor.

        Returns a dict with:
            feats: backbone features (N, features_dim), used downstream.
            z:     projector output (N, proj_output_dim).
            p:     predictor output (N, proj_output_dim), compared to a target.
        """
        feats = self.backbone(x)
        z = self.projector(feats)
        p = self.predictor(z)
        return {"feats": feats, "z": z, "p": p}

    @torch.no_grad()
    def momentum_forward(self, x):
        """Target branch: momentum backbone -> momentum projector, no grad.

        Returns a dict with:
            feats: momentum backbone features (N, features_dim).
            z:     momentum projector output (N, proj_output_dim), the target.
        """
        feats = self.momentum_backbone(x)
        z = self.momentum_projector(feats)
        return {"feats": feats, "z": z}

    @torch.no_grad()
    def update_momentum(self, tau=None):
        """EMA update: nudge the momentum network toward the online network.

        Call this once per training step, after the optimizer step. The rule
        is  target = tau * target + (1 - tau) * online  applied to every
        parameter of both the backbone and the projector.

        Args:
            tau: EMA momentum in [0, 1). Defaults to self.base_tau. Pass a
                scheduled value from momentum_tau() to anneal it over training.
        """
        if tau is None:
            tau = self.base_tau
        online_modules = (self.backbone, self.projector)
        target_modules = (self.momentum_backbone, self.momentum_projector)
        for online, target in zip(online_modules, target_modules):
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.data.mul_(tau).add_(po.data, alpha=1 - tau)

    def momentum_tau(self, step, max_steps):
        """Cosine-annealed EMA momentum, from base_tau up to final_tau.

        Optional helper matching solo-learn's schedule: tau rises over training
        so the target network slows down as the online network converges. Pass
        the result to update_momentum(). If you don't want a schedule, just
        call update_momentum() with the constant base_tau.
        """
        import math
        cos = (math.cos(math.pi * step / max_steps) + 1) / 2
        return self.final_tau - (self.final_tau - self.base_tau) * cos

    def training_loss(self, x1, x2):
        """Symmetric BYOL loss for two augmented views of a batch.

        The online branch predicts each view; the momentum branch provides the
        target projection of the *other* view. The two directions are summed,
        matching solo-learn. Remember to call update_momentum() after the
        optimizer step in your training loop.

        Args:
            x1, x2: two augmentations of the same images, each (N, C, H, W).

        Returns:
            Scalar loss tensor.
        """
        p1 = self.forward(x1)["p"]
        p2 = self.forward(x2)["p"]
        z1 = self.momentum_forward(x1)["z"]
        z2 = self.momentum_forward(x2)["z"]
        return byol_loss_func(p1, z2) + byol_loss_func(p2, z1)
