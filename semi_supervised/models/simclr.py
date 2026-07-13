"""SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.

Paper: https://arxiv.org/abs/2002.05709
Reference implementation: vturrisi/solo-learn (solo/methods/simclr.py).

This is a lean, plain-PyTorch port: an encoder (backbone) plus a projection
head. The contrastive loss lives in ..losses (simclr_nt_xent_loss) and the
training loop lives with the algorithm that uses this model, so the same
SimCLR encoder can be dropped into different experiments.
"""

import torch.nn as nn

from ..losses import simclr_nt_xent_loss


class SimCLR(nn.Module):
    """SimCLR encoder + projection head.

    Args:
        backbone: any nn.Module mapping an image batch (N, C, H, W) to a flat
            feature vector (N, features_dim). Bring your own (e.g. a ResNet
            from scan/ or torchvision with the classifier removed).
        features_dim: output width of the backbone.
        proj_hidden_dim: hidden width of the projection MLP (solo-learn: 2048).
        proj_output_dim: width of the projected space where the contrastive
            loss is applied (solo-learn: 256).
        temperature: NT-Xent temperature (solo-learn: 0.1).
    """

    def __init__(self, backbone, features_dim, proj_hidden_dim=2048,
                 proj_output_dim=256, temperature=0.1):
        super().__init__()
        self.backbone = backbone
        self.features_dim = features_dim
        self.temperature = temperature

        # Projection head, matching solo-learn's SimCLR: Linear -> ReLU ->
        # Linear, no batch-norm. The loss is computed on the projector output;
        # downstream tasks use the backbone features instead.
        self.projector = nn.Sequential(
            nn.Linear(features_dim, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    def forward(self, x):
        """Return both the backbone features and the projected features.

        Args:
            x: image batch (N, C, H, W).

        Returns:
            dict with:
                feats: backbone output (N, features_dim), used downstream.
                z:     projected output (N, proj_output_dim), used for the loss.
        """
        feats = self.backbone(x)
        z = self.projector(feats)
        return {"feats": feats, "z": z}

    def training_loss(self, x1, x2):
        """Convenience: contrastive loss for two augmented views of a batch.

        Args:
            x1, x2: two augmentations of the same images, each (N, C, H, W).

        Returns:
            Scalar NT-Xent loss.
        """
        z1 = self.forward(x1)["z"]
        z2 = self.forward(x2)["z"]
        return simclr_nt_xent_loss(z1, z2, temperature=self.temperature)
