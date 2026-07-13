"""Self-supervised loss functions, shared across models in ..models.

Kept separate from the model definitions so several models can reuse the same
loss and so the losses can be unit-tested on their own. Ported (lean, plain
PyTorch) from the reference implementation in vturrisi/solo-learn.
"""

import torch
import torch.nn.functional as F


def simclr_nt_xent_loss(z1, z2, temperature=0.1):
    """NT-Xent contrastive loss for SimCLR (the two-view formulation).

    Given two augmented views of the same batch, the positive pair for view-1
    sample i is view-2 sample i (and vice versa); every other sample in the
    batch, across both views, is a negative. This is the loss from the SimCLR
    paper (https://arxiv.org/abs/2002.05709), equivalent to solo-learn's
    simclr_loss_func in the single-GPU, two-view case.

    Args:
        z1: projected features of the first view, shape (N, D). Need not be
            L2-normalized; this function normalizes internally.
        z2: projected features of the second view, shape (N, D).
        temperature: softmax temperature (solo-learn default 0.1).

    Returns:
        Scalar loss tensor.
    """
    n = z1.shape[0]
    device = z1.device

    # L2-normalize so the dot product is a cosine similarity.
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # Stack both views: rows 0..n-1 are view 1, rows n..2n-1 are view 2.
    z = torch.cat([z1, z2], dim=0)                       # (2N, D)
    sim = torch.matmul(z, z.t()) / temperature           # (2N, 2N)

    # A sample is never its own positive: mask the diagonal to -inf so it
    # contributes zero after softmax.
    self_mask = torch.eye(2 * n, dtype=torch.bool, device=device)
    sim.masked_fill_(self_mask, float("-inf"))

    # For row i the positive column is i+n (view 1 -> view 2) or i-n
    # (view 2 -> view 1). Build that target index vector.
    targets = torch.cat([
        torch.arange(n, device=device) + n,
        torch.arange(n, device=device),
    ], dim=0)                                            # (2N,)

    # Cross-entropy over the similarity rows == the NT-Xent loss: each row is a
    # (2N-1)-way classification whose correct class is its positive partner.
    return F.cross_entropy(sim, targets)


def byol_loss_func(p, z):
    """BYOL loss: negative cosine similarity between a prediction and a target.

    BYOL (https://arxiv.org/abs/2006.07733) has no negatives. The online branch
    produces a prediction p; the momentum (target) branch produces z. We simply
    pull p toward z. z is detached so no gradient flows into the target branch
    (it is updated only by the EMA in the model, never by backprop).

    With unit-normalized vectors, 2 - 2*cos ranges in [0, 4] and equals the
    squared L2 distance between them; minimizing it aligns p with z. This
    matches solo-learn's byol_loss_func.

    Args:
        p: online predictor output, shape (N, D).
        z: momentum projector output (the target), shape (N, D).

    Returns:
        Scalar loss tensor.
    """
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z.detach()).sum(dim=-1).mean()
