"""Small numeric kernels used by geometric acquisition strategies."""
from __future__ import annotations

import torch


def squared_euclidean(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return pairwise squared Euclidean distances in the supplied geometry."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("distance inputs must be matrices with matching feature dimensions")
    return torch.cdist(left.float(), right.float()).square()


def rbf_kernel(left: torch.Tensor, right: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return torch.exp(-squared_euclidean(left, right) / (2 * sigma**2))
