"""Shared LowBudgetAUM detector for direct filtering and NAS/NPC.

Paper-explicit choices from Shafir et al.:
- linear classifier on frozen self-supervised representations;
- fake-class samples are drawn from the unlabeled pool;
- 40 training epochs;
- 80th percentile of the fake-class AUM distribution is the threshold;
- no extra predicted-noise-rate cap.

The paper does NOT specify the number of fake samples or the optimizer used
specifically for the LowBudgetAUM detector. Those remain configurable and keep
this project's previous defaults (fake_ratio=1.0, lr=0.1, weight_decay=0.0).
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def lowbudget_aum_noisy_mask(
    all_features: np.ndarray,
    all_labels: np.ndarray,
    lset: np.ndarray,
    uset: np.ndarray,
    *,
    num_classes: int,
    seed: int,
    epochs: int = 40,
    fake_ratio: float = 1.0,
    fake_percentile: float = 80.0,
    lr: float = 0.1,
    weight_decay: float = 0.0,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    lset = np.asarray(lset, dtype=np.int64)
    uset = np.asarray(uset, dtype=np.int64)
    all_labels = np.asarray(all_labels, dtype=np.int64)
    n = len(lset)
    if n < 2 or len(uset) == 0:
        return np.zeros(n, dtype=bool), {'reason': 'too_few_labeled_or_no_unlabeled'}
    y_l = all_labels[lset]
    if len(np.unique(y_l)) < 2:
        return np.zeros(n, dtype=bool), {'reason': 'too_few_labeled_classes'}
    x_l = _l2_normalize(all_features[lset])

    # Paper says fake examples come from U; their number is not specified.
    fake_ratio = max(0.0, float(fake_ratio))
    n_fake = int(max(1, min(len(uset), round(fake_ratio * n))))
    rng = np.random.default_rng(int(seed) + 1337 + n)
    fake_idx = rng.choice(uset, size=n_fake, replace=False)
    x_fake = _l2_normalize(all_features[fake_idx])

    fake_class = int(num_classes)
    x = np.concatenate([x_l, x_fake], axis=0).astype(np.float32)
    y = np.concatenate([y_l, np.full(n_fake, fake_class, dtype=np.int64)])
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    fork_devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    detector_seed = int(seed) + 4242 + n

    with torch.inference_mode(False), torch.enable_grad(), torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(detector_seed)
        if device.type == 'cuda': torch.cuda.manual_seed_all(detector_seed)
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        model = nn.Linear(xt.shape[1], int(num_classes) + 1).to(device)
        # Optimizer details are not specified by the LowBudgetAUM paper.
        opt = torch.optim.SGD(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        epochs = int(max(1, epochs))
        margin_sum = torch.zeros(xt.shape[0], device=device)
        for _ in range(epochs):
            logits = model(xt)
            with torch.no_grad():
                assigned = logits.gather(1, yt.view(-1, 1)).squeeze(1)
                other = logits.detach().clone()
                other[torch.arange(yt.numel(), device=device), yt] = -float('inf')
                margin_sum += assigned.detach() - other.max(dim=1).values
            loss = F.cross_entropy(logits, yt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        aum = (margin_sum / float(epochs)).detach().cpu().numpy()

    labeled_aum = aum[:n]
    fake_aum = aum[n:]
    threshold = float(np.percentile(fake_aum, float(fake_percentile)))
    noisy = labeled_aum <= threshold
    return noisy.astype(bool), {
        'method': 'lowbudget_aum', 'epochs': int(epochs),
        'fake_percentile': float(fake_percentile), 'threshold': threshold,
        'n_fake': int(n_fake), 'fake_ratio': float(fake_ratio),
        'fake_source': 'unlabeled_pool', 'linear_on_frozen_ssl_features': True,
        'pred_noisy': int(noisy.sum()),
        'predicted_noise_rate': float(noisy.mean()) if len(noisy) else 0.0,
        'labeled_aum_mean': float(labeled_aum.mean()), 'labeled_aum_std': float(labeled_aum.std()),
        'fake_aum_mean': float(fake_aum.mean()), 'fake_aum_std': float(fake_aum.std()),
        'paper_unspecified_fake_count': True,
        'paper_unspecified_detector_optimizer': True,
        'detector_lr_project_assumption': float(lr),
        'detector_weight_decay_project_assumption': float(weight_decay),
    }
