"""Shared LowBudgetAUM detector for direct filtering and NAS/NPC.

Netta-aligned LowBudgetAUM choices used here:
- linear classifier on frozen self-supervised representations;
- fake-class samples are drawn from the unlabeled pool;
- ceil(|L| / number_of_classes) fake samples per detector repeat;
- 40 training epochs;
- SGD with batch size 64, lr=0.1, momentum=0.9, Nesterov, wd=1e-4;
- 80th percentile of the fake-class AUM distribution is the threshold;
- no extra predicted-noise-rate cap.

``fake_ratio`` is retained for experiment compatibility, but now scales Netta's
ceil(|L| / C) threshold-sample count instead of scaling |L| directly.
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


def _netta_fake_sample_count(
    labeled_size: int,
    num_classes: int,
    available_unlabeled: int,
    multiplier: float = 1.0,
) -> Tuple[int, int]:
    """Return (effective_count, Netta base count) for fake-class samples."""
    base_count = int(max(1, np.ceil(float(labeled_size) / float(max(1, num_classes)))))
    effective_count = int(
        max(1, min(int(available_unlabeled), np.ceil(max(0.0, float(multiplier)) * base_count)))
    )
    return effective_count, base_count


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
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    nesterov: bool = True,
    batch_size: int = 64,
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

    # Netta's implementation uses ceil(|L| / C) threshold samples. The old
    # shared implementation used |L| fake samples when fake_ratio=1, making the
    # fake class as large as all real classes combined. In the 100-label / 100-
    # class setting that raised the fake threshold above every labeled AUM.
    fake_ratio = max(0.0, float(fake_ratio))
    n_fake, netta_n_fake = _netta_fake_sample_count(n, num_classes, len(uset), fake_ratio)
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
        momentum = float(momentum)
        effective_nesterov = bool(nesterov) and momentum > 0.0
        opt = torch.optim.SGD(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
            momentum=momentum,
            nesterov=effective_nesterov,
        )
        epochs = int(max(1, epochs))
        batch_size = int(max(1, min(batch_size, xt.shape[0])))
        margin_sum = torch.zeros(xt.shape[0], device=device)
        margin_count = torch.zeros(xt.shape[0], device=device)
        for _ in range(epochs):
            permutation = torch.randperm(xt.shape[0], device=device)
            for start in range(0, xt.shape[0], batch_size):
                batch_indices = permutation[start:start + batch_size]
                logits = model(xt[batch_indices])
                batch_labels = yt[batch_indices]
                with torch.no_grad():
                    assigned = logits.gather(1, batch_labels.view(-1, 1)).squeeze(1)
                    other = logits.detach().clone()
                    other[torch.arange(batch_labels.numel(), device=device), batch_labels] = -float('inf')
                    margin_sum[batch_indices] += assigned.detach() - other.max(dim=1).values
                    margin_count[batch_indices] += 1
                loss = F.cross_entropy(logits, batch_labels)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        aum = (margin_sum / margin_count.clamp_min(1.0)).detach().cpu().numpy()

    labeled_aum = aum[:n]
    fake_aum = aum[n:]
    fake_percentile = float(np.clip(fake_percentile, 0.0, 100.0))
    threshold = float(np.percentile(fake_aum, fake_percentile))
    noisy = labeled_aum <= threshold
    return noisy.astype(bool), {
        'method': 'lowbudget_aum', 'epochs': int(epochs),
        'fake_percentile': fake_percentile, 'threshold': threshold,
        'n_fake': int(n_fake), 'netta_n_fake_before_multiplier': int(netta_n_fake),
        'fake_ratio': float(fake_ratio), 'fake_ratio_semantics': 'multiplier_on_ceil_lset_over_num_classes',
        'fake_source': 'unlabeled_pool', 'linear_on_frozen_ssl_features': True,
        'pred_noisy': int(noisy.sum()),
        'predicted_noise_rate': float(noisy.mean()) if len(noisy) else 0.0,
        'labeled_aum_mean': float(labeled_aum.mean()), 'labeled_aum_std': float(labeled_aum.std()),
        'fake_aum_mean': float(fake_aum.mean()), 'fake_aum_std': float(fake_aum.std()),
        'detector_batch_size': int(batch_size),
        'detector_lr': float(lr), 'detector_weight_decay': float(weight_decay),
        'detector_momentum': momentum, 'detector_nesterov': effective_nesterov,
    }
