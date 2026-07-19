"""Kernel label propagation: turn a small labeled set into soft labels for the
whole pool, then hard-label the confident ones for the SupCon phase.

The idea (a lightweight, thresholded version of the coverage/"veracity" matrix
used by ProbCover/DCoM/MISP in deep_al): every pool point receives, per class,
a score built from its kernel figures to the *labeled* points of that class.
A figures floor makes far-apart points not help each other -- pairs whose
kernel value is below `sim_threshold` contribute exactly 0 to the matrix. The
per-class scores form a coverage matrix C (N x num_classes); normalising a row
gives a soft label, and points whose confidence clears `conf_threshold` get the
hard argmax label and join the SupCon training set as pseudo-labels.

Two thresholds are the main knobs, and both are meant to be swept:
    sim_threshold   -- kernel floor; below it, "these two points are unrelated".
    conf_threshold  -- how sure the soft label must be before we trust it.

Three figures kernels are implemented so runs can compare them; only RBF
(gaussian) is expected to be used first, the others are ready for later:
    gaussian    k = exp(-(d / h)^2)          (d = Euclidean distance)
    laplacian   k = exp(-(d / h))            (heavier tails than gaussian)
    cosine      k = cosine figures        (h ignored; floor still applies)

Nothing here imports solo-learn or trains anything -- it operates purely on the
frozen backbone features handed to it, so it is cheap to sweep.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

KERNELS = ("gaussian", "laplacian", "cosine")
AGGREGATIONS = ("max", "sum", "mean")
CONFIDENCE_METRICS = ("prob", "margin", "coverage")


@dataclass
class PropagationResult:
    """Everything a propagation run produces, for logging and for SupCon."""

    coverage: np.ndarray          # (N, num_classes) raw per-class kernel scores
    soft_labels: np.ndarray       # (N, num_classes) row-normalised (0 where no evidence)
    confidence: np.ndarray        # (N,) the chosen confidence metric per point
    hard_labels: np.ndarray       # (N,) argmax class, -1 where there is no evidence
    pseudo_indices: np.ndarray    # pool indices that cleared conf_threshold (excl. labeled)
    pseudo_labels: np.ndarray     # their assigned hard labels, index-aligned
    stats: dict                   # counts + optional pseudo-label accuracy


def _kernel_from_batch(feats_batch, anchors, kernel, bandwidth, sim_threshold, device):
    """Similarity of one batch of points to every anchor, floored at sim_threshold.

    Returns a (batch, n_anchors) tensor on `device`. Distances/similarities are
    computed in the batch to keep the full N x M matrix from ever materialising.
    """
    fb = feats_batch.to(device)
    an = anchors.to(device)
    if kernel == "cosine":
        fb_n = torch.nn.functional.normalize(fb, dim=1)
        an_n = torch.nn.functional.normalize(an, dim=1)
        sim = fb_n @ an_n.T
    else:
        dist = torch.cdist(fb, an)
        if kernel == "gaussian":
            sim = torch.exp(-((dist / bandwidth) ** 2))
        elif kernel == "laplacian":
            sim = torch.exp(-(dist / bandwidth))
        else:
            raise ValueError(f"unknown kernel {kernel!r}; choose from {KERNELS}")
    sim = torch.where(sim >= sim_threshold, sim, torch.zeros_like(sim))
    return sim


def median_bandwidth(features, sample=2000, device=None, seed=0):
    """The classic 'median heuristic' RBF bandwidth: the median pairwise Euclidean
    distance over a random sample of points. Distance-based kernels (gaussian,
    laplacian) need a width matched to the feature scale, which differs across
    encoders (SimCLR vs BYOL) and datasets -- this picks one automatically so a
    sweep can run hands-off. Cosine ignores bandwidth entirely.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feats = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    n = feats.shape[0]
    if n > sample:
        idx = np.random.default_rng(seed).choice(n, size=sample, replace=False)
        feats = feats[idx]
    feats = feats.to(device)
    d = torch.cdist(feats, feats)
    med = d[d > 0].median().item()
    return float(med)


def propagate(
    features,
    labeled_indices,
    labeled_targets,
    num_classes,
    *,
    kernel="gaussian",
    bandwidth=1.0,
    sim_threshold=0.0,
    conf_threshold=0.9,
    agg="max",
    confidence_metric="prob",
    max_pseudo=None,
    true_targets=None,
    batch_size=2048,
    device=None,
):
    """Propagate labels from a labeled subset to the whole pool via a kernel.

    Args:
        features: (N, D) frozen backbone features for the whole pool (np or torch).
        labeled_indices: indices into `features` whose labels are known.
        labeled_targets: labels of those indices, index-aligned with them.
        num_classes: number of classes (e.g. 100 for cifar100).
        kernel: "gaussian" | "laplacian" | "cosine" (see module docstring).
        bandwidth: kernel width h (ignored for cosine).
        sim_threshold: kernel floor; pair similarities below it contribute 0.
        conf_threshold: minimum confidence for a point to be pseudo-labeled.
        agg: how to pool a point's similarities to a class's anchors: max | sum | mean.
        confidence_metric: "prob" (max soft-label prob), "margin" (top1-top2 prob),
            or "coverage" (raw max kernel score, an absolute amount of evidence).
        max_pseudo: optional cap; keep only this many most-confident pseudo-labels.
        true_targets: optional (N,) ground truth, used ONLY to report pseudo-label
            accuracy (never fed into training).
        batch_size, device: compute knobs.

    Returns:
        PropagationResult.
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    if agg not in AGGREGATIONS:
        raise ValueError(f"agg must be one of {AGGREGATIONS}")
    if confidence_metric not in CONFIDENCE_METRICS:
        raise ValueError(f"confidence_metric must be one of {CONFIDENCE_METRICS}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feats = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    labeled_indices = np.asarray(labeled_indices)
    labeled_targets = np.asarray(labeled_targets)
    anchors = feats[labeled_indices]

    # Column groups: which anchor columns belong to each class.
    class_cols = [np.where(labeled_targets == c)[0] for c in range(num_classes)]

    n = feats.shape[0]
    coverage = torch.zeros(n, num_classes, device=device)
    counts = torch.tensor([len(c) for c in class_cols], device=device).clamp(min=1)

    for start in range(0, n, batch_size):
        sim = _kernel_from_batch(feats[start:start + batch_size], anchors,
                                 kernel, bandwidth, sim_threshold, device)
        for c, cols in enumerate(class_cols):
            if len(cols) == 0:
                continue
            block = sim[:, cols]
            if agg == "max":
                coverage[start:start + block.shape[0], c] = block.max(dim=1).values
            elif agg == "sum":
                coverage[start:start + block.shape[0], c] = block.sum(dim=1)
            else:  # mean
                coverage[start:start + block.shape[0], c] = block.sum(dim=1) / counts[c]

    coverage = coverage.cpu()
    row_sum = coverage.sum(dim=1)
    has_evidence = row_sum > 0

    # Soft labels: row-normalise where there is any evidence, else leave at 0.
    soft = torch.zeros_like(coverage)
    soft[has_evidence] = coverage[has_evidence] / row_sum[has_evidence].unsqueeze(1)

    hard = torch.where(has_evidence, coverage.argmax(dim=1), torch.full_like(row_sum, -1).long())

    # Confidence metric.
    top2 = soft.topk(min(2, num_classes), dim=1).values
    if confidence_metric == "prob":
        conf = soft.max(dim=1).values
    elif confidence_metric == "margin":
        conf = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)
    else:  # coverage: absolute evidence, not a probability
        conf = coverage.max(dim=1).values
    conf = torch.where(has_evidence, conf, torch.zeros_like(conf))

    # Candidates: confident, has evidence, and not already labeled.
    labeled_mask = torch.zeros(n, dtype=torch.bool)
    labeled_mask[labeled_indices] = True
    eligible = has_evidence & ~labeled_mask & (conf >= conf_threshold)
    pseudo_idx = torch.where(eligible)[0]

    if max_pseudo is not None and pseudo_idx.numel() > max_pseudo:
        keep = conf[pseudo_idx].topk(int(max_pseudo)).indices
        pseudo_idx = pseudo_idx[keep]

    pseudo_lab = hard[pseudo_idx]

    stats = {
        "n_labeled": int(labeled_mask.sum()),
        "n_pseudo": int(pseudo_idx.numel()),
        "n_no_evidence": int((~has_evidence).sum()),
        "kernel": kernel,
        "bandwidth": float(bandwidth),
        "sim_threshold": float(sim_threshold),
        "conf_threshold": float(conf_threshold),
        "agg": agg,
        "confidence_metric": confidence_metric,
    }
    if true_targets is not None and pseudo_idx.numel() > 0:
        tt = np.asarray(true_targets)
        stats["pseudo_accuracy"] = float((pseudo_lab.numpy() == tt[pseudo_idx.numpy()]).mean())

    return PropagationResult(
        coverage=coverage.numpy(),
        soft_labels=soft.numpy(),
        confidence=conf.numpy(),
        hard_labels=hard.numpy(),
        pseudo_indices=pseudo_idx.numpy(),
        pseudo_labels=pseudo_lab.numpy(),
        stats=stats,
    )
