"""Offline, apples-to-apples evaluation of a frozen backbone's features.

solo-learn logs val_acc1/5 and val_knn_acc1/5 *during* training from its online
heads, which is great as a live signal but is trained on whatever subset the
phase saw. To compare conditions fairly (raw SimCLR vs SimCLR->SupCon vs
SimCLR->SupCon+propagation) we also measure every backbone with one identical
yardstick here, computed from extracted features:

    linear probe   -- a fresh linear classifier trained on the frozen train
                      features and evaluated on val (top-1 / top-5).
    weighted k-NN  -- solo's own WeightedKNNClassifier over the same features,
                      so the number matches the val_knn_acc the framework logs.

Same features in, same procedure, for all three conditions.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_SOLO_ROOT = Path(__file__).resolve().parents[1] / "solo-learn"
if str(_SOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOLO_ROOT))
from solo.utils.knn import WeightedKNNClassifier  # noqa: E402


@torch.no_grad()
def _topk_acc(logits, targets, ks=(1, 5)):
    # Percentages, to match solo's WeightedKNNClassifier / accuracy_at_k units.
    maxk = min(max(ks), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    correct = pred == targets.unsqueeze(1)
    return {f"top{k}": 100.0 * float(correct[:, :min(k, maxk)].any(dim=1).float().mean()) for k in ks}


def linear_probe(train_feats, train_labels, val_feats, val_labels, *,
                 epochs=100, lr=1e-2, weight_decay=0.0, batch_size=1024,
                 device=None):
    """Train a linear classifier on frozen train features, score top-1/5 on val.

    A plain SGD linear probe -- the standard frozen-feature eval. Features are
    used as-is (the backbone output), matching solo's online classifier input.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    xtr = torch.as_tensor(np.asarray(train_feats), dtype=torch.float32)
    ytr = torch.as_tensor(np.asarray(train_labels), dtype=torch.long)
    xva = torch.as_tensor(np.asarray(val_feats), dtype=torch.float32).to(device)
    yva = torch.as_tensor(np.asarray(val_labels), dtype=torch.long).to(device)

    num_classes = int(max(ytr.max().item(), yva.max().item())) + 1
    clf = nn.Linear(xtr.shape[1], num_classes).to(device)
    opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    n = xtr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        clf.train()
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb = xtr[idx].to(device)
            yb = ytr[idx].to(device)
            opt.zero_grad()
            loss_fn(clf(xb), yb).backward()
            opt.step()
        sched.step()

    clf.eval()
    return _topk_acc(clf(xva), yva)


def weighted_knn(train_feats, train_labels, val_feats, val_labels, *,
                 k=20, distance_fx="cosine", T=0.07):
    """solo-learn's weighted k-NN top-1/5 over frozen features (matches val_knn_acc)."""
    knn = WeightedKNNClassifier(k=k, distance_fx=distance_fx, T=T)
    knn.update(train_features=torch.as_tensor(np.asarray(train_feats), dtype=torch.float32),
               train_targets=torch.as_tensor(np.asarray(train_labels), dtype=torch.long))
    knn.update(test_features=torch.as_tensor(np.asarray(val_feats), dtype=torch.float32),
               test_targets=torch.as_tensor(np.asarray(val_labels), dtype=torch.long))
    top1, top5 = knn.compute()
    return {"top1": float(top1), "top5": float(top5)}


def evaluate_features(train_feats, train_labels, val_feats, val_labels, *,
                      knn_k=20, knn_distance="cosine", probe_epochs=100):
    """Full offline scorecard for one backbone's features.

    Returns a flat dict with the four headline numbers, named to line up with
    solo's own metrics so they can sit in the same table / wandb panel:
        val_acc1, val_acc5          (linear probe)
        val_knn_acc1, val_knn_acc5  (weighted k-NN)
    """
    probe = linear_probe(train_feats, train_labels, val_feats, val_labels, epochs=probe_epochs)
    knn = weighted_knn(train_feats, train_labels, val_feats, val_labels,
                       k=knn_k, distance_fx=knn_distance)
    return {
        "val_acc1": probe["top1"], "val_acc5": probe["top5"],
        "val_knn_acc1": knn["top1"], "val_knn_acc5": knn["top5"],
    }
