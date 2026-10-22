"""
Soft-target distillation from an AL algorithm's evidence matrix (e.g.
BAYES_MISP's C_general): turns per-point, per-class kernel evidence into
normalized soft targets plus a confidence mask/weights, for use as an
auxiliary KL loss alongside standard cross-entropy training (see
losses.distillation_loss).

Not wired into the training loop (train_al.py) yet -- this module is
standalone and importable/testable on its own. A follow-up should attach it
to an actual training run (soft targets on the dataset, a modified
train_epoch, and cfg/CLI flags).

Ported from Itai's TypiClust experimental workspace, trimmed to the "legacy"
path: normalized soft-target + single scalar threshold, maxprob weighting.
Left out: the Bayesian/Otsu-gated distillation path, per-class thresholds,
the raw-C dual threshold, the omega_zeta/uniform weighting modes, and the
optional label-propagation step beforehand -- see propagation.py if that's
needed separately.
"""
import numpy as np
import torch
import torch.nn.functional as F


def prepare_distillation_targets(C_matrix, threshold, labeled_mask=None,
                                 train_labels=None, normalization='sum'):
    """
    Prepare soft targets and a distillation mask from a raw evidence matrix.

    Args:
        C_matrix: raw evidence matrix (N, num_classes), e.g. an AL algorithm's
            C_general -- torch tensor, on CPU or GPU.
        threshold: minimum normalized max-probability to select a point for
            distillation.
        labeled_mask: optional boolean (N,) -- when given (together with
            train_labels), used only for diagnostic logging (accuracy of the
            gated pseudo-labels among unlabeled points); does not affect
            which points are gated.
        train_labels: optional ground-truth labels (N,), for the same
            diagnostic logging as labeled_mask.
        normalization: 'sum' (divide each row by its sum) or 'softmax'.

    Returns:
        soft_probs: normalized soft targets, shape (N, num_classes), same
            device as C_matrix.
        distill_mask: boolean mask, shape (N,), True where max probability
            exceeds threshold.
        distill_weights: per-point weight (max probability), shape (N,).
    """
    if normalization == 'softmax':
        soft_probs = F.softmax(C_matrix, dim=1)
    else:
        soft_probs = C_matrix / torch.sum(C_matrix, dim=1, keepdim=True)

    max_probs = soft_probs.max(dim=1).values
    max_classes = soft_probs.argmax(dim=1)

    distill_mask = max_probs > threshold
    distill_weights = max_probs

    total_points = distill_mask.shape[0]
    num_distill_points = distill_mask.sum().item()
    print(f"[Distillation] {num_distill_points} / {total_points} points will participate "
          f"in distillation ({100.0 * num_distill_points / total_points:.1f}%)")

    _log_pseudo_label_accuracy(max_classes, distill_mask, labeled_mask, train_labels)

    return soft_probs, distill_mask, distill_weights


def _log_pseudo_label_accuracy(pred_classes, distill_mask, labeled_mask, train_labels=None):
    """Diagnostic logging only -- never affects training. Reports ground-truth
    accuracy of the gated distillation set's pseudo-labels (argmax of C), if
    train_labels are supplied, so gate quality can be read directly from the
    log instead of inferred from final test accuracy."""
    if train_labels is None:
        return
    try:
        train_labels = np.asarray(train_labels)
        pred = pred_classes.cpu().numpy() if torch.is_tensor(pred_classes) else np.asarray(pred_classes)
        mask = distill_mask.cpu().numpy() if torch.is_tensor(distill_mask) else np.asarray(distill_mask)
        mask = mask.astype(bool)
        if len(train_labels) != len(pred):
            return
        if labeled_mask is not None:
            lab = labeled_mask.cpu().numpy() if torch.is_tensor(labeled_mask) else np.asarray(labeled_mask)
            lab = lab.astype(bool)
            mask = mask & ~lab
            unlabeled = ~lab
        else:
            unlabeled = np.ones(len(pred), dtype=bool)
        if mask.sum() == 0 or unlabeled.sum() == 0:
            return
        gated_acc = (pred[mask] == train_labels[mask]).mean()
        all_acc = (pred[unlabeled] == train_labels[unlabeled]).mean()
        print(f"[Distill Diagnostic] gated pseudo-label accuracy: {100.0 * gated_acc:.2f}% "
              f"over {mask.sum()} points (baseline: {100.0 * all_acc:.2f}% over all "
              f"{unlabeled.sum()} unlabeled)")
    except Exception as e:
        print(f"[Distill Diagnostic] skipped ({e})")
