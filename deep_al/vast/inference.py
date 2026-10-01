"""Evidence/gate ports with paper corrections; propagation is experimental."""
import numpy as np
from scipy.signal import find_peaks


class UnresolvedPropagationError(ValueError):
    pass


def require_propagation(mode):
    if mode != "literal-paper":
        raise UnresolvedPropagationError(
            "Propagation is unresolved with Itai. No reproduction default is available. "
            "For an explicitly non-reproduction diagnostic, select --propagation literal-paper. "
            "See docs/VAST_REPRODUCTION_SPEC.md section 6.")


def normalize_evidence(evidence):
    # Author prepare_distillation_targets sum-normalization path, in float64.
    if not np.isfinite(evidence).all() or np.any(evidence <= 0):
        raise ValueError("Evidence must be finite and positive; numerical overflow is not convergence")
    totals = evidence.sum(axis=1, keepdims=True)
    if not np.isfinite(totals).all():
        raise ValueError("Evidence row sums overflowed")
    return evidence / totals


def seed_evidence(kernel, labeled_indices, labeled_labels, classes):
    """Adapted author labeled-contribution sum; only revealed labels enter here."""
    evidence = np.full((kernel.shape[0], classes), 1.0 / classes, dtype=np.float64)
    for cls in range(classes):
        sources = labeled_indices[labeled_labels == cls]
        if len(sources):
            evidence[:, cls] += np.asarray(kernel[:, sources].sum(axis=1)).ravel()
    return evidence


def propagate_literal(kernel, evidence, labeled_mask, max_rounds=None, observer=None):
    """Literal Eq.5 diagnostic, NOT the confirmed implementation behind results.

    Adapts author sparse multiplication/frozen-row structure, replacing its
    prior-subtracted reconstruction and fixed rounds with paper additive update.
    """
    current = np.asarray(evidence, dtype=np.float64).copy()
    labeled_mask = np.asarray(labeled_mask, dtype=bool)
    classes = current.shape[1]
    q = normalize_evidence(current)
    frozen = labeled_mask | (q.max(axis=1) >= 0.75)
    bound = int((~labeled_mask).sum()) + 1
    limit = bound if max_rounds is None else min(bound, max_rounds)
    if limit < 1:
        raise ValueError("Propagation round limit must be positive")
    if observer is not None:
        empty = np.zeros(len(current), dtype=bool)
        observer(0, q, empty, empty, frozen, empty)
    history = []
    for iteration in range(limit):
        confidence = q.max(axis=1)
        senders = confidence > 1.0 / classes
        receivers = ~frozen
        # Synchronous: every sender reads current, before any receiver is updated.
        updated = current.copy()
        updated[receivers] += (kernel @ (current * senders[:, None]))[receivers]
        new_q = normalize_evidence(updated)
        new_frozen = frozen | ((new_q.max(axis=1) >= 0.75) & ~labeled_mask)
        newly_frozen_mask = new_frozen & ~frozen
        newly_frozen = int(newly_frozen_mask.sum())
        if observer is not None:
            observer(iteration + 1, new_q, senders, receivers, new_frozen, newly_frozen_mask)
        history.append({"round": iteration + 1, "senders": int(senders.sum()),
                        "receivers_before": int(receivers.sum()), "newly_frozen": newly_frozen,
                        "max_confidence": float(new_q.max())})
        current, q, frozen = updated, new_q, new_frozen
        if newly_frozen == 0:
            return current, {"mode": "literal-paper", "confirmed_reproduction": False,
                             "termination": "no_newly_frozen", "rounds": len(history),
                             "history": history}
    raise RuntimeError("Propagation reached the configured cap without the paper stopping condition")


def mode_threshold(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if not len(scores) or scores.max() == scores.min():
        return 0.75, False
    histogram, edges = np.histogram(scores, bins=100, range=(scores.min(), scores.max()))
    smoothed = np.convolve(histogram, np.ones(3) / 3.0, mode="same")
    peaks = find_peaks(np.r_[-np.inf, smoothed, -np.inf], plateau_size=True)[0] - 1
    ranked = sorted(peaks, key=lambda p: (-smoothed[p], p))
    if not ranked:
        return 0.75, False
    primary = ranked[0]
    secondary = next((p for p in ranked[1:] if abs(p - primary) >= 5
                      and smoothed[p] >= 0.15 * smoothed[primary]), None)
    if secondary is None:
        return 0.75, False
    return float(edges[max(primary, secondary)]), True


def prepare_targets(evidence, labeled_mask):
    """Author soft targets/top-k class scaling, corrected per Appendix B.1."""
    probs = normalize_evidence(evidence)
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    base, bimodal = mode_threshold(confidence[~labeled_mask])
    top_n = max(1, len(probs) // probs.shape[1])
    means = np.partition(probs, len(probs) - top_n, axis=0)[-top_n:].mean(axis=0)
    thresholds = np.maximum(base * means / means.max(), 1.0 / probs.shape[1])
    admitted = (~labeled_mask) & (confidence > thresholds[predicted])
    return probs, confidence, admitted, {
        "base_threshold": base, "bimodal": bool(bimodal),
        "class_thresholds": thresholds.tolist(), "top_n_per_class": top_n,
        "admitted_count": int(admitted.sum()), "unlabeled_count": int((~labeled_mask).sum())}
