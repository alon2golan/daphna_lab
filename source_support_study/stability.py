"""Attribute a protected propagation checkpoint to its original annotations."""

import numpy as np


def source_stability(graph, labeled, labels, probabilities, masks, block_size=32):
    """Return the worst single-anchor deletion margin under the recorded masks.

    Protected labeled rows never send. Their intermediate evidence is unused.
    Distinct paths from one annotation contribute to the same column.
    """
    rows, classes = probabilities.shape
    predicted = probabilities.argmax(axis=1)
    class_mass = np.zeros_like(probabilities)
    largest = np.zeros(rows, dtype=probabilities.dtype)

    for start in range(0, len(labeled), block_size):
        anchors = labeled[start:start + block_size]
        observed = labels[start:start + block_size]
        direct = np.ascontiguousarray(graph[anchors].T.toarray())
        influence = direct.copy()
        for senders, receivers in masks:
            updated = direct + graph @ (influence * senders[:, None])
            influence[receivers] = updated[receivers]
        for column, cls in enumerate(observed):
            class_mass[:, cls] += influence[:, column]
        winner_mass = np.where(predicted[:, None] == observed, influence, 0)
        np.maximum(largest, winner_mass.max(axis=1), out=largest)

    evidence = 1.0 / classes + class_mass
    total = evidence.sum(axis=1)
    rebuilt = evidence / total[:, None]
    unlabeled = np.ones(rows, dtype=bool)
    unlabeled[labeled] = False
    error = float(np.max(np.abs(rebuilt[unlabeled] - probabilities[unlabeled])))
    if not np.isfinite(error) or error > 3e-6:
        raise ValueError(f"Source contributions did not reconstruct the teacher: {error:g}")

    fraction = largest / total
    row_ids = np.arange(rows)
    winner_excess = class_mass[row_ids, predicted]
    class_mass[row_ids, predicted] = -np.inf
    runner_excess = class_mass.max(axis=1)
    # Subtract in evidence space so an exact tie stays a tie after normalization.
    margin = (winner_excess - largest - runner_excess) / (total - largest)
    winning_share = np.divide(largest, winner_excess, out=np.zeros_like(largest), where=winner_excess > 0)
    return margin, fraction, winning_share, error


def rerank(candidates, admitted, predicted, scores, confidence, classes):
    """Change membership while preserving every predicted-class admission count."""
    if np.any(admitted & ~candidates):
        raise ValueError("Original admission contains ineligible rows")
    chosen = np.zeros_like(admitted)
    quotas = np.bincount(predicted[admitted], minlength=classes)
    for cls in np.flatnonzero(quotas):
        rows = np.flatnonzero(candidates & (predicted == cls))
        order = np.lexsort((rows, ~admitted[rows], -confidence[rows], -scores[rows]))
        chosen[rows[order[:quotas[cls]]]] = True
    return chosen
