"""Reorder AUM's final retention within each observed class."""

import numpy as np


def peer_agreement(features, observed_labels):
    """Estimate support using the nearest other acquired images."""
    z = np.asarray(features, dtype=np.float64)
    observed = np.asarray(observed_labels)
    count = len(z)
    if count < 2:
        return np.zeros(count)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.where(norms > 0, norms, 1)
    squares = np.einsum("ij,ij->i", z, z)
    distances = np.maximum(squares[:, None] + squares[None, :] - 2 * (z @ z.T), 0)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, : min(13, count - 1)]
    nearest = np.take_along_axis(distances, neighbors, axis=1)
    outer = nearest[:, -1:]
    scale = np.where(outer > 0, outer, 1)
    weights = np.exp(-nearest / scale)
    matches = observed[neighbors] == observed[:, None]
    return (weights * matches).sum(axis=1) / weights.sum(axis=1)


def _midranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (counts.cumsum() - (counts + 1) / 2)[inverse]


def peer_aum_mask(features, observed_labels, aum_scores, base_keep):
    observed = np.asarray(observed_labels)
    aum = np.asarray(aum_scores)
    keep = np.asarray(base_keep, dtype=bool)
    if not (observed.shape == aum.shape == keep.shape == (len(features),)):
        raise ValueError("The acquired feature, label, AUM, and mask arrays must align")
    if not np.isfinite(aum).all():
        raise ValueError("AUM returned nonfinite scores")
    agreement = peer_agreement(features, observed)
    revised = keep.copy()
    for label in np.unique(observed):
        rows = np.flatnonzero(observed == label)
        quota = int(keep[rows].sum())
        if quota == 0 or quota == len(rows):
            continue
        combined = _midranks(aum[rows]) + _midranks(agreement[rows])
        ranking = np.lexsort((rows, -aum[rows], -keep[rows].astype(int), -combined))
        revised[rows] = False
        revised[rows[ranking[:quota]]] = True
    return revised
