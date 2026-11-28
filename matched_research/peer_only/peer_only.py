"""Rank the final LowBudgetAUM quota by local agreement alone."""

import numpy as np

from filter_swap import peer_agreement


def peer_only_mask(features, observed_labels, aum_scores, base_keep):
    observed = np.asarray(observed_labels)
    aum = np.asarray(aum_scores)
    keep = np.asarray(base_keep, dtype=bool)
    if not (observed.shape == aum.shape == keep.shape == (len(features),)):
        raise ValueError('The feature, label, AUM, and mask arrays must align')
    if not np.isfinite(aum).all():
        raise ValueError('AUM returned nonfinite scores')
    agreement = peer_agreement(features, observed)
    revised = keep.copy()
    for label in np.unique(observed):
        rows = np.flatnonzero(observed == label)
        quota = int(keep[rows].sum())
        if quota == 0 or quota == len(rows):
            continue
        ranking = np.lexsort((rows, -aum[rows], -keep[rows].astype(int), -agreement[rows]))
        revised[rows] = False
        revised[rows[ranking[:quota]]] = True
    return revised
