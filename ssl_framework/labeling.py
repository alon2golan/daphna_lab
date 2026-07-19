"""Label-selection strategies for the semi-supervised algorithms.

A Selector answers one question: given a pool of `n_pool` unlabeled samples and
a budget of `n_labeled`, which indices should be labeled? The interface is kept
deliberately tiny so today's random choice can later be replaced by an
active-learning strategy (e.g. one backed by deep_al) without touching the
algorithms that consume it.

A future AL selector will need more inputs than random does (features, a model,
the current labeled set). Those belong as constructor arguments or extra
keyword arguments on select(); the (n_pool, n_labeled, seed) core stays the
contract every selector honors.
"""

from typing import List, Protocol

import numpy as np


class Selector(Protocol):
    """Anything that can choose which pool indices to label."""

    def select(self, n_pool: int, n_labeled: int, seed: int) -> List[int]:
        ...


class RandomSelector:
    """Pick `n_labeled` indices uniformly at random from range(n_pool).

    The choice is fully determined by `seed`, so the same seed always yields the
    same labeled set -- important for comparing runs.
    """

    def select(self, n_pool: int, n_labeled: int, seed: int = 0) -> List[int]:
        if n_labeled > n_pool:
            raise ValueError(f"n_labeled={n_labeled} exceeds n_pool={n_pool}")
        rng = np.random.default_rng(seed)
        chosen = rng.choice(n_pool, size=n_labeled, replace=False)
        return sorted(int(i) for i in chosen)


def class_counts(indices, targets, num_classes):
    """How many labeled points landed in each class.

    A random (or AL) selector does not spread evenly over classes, so we record
    the actual makeup of every labeled set. `targets` are the pool's true labels
    (index-aligned with the pool); returns a length-`num_classes` numpy array of
    counts, indexable by class id.
    """
    targets = np.asarray(targets)
    counts = np.bincount(targets[np.asarray(indices)], minlength=num_classes)
    return counts
