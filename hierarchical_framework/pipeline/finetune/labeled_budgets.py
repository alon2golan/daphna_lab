# Builds the labeled subset the finetune trains on, split by ANNOTATION GRANULARITY:
#   fine-labeled points  -- their fine class is known (usable by both heads),
#   coarse-labeled points -- only their superclass is known (usable by the coarse head only).
# The count per group is the experiment axis. WHICH points get picked is delegated to a
# pluggable selector (random by default) so the selection strategy can be swapped later
# without touching the budget logic.

from dataclasses import dataclass, field

import numpy as np


# --- The labeled set: two (possibly overlapping) groups by annotation granularity ---

@dataclass
class LabeledSet:
    """The finetune's labeled data, split by what is known about each point."""
    fine_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))    # fine class known
    coarse_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))  # only superclass known

    @property
    def num_fine(self):
        return len(self.fine_indices)

    @property
    def num_coarse(self):
        return len(self.coarse_indices)

    @property
    def total(self):
        return self.num_fine + self.num_coarse


# --- Selection strategies (decoupled from the budget) ---
# A selector picks `count` indices out of `pool`. Swap this out for a non-random strategy
# (e.g. coreset / typiclust / margin) without changing anything below.

def random_selector(pool, count, rng):
    """Uniformly samples `count` indices from `pool` without replacement."""
    take = min(count, len(pool))                     # never ask for more than the pool holds
    return rng.choice(pool, take, replace=False)     # the actual random draw


def select_per_group(group_ids, per_group, selector, rng, exclude=None):
    """Runs `selector` once per distinct id in group_ids, returning all picked indices.
    `exclude` is a set of indices removed from every pool first (e.g. already-labeled points)."""
    exclude = set() if exclude is None else set(exclude)
    chosen = []
    for group in np.unique(group_ids):
        pool = np.where(group_ids == group)[0]                # all rows carrying this id
        if exclude:
            pool = np.array([i for i in pool if i not in exclude])  # drop excluded rows
        if len(pool):
            chosen.extend(int(i) for i in selector(pool, per_group, rng))
    return chosen


# --- Budget builder ---

def build_labeled_set(fine_labels, coarse_map, per_fine=None, per_super=None,
                      selector=random_selector, seed=0):
    """Constructs the LabeledSet.

    per_fine   -- take this many FINE-labeled points of each of the 100 fine classes.
    per_super  -- take this many COARSE-labeled points of each of the 20 superclasses;
                  these are treated as known only at the superclass level.
    selector   -- how to pick points from a pool (default random_selector).

    The fine and coarse groups are selected INDEPENDENTLY and may OVERLAP: the same point can
    be picked as a fine example and as a coarse example, since the two finetune signals are
    different (fine head vs coarse head). We currently allow this and do NOT charge extra for a
    shared point.
    # A shared point feeds each head at most once: the fine head trains only on fine rows and
    # the coarse head only on coarse rows (see two_head_finetune), so there is no double-feed.
    # TODO(cost): revisit annotation-cost accounting for points that land in BOTH groups.
    #   Options once we price this: charge as one fine label, charge fine+coarse (the sum), or
    #   an order-dependent rule. For now overlap is allowed and not charged.
    Returns a LabeledSet whose two index groups may share indices."""
    if per_fine is None and per_super is None:
        raise ValueError("give at least one of per_fine / per_super")

    rng = np.random.default_rng(seed)  # seeded generator so the labeled set is reproducible

    # Fine-labeled points: balanced per fine class
    fine_chosen = []
    if per_fine is not None:
        fine_chosen = select_per_group(fine_labels, per_fine, selector, rng)

    # Coarse-labeled points: balanced per superclass, selected INDEPENDENTLY (overlap allowed)
    coarse_chosen = []
    if per_super is not None:
        super_labels = coarse_map[fine_labels]                       # fine -> superclass per row
        coarse_chosen = select_per_group(super_labels, per_super, selector, rng)

    return LabeledSet(
        fine_indices=np.array(sorted(set(fine_chosen)), dtype=np.int64),
        coarse_indices=np.array(sorted(set(coarse_chosen)), dtype=np.int64),
    )
