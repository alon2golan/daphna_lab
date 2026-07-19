# Point-selection strategies for the AL pipeline -- the overridable "choice".
#
# The pool starts fully UNLABELED. Each round we CHOOSE some unlabeled points to
# label, then LABEL them (move them from the unlabeled set into the labeled set).
# A strategy only decides WHICH unlabeled point to take next; the choose->label
# round bookkeeping is shared here. Add a strategy by subclassing and registering
# it in SELECTORS.

import numpy as np
import torch


def build_graph(features, delta, batch_size=500, device=None):
    # pick GPU when available
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # features onto the device once
    X = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
    n = X.shape[0]
    xs, ys = [], []
    # build the directed delta-graph in row-batches so we never hold the full N x N
    for start in range(0, n, batch_size):
        cur = X[start:start + batch_size]
        # L2 distances from this batch to all points -> (b, N)
        dist = torch.cdist(cur, X)
        # edge x->y wherever distance < delta (strict <, like deep_al); the diagonal
        # (d=0) passes too, so every point covers itself
        under_delta = dist < delta
        # nonzero gives (row, col) index pairs of the surviving edges
        x, y = under_delta.nonzero(as_tuple=False).T
        # x is local to the batch; shift it to a global point index
        xs.append((x + start).cpu().numpy())
        ys.append(y.cpu().numpy())
    src = np.concatenate(xs).astype(np.int64)
    dst = np.concatenate(ys).astype(np.int64)
    return src, dst


# base strategy: the shared "label `budget` points this round" loop.
# subclasses implement choose_next() -- which unlabeled point to take.
class Selector:

    name = "base"

    def __init__(self, features, **kwargs):
        self.features = np.asarray(features)
        self.n = self.features.shape[0]
        # the labeled set, as a boolean over the whole pool: is_labeled[i] is True
        # once point i has been chosen and labeled (persists across rounds).
        self.is_labeled = np.zeros(self.n, dtype=bool)
        # score of each labeled point, in the order they were labeled
        # (coverage-family: how many still-uncovered neighbors it covered).
        self.pick_scores = []

    def select_round(self, budget):
        # choose -> label, up to `budget` times; return the points labeled now.
        newly_labeled = []
        for _ in range(budget):
            i = self.choose_next()
            if i is None:                 # unlabeled set is empty
                break
            self.label(i)
            newly_labeled.append(i)
        return newly_labeled

    def choose_next(self):
        # return the next unlabeled point to label (or None if none remain)
        raise NotImplementedError

    def label(self, i):
        # move point i from the unlabeled set into the labeled set;
        # subclasses extend this (e.g. update coverage).
        self.is_labeled[i] = True


# ProbCover family: also tracks the delta-graph + which points are covered;
# subclasses override score().
class CoverageSelector(Selector):

    name = "coverage"

    def __init__(self, features, delta=0.65, batch_size=500, device=None, **kwargs):
        super().__init__(features, **kwargs)
        # directed edges of the delta-graph, built once
        self.src, self.dst = build_graph(features, delta, batch_size, device)
        # covered[j] is True once some labeled point's delta-ball reaches point j
        self.covered = np.zeros(self.n, dtype=bool)
        self.delta = delta

    def score(self):
        # per-candidate score used to rank the next choice; override this
        raise NotImplementedError

    def choose_next(self):
        s = self.score().astype(np.float64)
        # never choose a point that is already in the labeled set
        s[self.is_labeled] = -np.inf
        i = int(np.argmax(s))
        # best candidate covers nothing new -> just take any unlabeled point
        if not np.isfinite(s[i]) or s[i] <= 0:
            unlabeled = np.where(~self.is_labeled)[0]
            if unlabeled.size == 0:
                return None
            self.pick_scores.append(0.0)          # this label covers nothing new
            return int(unlabeled[0])
        # s[i] = how many still-uncovered neighbors labeling i will cover
        self.pick_scores.append(float(s[i]))
        return i

    def label(self, i):
        # newly-covered points = i's out-edges whose target is still uncovered
        still_uncovered = ~self.covered[self.dst]
        newly = self.dst[still_uncovered & (self.src == i)]
        # those targets (and i itself) are now covered
        self.covered[newly] = True
        self.covered[i] = True
        self.is_labeled[i] = True

    def coverage(self):
        return float(self.covered.mean())


# classic ProbCover: label the point that covers the most still-uncovered points
class ProbCoverSelector(CoverageSelector):

    name = "probcover"

    def score(self):
        # keep only edges pointing at still-uncovered targets
        still_uncovered = ~self.covered[self.dst]
        # out-degree over those = how many new points each candidate would cover
        return np.bincount(self.src[still_uncovered], minlength=self.n)


# baseline: choose uniformly at random among the unlabeled points
class RandomSelector(Selector):

    name = "random"

    def __init__(self, features, seed=0, **kwargs):
        super().__init__(features, **kwargs)
        self.rng = np.random.default_rng(seed)

    def choose_next(self):
        unlabeled = np.where(~self.is_labeled)[0]
        if unlabeled.size == 0:
            return None
        return int(self.rng.choice(unlabeled))


# name -> class, so the CLI can select a strategy by string
SELECTORS = {
    ProbCoverSelector.name: ProbCoverSelector,
    RandomSelector.name: RandomSelector,
}
