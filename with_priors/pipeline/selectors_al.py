# Point-selection strategies for the AL pipeline -- the overridable "choice".
# Add a strategy by subclassing a class here and registering it in SELECTORS.

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
        mask = dist < delta
        # nonzero gives (row, col) index pairs of the surviving edges
        x, y = mask.nonzero(as_tuple=False).T
        # x is local to the batch; shift it to a global point index
        xs.append((x + start).cpu().numpy())
        ys.append(y.cpu().numpy())
    src = np.concatenate(xs).astype(np.int64)
    dst = np.concatenate(ys).astype(np.int64)
    return src, dst


# base strategy: the generic "pick budget points" loop; subclasses override choose()
class Selector:

    name = "base"

    def __init__(self, features, **kwargs):
        self.features = np.asarray(features)
        self.n = self.features.shape[0]
        # which points have been picked so far (persists across rounds)
        self.selected_mask = np.zeros(self.n, dtype=bool)
        # score of each pick, in order (coverage-family: uncovered neighbors it covered)
        self.pick_scores = []

    def select_round(self, budget):
        chosen = []
        # choose -> commit, budget times
        for _ in range(budget):
            i = self.choose()
            if i is None:                 # nothing left to pick
                break
            self.commit(i)
            chosen.append(i)
        return chosen

    def choose(self):
        # return the next point index to label (or None if exhausted)
        raise NotImplementedError

    def commit(self, i):
        # record the pick; subclasses extend this (e.g. update coverage)
        self.selected_mask[i] = True


# ProbCover family: keeps the delta-graph + covered set; subclasses override score()
class CoverageSelector(Selector):

    name = "coverage"

    def __init__(self, features, delta=0.65, batch_size=500, device=None, **kwargs):
        super().__init__(features, **kwargs)
        # directed edges of the delta-graph, built once
        self.src, self.dst = build_graph(features, delta, batch_size, device)
        # which points are already covered by some picked point's delta-ball
        self.covered = np.zeros(self.n, dtype=bool)
        self.delta = delta

    def score(self):
        # per-candidate score used to rank the next pick; override this
        raise NotImplementedError

    def choose(self):
        s = self.score().astype(np.float64)
        # never re-pick an already selected point
        s[self.selected_mask] = -np.inf
        i = int(np.argmax(s))
        # best pick adds nothing (all reachable already covered) -> take any free point
        if not np.isfinite(s[i]) or s[i] <= 0:
            free = np.where(~self.selected_mask)[0]
            if free.size == 0:
                return None
            self.pick_scores.append(0.0)          # this pick covers nothing new
            return int(free[0])
        # s[i] = how many still-uncovered neighbors this pick will cover
        self.pick_scores.append(float(s[i]))
        return i

    def commit(self, i):
        # edges from i whose target is still uncovered
        active = ~self.covered[self.dst]
        newly = self.dst[active & (self.src == i)]
        # those targets (and i itself) are now covered
        self.covered[newly] = True
        self.covered[i] = True
        self.selected_mask[i] = True

    def coverage(self):
        return float(self.covered.mean())


# classic ProbCover: pick the point covering the most still-uncovered points
class ProbCoverSelector(CoverageSelector):

    name = "probcover"

    def score(self):
        # keep only edges to uncovered targets
        active = ~self.covered[self.dst]
        # out-degree over those = how many new points each source would cover
        return np.bincount(self.src[active], minlength=self.n)


# baseline: pick uniformly at random among not-yet-selected points
class RandomSelector(Selector):

    name = "random"

    def __init__(self, features, seed=0, **kwargs):
        super().__init__(features, **kwargs)
        self.rng = np.random.default_rng(seed)

    def choose(self):
        free = np.where(~self.selected_mask)[0]
        if free.size == 0:
            return None
        return int(self.rng.choice(free))


# name -> class, so the CLI can pick a strategy by string
SELECTORS = {
    ProbCoverSelector.name: ProbCoverSelector,
    RandomSelector.name: RandomSelector,
}
