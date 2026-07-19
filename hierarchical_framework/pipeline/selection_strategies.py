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
    device = device or ("cuda" if torch.cuda.is_available() else "cpu") # use the GPU if one is available, else fall back to CPU
    X = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device) # move the whole feature matrix onto the device once
    n = X.shape[0]
    covering, covered = [], [] # the two endpoints of every edge we keep: a point, and a point its delta-ball reaches
    for start in range(0, n, batch_size): # walk the points in row-batches so we never hold the full N x N distance matrix
        cur = X[start:start + batch_size]
        dist = torch.cdist(cur, X) # L2 distance from each point in this batch to every point -> (batch, N)
        under_delta = dist < delta # an edge exists wherever two points are closer than delta (a point also covers itself, dist 0)
        a, b = under_delta.nonzero(as_tuple=False).T # (row, col) of each surviving edge: labeling point a would cover point b
        covering.append((a + start).cpu().numpy()) # shift the batch-local row index back to a global point index
        covered.append(b.cpu().numpy())
    covering_point = np.concatenate(covering).astype(np.int64) # for each edge e: labeling covering_point[e] covers covered_point[e]
    covered_point = np.concatenate(covered).astype(np.int64)
    return covering_point, covered_point


# base strategy: the shared "label `budget` points this round" loop.
# subclasses implement choose_next() -- which unlabeled point to take.
class Selector:

    name = "base"

    def __init__(self, features, **kwargs):
        self.features = np.asarray(features)
        self.n = self.features.shape[0] # total number of data points available in dataset pool
        self.is_labeled = np.zeros(self.n, dtype=bool) # initializes  all as unlabeled
        self.coverage_gains = [] # per labeled point, how many new points it covered; only coverage strategies fill this, else it stays empty

    def select_round(self, budget):
        newly_labeled = [] # the points we label this round, handed back to the caller
        for _ in range(budget): # spend the round's label budget one point at a time
            i = self.choose_next() # ask the strategy which unlabeled point to take next
            if i is None: # unlabeled set ran out -> stop early
                break
            self.label(i) # move that point into the labeled set
            newly_labeled.append(i)
        return newly_labeled

    def choose_next(self):
        # each strategy decides which unlabeled point to label next (None if none remain)
        raise NotImplementedError

    def label(self, i):
        self.is_labeled[i] = True # mark point i as labeled; subclasses extend this (e.g. update coverage)


# ProbCover family: also tracks the delta-graph + which points are covered;
# subclasses override score().
class CoverageSelector(Selector):

    name = "coverage"

    def __init__(self, features, delta=0.65, batch_size=500, device=None, **kwargs):
        super().__init__(features, **kwargs)
        self.covering_point, self.covered_point = build_graph(features, delta, batch_size, device) # delta-graph edge list: labeling covering_point[e] covers covered_point[e]
        self.is_covered = np.zeros(self.n, dtype=bool) # is_covered[j] becomes True once a labeled point's delta-ball reaches point j
        self.delta = delta

    def score(self):
        # per-candidate ranking of the next point to label; subclasses define it
        raise NotImplementedError

    def choose_next(self):
        s = self.score().astype(np.float64) # score every point as a candidate to label next
        s[self.is_labeled] = -np.inf # rule out points already in the labeled set
        i = int(np.argmax(s)) # the best-scoring candidate
        if not np.isfinite(s[i]) or s[i] <= 0: # nothing left that covers new ground -> just take any unlabeled point
            unlabeled = np.where(~self.is_labeled)[0]
            if unlabeled.size == 0: # whole pool is labeled, nothing to return
                return None
            self.coverage_gains.append(0.0) # this label adds no new coverage
            return int(unlabeled[0])
        self.coverage_gains.append(float(s[i])) # record how many still-uncovered neighbors labeling i will cover
        return i

    def label(self, i):
        edge_reaches_uncovered = ~self.is_covered[self.covered_point] # for every edge, is the point it reaches still uncovered?
        newly = self.covered_point[edge_reaches_uncovered & (self.covering_point == i)] # the still-uncovered points that labeling i now covers
        self.is_covered[newly] = True # those points are now covered
        self.is_covered[i] = True # and i covers itself
        self.is_labeled[i] = True # finally, i joins the labeled set

    def coverage(self):
        return float(self.is_covered.mean()) # fraction of the pool currently covered


# classic ProbCover: label the point that covers the most still-uncovered points
class ProbCoverSelector(CoverageSelector):

    name = "probcover"

    def score(self):
        edge_reaches_uncovered = ~self.is_covered[self.covered_point] # keep only edges that reach a still-uncovered point
        return np.bincount(self.covering_point[edge_reaches_uncovered], minlength=self.n) # per point, count those edges = how many new points labeling it would cover


# baseline: choose uniformly at random among the unlabeled points
class RandomSelector(Selector):

    name = "random"

    def __init__(self, features, seed=0, **kwargs):
        super().__init__(features, **kwargs)
        self.rng = np.random.default_rng(seed) # seeded RNG so a run is reproducible

    def choose_next(self):
        unlabeled = np.where(~self.is_labeled)[0] # indices still in the unlabeled set
        if unlabeled.size == 0: # nothing left to label
            return None
        return int(self.rng.choice(unlabeled)) # pick one uniformly at random


# name -> class, so the CLI can select a strategy by string
SELECTORS = {
    ProbCoverSelector.name: ProbCoverSelector,
    RandomSelector.name: RandomSelector,
}
