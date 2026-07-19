# Baseline strategies to compare ProbCover against: uniform random, and full supervision.

import numpy as np

from .selector_base import Selector


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


# baseline: reveal the WHOLE pool at once -- the full-supervision reference (upper end)
class AllLabeledSelector(Selector):

    name = "all"

    def select_round(self, budget):
        unlabeled = np.where(~self.is_labeled)[0] # everything not yet revealed (all of it on round 0)
        self.is_labeled[unlabeled] = True         # mark the entire pool labeled in one shot
        return [int(i) for i in unlabeled]        # one big set -> a single point, not a curve (budget ignored)
