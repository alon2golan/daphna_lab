# ProbCover: the coverage-greedy strategy this whole framework is built around.

import numpy as np

from .base import CoverageSelector


# classic ProbCover: label the point that covers the most still-uncovered points
class ProbCoverSelector(CoverageSelector):

    name = "probcover"

    def score(self):
        edge_reaches_uncovered = ~self.is_covered[self.covered_point] # keep only edges that reach a still-uncovered point
        return np.bincount(self.labeling_point[edge_reaches_uncovered], minlength=self.n) # per point, count those edges = how many new points labeling it would cover
