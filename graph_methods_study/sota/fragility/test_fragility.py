import unittest

import numpy as np
from scipy.sparse import csr_matrix

from fragility import source_fragility


class FragilityTests(unittest.TestCase):
    def test_deletion_breaks_a_strict_lead_at_the_threshold(self):
        paths = csr_matrix([[16, 1, 1], [8, 5, 5], [2, 2, 2], [0, 0, 0], [0, 4, 0]])
        weights = source_fragility(paths, np.array([0, 1, 1]), 3)
        np.testing.assert_allclose(weights, [1 + 16/18, 1 + 10/18, 1 + 4/6, 1, 2])

    def test_matches_removing_each_original_annotation(self):
        rng = np.random.default_rng(4)
        counts = rng.integers(0, 20, (60, 13))
        labels = rng.integers(0, 5, counts.shape[1])
        indicators = np.eye(5)[labels]
        weights = []
        for row in counts:
            vote = row @ indicators
            winner = vote.argmax()
            break_mass = 0
            for col, h in enumerate(row):
                if labels[col] != winner:
                    continue
                after = vote - h * indicators[col]
                if after[winner] <= np.delete(after, winner).max():
                    break_mass += h
            weights.append(1 + break_mass / max(1, row.sum()))
        np.testing.assert_allclose(source_fragility(csr_matrix(counts), labels, 5), weights)


if __name__ == "__main__":
    unittest.main()
