import unittest

import numpy as np
from scipy.sparse import csr_matrix

from npc_backup import radius_backup_batch
from netta_baselines import radius_batch


class RadiusBackupTests(unittest.TestCase):
    def test_matches_npc_at_zero_and_exhaustive_backup(self):
        rng = np.random.default_rng(88)
        mask = rng.random((19, 19)) > .66
        matrix = (mask | mask.T).astype(float)
        np.fill_diagonal(matrix, 1.)
        graph = csr_matrix(matrix)
        incoming = graph.tocsc()
        labeled = np.array([0, 2, 4, 7])
        noisy = np.array([True, False, False, True])
        for weighted in (False, True):
            seed = 3
            actual = radius_backup_batch(graph, incoming, labeled, noisy, weighted, np.random.RandomState(seed),
                                         count=7, backup=0.)
            exact = radius_batch(graph, incoming, labeled, noisy, weighted, np.random.RandomState(seed), batch_size=7)
            np.testing.assert_array_equal(actual, exact)
            selected = []
            expected_rng = np.random.RandomState(seed)
            demand = np.ones(len(matrix))
            if weighted:
                demand[matrix[labeled[noisy]].sum(axis=0) > 0] *= 1 - noisy.mean()
            for _ in range(7):
                centers = np.r_[labeled[~noisy], np.asarray(selected, dtype=int)]
                cover = matrix[centers].sum(axis=0)
                gain = matrix @ (demand * ((cover == 0) + .5 * (cover == 1)))
                gain[np.r_[labeled, np.asarray(selected, dtype=int)]] = -np.inf
                selected.append(expected_rng.choice(np.flatnonzero(gain == gain.max())))
            actual = radius_backup_batch(graph, incoming, labeled, noisy, weighted,
                                         np.random.RandomState(seed), count=7)
            np.testing.assert_array_equal(actual, selected)


if __name__ == "__main__":
    unittest.main()
