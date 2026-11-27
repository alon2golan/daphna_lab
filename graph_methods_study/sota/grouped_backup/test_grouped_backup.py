import unittest

import numpy as np

from backup_coverage import select
from grouped_backup import grouped_select, paired_observations


class GroupedBackupTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1789)
        self.matrix = rng.uniform(.01, .99, (19, 19))
        self.matrix[rng.random(self.matrix.shape) < .42] = 0
        np.fill_diagonal(self.matrix, 1)
        self.target, self.candidate = np.nonzero(self.matrix)
        self.similarity = self.matrix[self.target, self.candidate]
        self.warm = [17, 2, 8]

    def test_singleton_groups_match_existing_backup(self):
        n = len(self.matrix)
        expected = select(self.target, self.candidate, self.similarity, n, 16, 3, 9, .5, self.warm)
        actual = grouped_select(self.target, self.candidate, self.similarity, np.arange(n),
                                16, 3, 9, self.warm)
        np.testing.assert_array_equal(actual, expected)

    def test_group_partition_matches_exhaustive_greedy(self):
        groups = (np.arange(len(self.matrix)) // 4 + 7) % 5
        n = len(groups)
        tie = np.random.default_rng(9).random(n) * 1e-10
        chosen = []

        def objective(selected, demand):
            maxima = np.zeros((len(demand), 5))
            for row in selected:
                maxima[:, groups[row]] = np.maximum(maxima[:, groups[row]], self.matrix[demand, row])
            maxima.sort(axis=1)
            return (maxima[:, -1] + .5 * maxima[:, -2]).sum()

        for begin in range(0, 15, 3):
            demand = np.setdiff1d(np.arange(n), chosen)
            for offset in range(3):
                remaining = np.setdiff1d(np.arange(n), chosen)
                if begin == 0:
                    chosen.append(self.warm[offset])
                else:
                    scores = [objective([*chosen, row], demand) + tie[row] for row in remaining]
                    chosen.append(int(remaining[np.argmax(scores)]))
        actual = grouped_select(self.target, self.candidate, self.similarity, groups,
                                15, 3, 9, self.warm)
        np.testing.assert_array_equal(actual, chosen)

    def test_one_group_removes_backup(self):
        n = len(self.matrix)
        expected = select(self.target, self.candidate, self.similarity, n, 16, 3, 17, 0., self.warm)
        actual = grouped_select(self.target, self.candidate, self.similarity, np.zeros(n, dtype=int),
                                16, 3, 17, self.warm)
        np.testing.assert_array_equal(actual, expected)

    def test_correlated_corruption_shares_flips_within_stress_cells(self):
        truth = np.tile(np.arange(100), 4)
        groups = np.repeat(np.arange(16), 25)
        successor = (np.arange(100) + 1) % 100
        labels = paired_observations(truth, groups, successor, 70)
        iid = labels['iid50'] != truth
        cyclic = labels['cyclic50'] != truth
        cluster = labels['cluster50'] != truth
        iid_matched = labels['iid_rate_matched'] != truth
        np.testing.assert_array_equal(iid, cyclic)
        self.assertEqual(int(cluster.sum()), int(iid_matched.sum()))
        self.assertTrue(np.all(cluster == cluster.reshape(16, 25)[:, :1].repeat(25, axis=1).ravel()))
        self.assertTrue(np.any(cluster) and np.any(~cluster))
        common = iid & cluster
        np.testing.assert_array_equal(labels['iid50'][common], labels['cluster50'][common])


if __name__ == '__main__':
    unittest.main()
