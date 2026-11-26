import unittest

import numpy as np

from dense_backup import gain, greedy


class BackupTests(unittest.TestCase):
    def test_gain_matches_objective_change_with_two_distinct_anchors(self):
        previous = np.array([[.9, .3, .1], [.5, .2, .7]])
        new = np.array([.6, .1, .8])
        weight = .5
        before = np.sort(previous, axis=0)
        after = np.sort(np.vstack((previous, new)), axis=0)[-2:]
        self.assertAlmostEqual(gain(new, before[1], before[0], weight),
                               (after[1] + weight * after[0]).sum() -
                               (before[1] + weight * before[0]).sum())

    def test_lazy_matches_brute_force_with_warm_start(self):
        values = np.random.default_rng(14).uniform(size=(15, 15))
        matrix = (values + values.T) / 2
        for weight in (0., .5):
            picked = [4, 9]
            for _ in range(8):
                objectives = []
                for candidate in range(len(matrix)):
                    support = np.sort(matrix[picked + [candidate]], axis=0)
                    score = (support[-1] + weight * support[-2]).sum()
                    objectives.append(-np.inf if candidate in picked else score)
                picked.append(int(np.argmax(objectives)))
            np.testing.assert_array_equal(greedy(matrix, [4, 9], 8, weight), picked[2:])


if __name__ == '__main__':
    unittest.main()
