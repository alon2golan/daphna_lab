import unittest

import numpy as np
from scipy.sparse import csr_matrix

from source_support_study.stability import source_stability, rerank


class StabilityTests(unittest.TestCase):
    def test_repeated_paths_to_one_annotation_are_one_source(self):
        graph = csr_matrix([
            [0, 0, 0, .9, .1, 0, 0],
            [0, 0, 0, 0, .3, 0, 0],
            [0, 0, 0, .2, .2, 0, 0],
            [.9, 0, .2, 0, 0, .6, 0],
            [.1, .3, .2, 0, 0, .6, .8],
            [0, 0, 0, .6, .6, 0, 0],
            [0, 0, 0, 0, .8, 0, 0],
        ], dtype=np.float32)
        labeled = np.array([0, 1, 2])
        labels = np.array([0, 0, 1])
        receivers = np.array([False, False, False, False, False, True, True])
        senders = np.array([False, False, False, True, True, False, False])

        direct = graph[labeled].T.toarray()
        influence = direct.copy()
        influence[receivers] += (graph @ (direct * senders[:, None]))[receivers]
        evidence = .5 + influence @ np.eye(2)[labels]
        probabilities = (evidence / evidence.sum(1, keepdims=True)).astype(np.float32)
        margin, fraction, winning_share, error = source_stability(
            graph, labeled, labels, probabilities, [(senders, receivers)], block_size=1,
        )

        # Row 5 receives anchor 0 along two different paths. Both vanish together.
        for row in (5, 6):
            expected = []
            for anchor in range(len(labeled)):
                after = evidence[row].copy()
                after[labels[anchor]] -= influence[row, anchor]
                expected.append((after[0] - after[1]) / after.sum())
            self.assertAlmostEqual(margin[row], min(expected), places=6)
        self.assertAlmostEqual(fraction[5], .6 / (1 + .6 + .18 + .24), places=6)
        self.assertAlmostEqual(winning_share[5], .6 / (.6 + .18), places=6)
        self.assertLess(error, 1e-6)

    def test_admission_matches_class_counts_and_excludes_other_rows(self):
        candidates = np.array([True, True, True, True, False, True, False, True])
        admitted = np.array([True, False, False, True, False, False, False, True])
        predicted = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        scores = np.array([.2, .3, .3, .1, .9, .4, .8, .1])
        confidence = np.array([.9, .7, .6, .9, .9, .6, .9, .8])
        mask = rerank(candidates, admitted, predicted, scores, confidence, 3)
        np.testing.assert_array_equal(np.flatnonzero(mask), [1, 5, 7])
        np.testing.assert_array_equal(
            np.bincount(predicted[admitted], minlength=3),
            np.bincount(predicted[mask], minlength=3),
        )


if __name__ == "__main__":
    unittest.main()
