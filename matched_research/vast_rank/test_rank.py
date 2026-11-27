import unittest

import numpy as np

from rank import align_rank, crossfit_depth, fold_ids, replay_folds


class RankTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.array([[.50, .30, .10, .07, .03], [.60, .20, .10, .05, .05]])
        self.proposed = np.array([[.15, .02, .25, .08, .50], [.30, .30, .10, .20, .10]])

    def test_winner_swap_and_top_three(self):
        np.testing.assert_array_equal(align_rank(self.raw, self.proposed, 0), self.raw)
        np.testing.assert_array_equal(align_rank(self.raw, self.proposed, 1)[0], [.03, .30, .10, .07, .50])
        aligned = align_rank(self.raw, self.proposed, 3)
        np.testing.assert_array_equal(aligned[0], [.10, .03, .30, .07, .50])
        np.testing.assert_array_equal(aligned[1], [.60, .20, .05, .10, .05])
        np.testing.assert_array_equal(np.sort(aligned, axis=1), np.sort(self.raw, axis=1))

    def test_noisy_cross_entropy_difference(self):
        q = align_rank(self.raw, self.proposed, 3)
        observed_dist = np.array([.50, .125, .125, .125, .125])
        observed_delta = -np.log(q[0] / self.raw[0]) @ observed_dist
        clean_delta = -np.log(q[0, 0] / self.raw[0, 0])
        self.assertAlmostEqual(observed_delta, .375 * clean_delta)

    def test_select_by_pooled_observed_loss_with_identity_ties(self):
        self.assertEqual(crossfit_depth(self.raw, self.raw, np.array([0, 1]))[0], 0)
        self.assertEqual(crossfit_depth(self.raw, self.proposed, np.array([4, 1]))[0], 1)
        self.assertEqual(crossfit_depth(self.raw, self.proposed, np.array([2, 1]))[0], 3)

    def test_folds_are_seeded_and_partition_acquisition_positions(self):
        assignment = fold_ids(22)
        np.testing.assert_array_equal(assignment, fold_ids(22))
        np.testing.assert_array_equal(np.bincount(assignment), [5, 5, 4, 4, 4])


    def test_fold_prediction_omits_heldout_sources_and_labels(self):
        labeled = np.array([11, 1, 7, 3, 8, 6, 12, 9, 2, 13])
        labels = np.arange(len(labeled)) % 3
        assignment = fold_ids(len(labels))
        remaining = []

        def teacher(rows, annotations, heldout_rows):
            fold = len(remaining)
            keep = assignment != fold
            np.testing.assert_array_equal(rows, labeled[keep])
            np.testing.assert_array_equal(annotations, labels[keep])
            np.testing.assert_array_equal(heldout_rows, labeled[~keep])
            counts = np.bincount(annotations, minlength=3) + 1
            probabilities = np.tile(counts / counts.sum(), (len(heldout_rows), 1))
            remaining.append(counts)
            return probabilities, probabilities[:, ::-1]

        raw, proposal = replay_folds(labeled, labels, assignment, teacher)
        self.assertEqual(len(remaining), 5)
        for fold, counts in enumerate(remaining):
            np.testing.assert_allclose(raw[assignment == fold], [counts / counts.sum()] * 2)
        np.testing.assert_array_equal(proposal, raw[:, ::-1])


if __name__ == '__main__':
    unittest.main()
