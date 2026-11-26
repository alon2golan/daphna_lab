import unittest

import numpy as np

from active_audit import aligned_human_annotations, coverage_order, leave_one_out, path_counts, rank_audits


class ActiveAuditTests(unittest.TestCase):
    def test_heldout_walk_never_sends_evidence_to_training(self):
        train_neighbors = np.array([[1], [2], [0]])
        test_neighbors = np.array([[0], [1]])
        train_votes, test_votes = path_counts(train_neighbors, test_neighbors, np.array([0, 2]), 2)
        np.testing.assert_array_equal(train_votes, [[0, 1], [1, 0], [0, 0]])
        np.testing.assert_array_equal(test_votes, [[0, 0], [0, 1]])

    def test_leave_one_out_drops_own_vote_and_abstains_without_evidence(self):
        counts = np.array([[90, 2, 1], [0, 19, 0], [5, 1, 20]])
        margin, alternative = leave_one_out(counts, np.array([0, 1, 1]), np.eye(3))
        np.testing.assert_allclose(margin, [1, 0, 4 / 6])
        np.testing.assert_array_equal(alternative[[0, 2]], [1, 0])

    def test_diverse_rank_avoids_two_audits_with_identical_footprints(self):
        suspicion = np.array([1., .9, .8])
        footprints = [np.array([0, 1]), np.array([0, 1]), np.array([2, 3])]
        one_shot, diverse = rank_audits(suspicion, footprints, 4, 2)
        np.testing.assert_array_equal(one_shot, [0, 1])
        np.testing.assert_array_equal(diverse, [0, 2])

    def test_human_annotation_alignment_rejects_reordered_clean_labels(self):
        truth = np.array([2, 1])
        fixture = {'clean_label': np.array([1, 2]), 'noisy_label': np.array([3, 4])}
        with self.assertRaises(AssertionError):
            aligned_human_annotations(fixture, truth)

    def test_seeded_coverage_updates_remaining_marginal_gains(self):
        neighbors = np.array([[1], [0], [0], [2], [3]])
        np.testing.assert_array_equal(coverage_order(neighbors, 2, 1), [0, 3])


if __name__ == '__main__':
    unittest.main()
