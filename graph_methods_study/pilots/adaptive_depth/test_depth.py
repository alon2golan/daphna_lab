import unittest

import numpy as np

from depth import margin_score, neighbors_matrix


class DepthTests(unittest.TestCase):
    def test_many_paths_from_one_label_have_one_labels_evidence(self):
        weights = np.array([[30, 0, 0], [1, 1, 1], [20, 20, 20], [0, 0, 0]], dtype=float)
        scores = np.column_stack([weights.sum(axis=1), np.zeros(len(weights))])
        short, supported = margin_score(scores, np.linalg.norm(weights, axis=1))
        np.testing.assert_allclose(short[:3], [1, np.sqrt(3), np.sqrt(3)])
        self.assertEqual(short[3], -np.inf)
        np.testing.assert_array_equal(supported, [True, True, True, False])

    def test_walk_matrix_reads_neighbors_in_receiver_order(self):
        matrix = neighbors_matrix(np.array([[1], [2], [0]]), 3)
        np.testing.assert_array_equal(matrix @ np.array([8, 9, 7]), [9, 7, 8])


if __name__ == '__main__':
    unittest.main()
