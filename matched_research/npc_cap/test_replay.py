import unittest

import numpy as np
from replay import six_edge_counts, graph_correct


class CapReplayTests(unittest.TestCase):
    def test_six_edges_include_one_test_to_train_edge(self):
        train = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32)
        test = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float32)
        anchors = np.array([2, 0])
        expected = (test @ np.linalg.matrix_power(train, 5))[:, anchors]
        np.testing.assert_array_equal(six_edge_counts(train, test, anchors, 1), expected)

    def test_cap_uses_distinct_sources_before_adding_classes(self):
        paths = np.array([[16, 4, 3], [7, 0, 0]], dtype=np.float32)
        labels = np.array([5, 8, 8])
        truth = np.array([8, 5])
        self.assertEqual(graph_correct(paths, labels, truth), (1, 2))


if __name__ == "__main__":
    unittest.main()
