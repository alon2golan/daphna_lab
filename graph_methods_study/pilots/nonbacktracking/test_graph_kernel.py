import unittest

import numpy as np

from graph_kernel import heldout_walks, nonbacktracking_three


class GraphKernelTests(unittest.TestCase):
    def test_heldout_backtrack_removal_matches_enumerated_walks(self):
        graph = np.array([[1, 2], [0, 2], [0, 1]])
        test = np.array([[0, 1]])
        anchors = np.array([0, 2])
        walks = heldout_walks(graph, test, anchors)
        np.testing.assert_array_equal(walks[0], [[1, 0]])
        np.testing.assert_array_equal(walks[1], [[1, 2]])
        np.testing.assert_array_equal(walks[2], [[3, 2]])
        np.testing.assert_array_equal(walks[3], [[5, 6]])
        np.testing.assert_array_equal(nonbacktracking_three(graph, anchors, walks), [[1, 2]])

    def test_nonreciprocal_outgoing_edge_amplifies_no_return_walk(self):
        graph = np.array([[1], [2], [0]])
        test = np.array([[0]])
        anchors = np.array([0, 2])
        walks = heldout_walks(graph, test, anchors)
        np.testing.assert_array_equal(nonbacktracking_three(graph, anchors, walks), walks[2])


if __name__ == '__main__':
    unittest.main()
