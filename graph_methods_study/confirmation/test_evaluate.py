import unittest

import numpy as np

from evaluate import effective_support, walk_matrix


class EvidenceTests(unittest.TestCase):
    def test_root_flattens_repeated_source_votes(self):
        raw = np.array([[16, 1, 1], [1, 1, 1], [0, 0, 0.]], dtype=np.float64)
        result = effective_support(np.sqrt(raw))
        self.assertGreater(result[0], effective_support(raw)[0])
        np.testing.assert_allclose(result[1:], [3, 0])

    def test_sparse_product_preserves_multiple_walks_to_the_same_source(self):
        neighbor = np.array([[1, 2], [0, 2], [0, 1]])
        walk = walk_matrix(neighbor, 3)
        np.testing.assert_array_equal(walk @ (walk @ np.eye(3)), [[2, 1, 1], [1, 2, 1], [1, 1, 2]])


if __name__ == '__main__':
    unittest.main()
