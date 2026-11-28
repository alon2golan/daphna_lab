import unittest

import numpy as np

from rank_weights import class_weights


class RankWeightTests(unittest.TestCase):
    def test_classwise_weights_average_one(self):
        labels = np.array([3, 3, 3, 5, 5, 9])
        scores = np.array([-1.0, 0.0, 5.0, -4.0, -4.0, 2.0])
        expected = np.array([2 / 3, 1.0, 4 / 3, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(class_weights(labels, scores), expected)


if __name__ == "__main__":
    unittest.main()
