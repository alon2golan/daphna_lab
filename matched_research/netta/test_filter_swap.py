import unittest

import numpy as np

from filter_swap import peer_agreement, peer_aum_mask


class PeerAumTest(unittest.TestCase):
    def test_singleton_classes_cannot_vote_for_themselves(self):
        z = np.eye(4)
        agreement = peer_agreement(z, np.arange(4))
        np.testing.assert_array_equal(agreement, np.zeros(4))

    def test_selects_same_class_quotas_with_supported_swap(self):
        z = np.eye(2)[[0, 0, 0, 0, 1, 1, 0, 1]]
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        score = np.array([4, 3, 2, 1, 4, 3, 2, 1])
        keep = np.array([True, True, True, False, True, True, True, False])
        variant = peer_aum_mask(z, y, score, keep)
        np.testing.assert_array_equal(
            np.bincount(y[variant]), np.bincount(y[keep])
        )
        self.assertFalse(variant[6])
        self.assertTrue(variant[7])

    def test_peer_ties_preserve_aum_threshold_mask(self):
        z = np.eye(4)
        y = np.zeros(4, dtype=int)
        score = np.array([0.2, 0.3, 0.3, 0.4])
        keep = np.array([False, False, True, True])
        np.testing.assert_array_equal(peer_aum_mask(z, y, score, keep), keep)


if __name__ == "__main__":
    unittest.main()
