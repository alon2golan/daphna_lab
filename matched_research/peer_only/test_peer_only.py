import unittest

import numpy as np

from filter_swap import peer_agreement, peer_aum_mask
from peer_only import peer_only_mask


class PeerOnlyTest(unittest.TestCase):
    def test_peer_score_alone_overrides_aum_until_exact_ties(self):
        rng = np.random.default_rng(8)
        z = np.r_[rng.normal([1, 0, 0], .1, (16, 3)), rng.normal([0, 0, 1], .1, (16, 3))]
        y = np.r_[np.zeros(13, dtype=int), np.ones(3, dtype=int),
                  np.zeros(3, dtype=int), np.ones(13, dtype=int)]
        agree = peer_agreement(z, y)
        aum = -agree
        base = np.zeros(len(y), dtype=bool)
        base[np.argsort(agree)[:16]] = True
        selected = peer_only_mask(z, y, aum, base)
        fused = peer_aum_mask(z, y, aum, base)
        for label in np.unique(y):
            rows = np.flatnonzero(y == label)
            n = int(base[rows].sum())
            np.testing.assert_array_equal(np.sort(agree[rows[selected[rows]]]), np.sort(agree[rows])[-n:])
        self.assertFalse(np.array_equal(selected, fused))

    def test_exact_peers_tie_prefers_original_retention(self):
        z = np.eye(5)
        y = np.zeros(len(z), dtype=int)
        keep = np.array([False, False, True, True, False])
        aum = np.array([100, 100, 1, -50, 200])
        np.testing.assert_array_equal(peer_only_mask(z, y, aum, keep), keep)

    def test_empty_and_full_class_quotas_are_preserved(self):
        z = np.eye(5)
        y = np.array([0, 0, 1, 1, 2])
        keep = np.array([False, False, True, True, True])
        np.testing.assert_array_equal(peer_only_mask(z, y, y, keep), keep)


if __name__ == '__main__':
    unittest.main()
