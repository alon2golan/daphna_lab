import unittest

import numpy as np

from winner_swap import swap_winners


class WinnerSwapTests(unittest.TestCase):
    def test_only_winner_entries_swap(self):
        raw = np.array([[.7, .1, .2], [.1, .2, .7], [.2, .6, .2]], dtype=np.float32)
        revised = np.array([[.1, .7, .2], [.2, .1, .7], [.4, .3, .3]], dtype=np.float32)
        swapped = swap_winners(raw, revised)
        expected = np.array([[.1, .7, .2], [.1, .2, .7], [.6, .2, .2]], dtype=raw.dtype)
        np.testing.assert_array_equal(swapped, expected)
        np.testing.assert_array_equal(swapped.argmax(axis=1), revised.argmax(axis=1))
        np.testing.assert_array_equal(np.sort(swapped, axis=1), np.sort(raw, axis=1))
        self.assertEqual(swapped.dtype, raw.dtype)
        self.assertEqual(raw[0].argmax(), 0)

    def test_tied_raw_winner_preserves_probability_multiset(self):
        raw = np.array([[.4, .4, .2]])
        revised = np.array([[.2, .3, .5]])
        swapped = swap_winners(raw, revised)
        np.testing.assert_array_equal(swapped, [[.2, .4, .4]])


if __name__ == "__main__":
    unittest.main()
