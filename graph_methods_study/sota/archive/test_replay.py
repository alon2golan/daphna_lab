import tempfile
from pathlib import Path
import unittest

import numpy as np

from replay import ordered_csv


class ReplayTests(unittest.TestCase):
    def test_archive_order_is_preserved_across_global_to_local_join(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'acquisitions.csv'
            path.write_text('order,row_index,observed_annotation,sample_id\n1,4,3,e\n2,1,5,b\n3,2,1,c\n')
            global_rows, local_rows = ordered_csv(path, np.array([1, 2, 4]), np.array([7, 5, 1, 9, 3]),
                                                   np.array(list('abcde')), 3)
            np.testing.assert_array_equal(global_rows, [4, 1, 2])
            np.testing.assert_array_equal(local_rows, [2, 0, 1])

    def test_wrong_saved_annotation_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'acquisitions.csv'
            path.write_text('order,row_index,observed_annotation,sample_id\n1,1,7,b\n')
            with self.assertRaises(AssertionError):
                ordered_csv(path, np.array([1]), np.array([7, 5]), np.array(list('ab')), 1)


if __name__ == '__main__':
    unittest.main()
