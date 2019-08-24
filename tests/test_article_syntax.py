from datetime import datetime
import unittest

import numpy as np

import article.syntax


class TestSyntax(unittest.TestCase):
    def test_date_from_cbin_filename(self):
        CBIN_FILENAME = 'bf_song_repo/gy6or6/032212/gy6or6_baseline_220312_0836.3.cbin'
        dt_from_cbin = article.syntax.date_from_cbin_filename(CBIN_FILENAME)
        self.assertTrue(
            isinstance(dt_from_cbin, datetime)
        )
        self.assertTrue(
            dt_from_cbin.date() == datetime(2012, 3, 22, 8, 36).date()
        )
        self.assertTrue(
            dt_from_cbin.time() == datetime(2012, 3, 22, 8, 36).time()
        )

    def test_find_branch_point(self):
        trans_mat = np.asarray(
            [
                [0., 1.0, 0., 0.],
                [0., 0., 0.1, 0.9],
                [0., 0., 0., 1.0],
            ])
        labels = list('abcd')
        bp_inds, bp_lbl = article.syntax.find_branch_points(trans_mat, labels)
        self.assertTrue(
            len(bp_inds) == 1
        )
        self.assertTrue(
            bp_lbl == ['b']
        )


if __name__ == '__main__':
    unittest.main()
