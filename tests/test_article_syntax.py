import unittest

import numpy as np

import article.syntax

git 
class TestSyntax(unittest.TestCase):
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
