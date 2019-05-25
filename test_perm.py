import unittest
from perm2 import *


class TestPerm(unittest.TestCase):
    def test_from_tup(self):
        tup = (1, 2, 3)
        perm = Perm2.from_tup(tup)
        self.assertTrue(perm.tup_rep == (1, 2, 3))

    def test_from_trans(self):
        trans = (3, 5)
        trans2 = (5, 3)
        exp_tup = (1, 2, 5, 4, 3, 6, 7)
        n = 7
        p1 = Perm2.from_trans(trans, n)
        p2 = Perm2.from_trans(trans2, n)
        self.assertTrue(p1.tup_rep == p2.tup_rep)
        self.assertTrue(p1.tup_rep == exp_tup)

    def test_perm_mult(self):
        t1 = (2, 3, 4, 1)
        t2 = (4, 1, 2, 3)
        p1 = Perm2.from_tup(t1)
        p2 = Perm2.from_tup(t2)
        self.assertTrue((p1 * p2).tup_rep == (p2 * p1).tup_rep)

        tup = (2, 3, 4, 1, 5)
        trans = (1, 5)
        p1 = Perm2.from_tup(tup)
        p2 = Perm2.from_trans(trans, len(tup))
        exp_12 = (5, 3, 4, 1, 2)
        exp_21 = (2, 3, 4, 5, 1)

        self.assertTrue((p1 * p2).tup_rep == exp_12)
        self.assertTrue((p2 * p1).tup_rep == exp_21)

if __name__ == '__main__':
    unittest.main()

