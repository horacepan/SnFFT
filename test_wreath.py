import numpy as np
import perm2
from yor import yor, load_yor
import unittest
from wreath import young_subgroup_yor

class TestWreath(unittest.TestCase):
    def test_prod_perms(self):
        d = young_subgroup_yor((0, 4, 4), [(), (2, 2), (3, 1)])
        p1 = perm2.Perm2.from_tup((1,3,2,4))
        p2 = perm2.Perm2.from_tup((1,3,2,4))
        p3 = perm2.Perm2.from_tup((1,2,3,4))

        pp12 = perm2.ProdPerm(p1, p2)
        pp13 = perm2.ProdPerm(p1, p3)
        pp33 = perm2.ProdPerm(p3, p3)
        p1212 = pp12 * pp12
        p1213 = pp12 * pp13
        p3333 = pp33 * pp33

        y_12 = d[p1212.tup_rep]
        y_13 = d[p1213.tup_rep]
        y_33 = d[p3333.tup_rep]
        prod_12 = d[pp12.tup_rep].dot(d[pp12.tup_rep]) # should be identity
        prod_13 = d[pp12.tup_rep].dot(d[pp13.tup_rep])
        prod_33 = d[pp33.tup_rep].dot(d[pp33.tup_rep])
        self.assertTrue(np.allclose(prod_12, y_12))
        self.assertTrue(np.allclose(prod_13, y_13))
        self.assertTrue(np.allclose(prod_33, y_33))

if __name__ == '__main__':
    unittest.main()
