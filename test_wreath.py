import sys
import pdb
import random
import numpy as np
import perm2
from yor import yor, load_yor
import unittest
from wreath import young_subgroup_yor, wreath_yor, mult, get_mat, WreathCycSn, cyclic_irreps, wreath_rep
from utils import load_irrep
sys.path.append('./cube')
from multi import *
from str_cube import *

class TestWreath(unittest.TestCase):
    '''
    def test_wreath(self):
        alpha = (0, 3, 2)
        _parts = ((), (2,1), (1,1))

        ydict = wreath_yor(alpha, _parts)

        ks = list(ydict.keys())
        vals = list(ydict.values())
        size = len(ydict) * len(vals)

        g = random.choice(ks)
        h = random.choice(ks)
        gh = perm2.Perm2.from_tup(g) * perm2.Perm2.from_tup(h)
        gh_mat = mult(g, h, ydict)
        self.assertTrue(np.allclose(gh_mat, get_mat(gh, ydict)))

        eye = perm2.Perm2.eye(len(g))
        eye_g_mat = mult(eye, g, ydict)
        eye_mat = get_mat(eye * perm2.Perm2.from_tup(g), ydict)
        self.assertTrue(np.allclose(eye_mat, eye_g_mat))
    '''

    def test_wreath_full(self):
        o1, p1 = get_wreath('YYRMRMWWRWRYWMYMGGGGBBBB') # 14
        o2, p2 = get_wreath('YYBWGYRWMRBWMRMGYBRBGGMW') # 3
        w1 = WreathCycSn.from_tup(o1, p1, order=3)
        w2 = WreathCycSn.from_tup(o2, p2, order=3)

        prod = w1 * w2
        o_prod = prod.cyc.cyc
        p_prod = prod.perm.tup_rep

        # load some pickle
        alpha = (5, 3, 0)
        parts = ((3,2), (2,1), ())
        cf = cyc_irrep_func = cyclic_irreps(alpha)

        start = time.time()
        print('Loading {} | {}'.format(alpha, parts))
        yor_dict = load_irrep('/local/hopan/cube/', alpha, parts)
        if yor_dict is None:
            exit()
        print('Done loading | {:.2f}s'.format(time.time() - start))

        wreath1 = wreath_rep(o1, p1, yor_dict, cyc_irrep_func)
        wreath2 = wreath_rep(o2, p2, yor_dict, cyc_irrep_func)
        wreath_mult = wreath1.dot(wreath2)
        wreath_dict = wreath_rep(o_prod, p_prod, yor_dict, cyc_irrep_func)
        self.assertTrue(np.allclose(wreath_mult, wreath_dict))

if __name__ == '__main__':
    unittest.main()
