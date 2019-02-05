import sys
import pdb
import random
import numpy as np
import perm2
from yor import yor, load_yor
import unittest
from wreath import young_subgroup_yor, wreath_yor, mult, get_mat, WreathCycSn, cyclic_irreps, wreath_rep
from utils import load_irrep
from coset_utils import young_subgroup_perm
sys.path.append('./cube')
from multi import *
from str_cube import *

class TestWreath(unittest.TestCase):
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

    def test_wreath_full(self):
        o1, p1 = get_wreath('YYRMRMWWRWRYWMYMGGGGBBBB') # 14
        o2, p2 = get_wreath('YYBWGYRWMRBWMRMGYBRBGGMW') # 3
        o3, p3 = get_wreath('GGBWMGBBMRYGMRYRYBYRWWWM') # 4
        w1 = WreathCycSn.from_tup(o1, p1, order=3)
        w2 = WreathCycSn.from_tup(o2, p2, order=3)
        w3 = WreathCycSn.from_tup(o3, p3, order=3)

        prod12 = w1 * w2
        prod13 = w1 * w3
        o12 = prod12.cyc.cyc
        o13 = prod13.cyc.cyc
        perm12 = prod12.perm.tup_rep
        perm13 = prod13.perm.tup_rep

        # load some pickle
        alpha = (2, 3, 3)
        parts = ((1,1), (1,1,1), (2,1))
        cos_reps = coset_reps(perm2.sn(8), young_subgroup_perm(alpha))
        cyc_irrep_func = cyclic_irreps(alpha)

        start = time.time()
        print('Loading {} | {}'.format(alpha, parts))
        yor_dict = load_irrep('/local/hopan/cube/', alpha, parts)
        if yor_dict is None:
            exit()
        print('Done loading | {:.2f}s'.format(time.time() - start))

        wreath1 = wreath_rep(o1, p1, yor_dict, cos_reps, cyc_irrep_func, alpha)
        wreath2 = wreath_rep(o2, p2, yor_dict, cos_reps, cyc_irrep_func, alpha)
        wreath3 = wreath_rep(o3, p3, yor_dict, cos_reps, cyc_irrep_func, alpha)
        w12 = wreath1.dot(wreath2)
        w13 = wreath1.dot(wreath3)
        wd12 = wreath_rep(o12, perm12, yor_dict, cos_reps, cyc_irrep_func, alpha)
        wd13 = wreath_rep(o13, perm13, yor_dict, cos_reps, cyc_irrep_func, alpha)

        self.assertTrue(np.allclose(w12, wd12))
        self.assertTrue(np.allclose(w13, wd13))

if __name__ == '__main__':
    unittest.main()
