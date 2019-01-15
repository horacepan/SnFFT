import pdb
import random
import numpy as np
import perm2
from yor import yor, load_yor
import unittest
from wreath import young_subgroup_yor, wreath_yor, mult, get_mat

class TestWreath(unittest.TestCase):
    def test_wreath(self):
        alpha = (0, 3, 2)
        _parts = ((), (2,1), (1,1))

        ydict, cnts = wreath_yor(alpha, _parts)

        ks = list(ydict.keys())
        vals = list(ydict.values())
        size = len(ydict) * len(vals)

        g = random.choice(ks)
        h = random.choice(ks)
        gh = g * h
        gh_mat = mult(g, h, ydict)
        self.assertTrue(np.allclose(gh_mat, get_mat(gh, ydict)))

        eye = perm2.Perm2.eye(g.size)
        eye_g_mat = mult(eye, g, ydict)
        eye_mat = get_mat(eye * g, ydict)
        self.assertTrue(np.allclose(eye_mat, eye_g_mat))

if __name__ == '__main__':
    unittest.main()
