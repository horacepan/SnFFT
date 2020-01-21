import unittest
import random
import numpy as np
from perm2 import Perm2, sn
from yor import yor
from young_tableau import FerrersDiagram
from cg_utils import intertwine, block_rep

class TestPerm(unittest.TestCase):
    def test_s5(self):
        s5 = sn(5)
        gtup = Perm2.from_tup((2, 1, 3, 4, 5))
        gcyc = Perm2.from_tup((2, 3, 4, 5, 1))
        f1 = FerrersDiagram((5, 1))
        f2 = FerrersDiagram((3, 2, 1))
        mult = 2

        m1 = yor(f1, gtup)
        m2 = yor(f1, gcyc)
        n1 = yor(f2, gtup)
        n2 = yor(f2, gcyc)

        kron1 = np.kron(m1, n1)
        kron2 = np.kron(m2, n2)
        intw = intertwine(kron1, kron2, n1, n2, mult, verbose=True)


        def test_g(g, intw):
            mg = yor(f1, g)
            ng = yor(f2, g)
            kg = np.kron(mg, ng)
            blocked = block_rep(ng, mult)
            return np.allclose(intw @ kg, block_rep(ng, mult) @ intw)

        self.assertTrue(test_g(gtup, intw))
        self.assertTrue(test_g(gcyc, intw))
        self.assertTrue(test_g(random.choice(s5), intw))
        self.assertTrue(test_g(random.choice(s5), intw))

if __name__ == '__main__':
    unittest.main()
