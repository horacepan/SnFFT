import pdb
import unittest
import perm2
import coset_utils as cu
import wreath

class TestCoset(unittest.TestCase):
    def check_coset(self, G, H):
        reps = cu.coset_reps(G, H)
        self.assertEqual(len(reps), len(G)/len(H))

        for p in reps:
            pH = cu.left_coset(p, H)
            set_pH = set(pH)
            # check that pH is the same as gH for all g in pH
            for g in pH:
                gH = cu.left_coset(g, H)
                set_gH = set(gH)
                self.assertTrue(set_pH == set_gH)

    #def test_coset(self):
    #    H = perm2.sn(4)
    #    G = perm2.sn(6)
    #    self.check_coset(G, H)

    def test_young_coset(self):
        alpha = (2, 2, 2, 1)
        G = perm2.sn(7)
        H = wreath.young_subgroup_perm(alpha)
        self.check_coset(G, H)

if __name__ == '__main__':
    unittest.main()
