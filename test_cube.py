import pdb
import unittest
import sys
sys.path.append('./cube/')
from collections import Counter
from cube_perms import rot_permutations
from wreath import WreathCycSn
from str_cube import *

def group_wreath(cube_str):
    otup, ptup = get_wreath(cube_str)
    return WreathCycSn.from_tup(otup, ptup, 3)

class TestCube(unittest.TestCase):
    def test_cube_sym_is_group(self):
        '''
        Check that the 24 rigid permutations of the cube form a group
        '''
        c = init_2cube()
        rot_perm_tups = set([get_wreath(c) for c in rot_permutations(c)])
        rot_perm_wreaths = [WreathCycSn.from_tup(*gt, 3) for gt in rot_perm_tups]

        # loop over every single pair:
        # check inverses
        for g in rot_perm_wreaths:
            self.assertTrue(g.inv().tup_rep in rot_perm_tups)

        for g in rot_perm_wreaths:
            for h in rot_perm_wreaths:
                self.assertTrue((g*h).tup_rep in rot_perm_tups)

    def test_cube_sym_moves(self):
        '''
        Check that the group elements for the rigid symmetry moves are the same
        when applied to the identity versus any random state
        '''
        exp_wreath = [get_wreath(c) for c in rot_permutations(init_2cube())]
        exp_set = set(exp_wreath)

        cube = scramble_fixedcore(init_2cube(), 200)
        g = WreathCycSn.from_tup(*get_wreath(cube), 3)
        rot_wreaths = [WreathCycSn.from_tup(*get_wreath(c), 3) for c in rot_permutations(cube)]
        group_elems = [w * g.inv() for w in rot_wreaths]
        g_set = set([g.tup_rep for g in group_elems])

        self.assertEqual(g_set, exp_set)

    def test_str_wreath_reps(self):
        '''
        Check that the string representation of the cube is identical to its wreath
        group representation.
        '''
        ts = time.time()
        c = init_2cube()
        move_dict = { f: group_wreath(rotate(c, f)) for f in FIXEDCORE_FACES }

        cubes = [scramble_fixedcore(c, 100) for _ in range(1000)]
        group_els = [group_wreath(s) for s in cubes]

        moves = [random.choice(FIXEDCORE_FACES) for _ in range(len(cubes))]
        move_els = [move_dict[f] for f in moves]

        # for each cube, rotate it by
        start = time.time()
        new_str_cubes = [rotate(s, f) for s, f in zip(cubes, moves)]
        end = time.time()
        print('String time: {:.4f}s'.format(end - start))

        start = time.time()
        new_group_els = [h * g for g, h in zip(group_els, move_els)]
        end = time.time()
        print('Group mult time: {:.4f}s'.format(end - start))

        for g, h in zip(new_str_cubes, new_group_els):
            self.assertEqual(get_wreath(g), h.tup_rep)

        print('String and group representations are equal!')
        te = time.time()
        print('Total time: {:.2f}s'.format(te - ts))

if __name__ == '__main__':
    unittest.main()

