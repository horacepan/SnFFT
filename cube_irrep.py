import sys
import os
import pdb
import numpy as np
from perm2 import sn
from coset_utils import young_subgroup_perm, coset_reps
from wreath import wreath_rep, get_mat, cyclic_irreps, block_cyclic_irreps
from utils import load_pkl
from yor import yor
import time

sys.path.append('./cube/')
from str_cube import get_wreath, init_2cube, render, scramble

IRREP_LOC_FMT = '/local/hopan/cube/pickles/{}/{}.pkl'
class Cube2Irrep(object):
    def __init__(self, alpha, parts, cached_loc=None):
        '''
        alpha: tuple of ints, the weak partition of 8 into 3 parts
        parts: tuple of ints, the partitions of each part of alpha
        cached_loc: string, location of the cached pickle file of S_8 mod S_alpha irreps
        '''
        self.alpha = alpha
        self.parts = parts
        self.cos_reps = coset_reps(sn(8), young_subgroup_perm(alpha))
        self.cyc_irrep_func = cyclic_irreps(alpha)
        self.yor_dict = None

        if cached_loc:
            # expect the full pickle filename
            self.yor_dict = load_pkl(cached_loc)
        else:
            pkl_loc = IRREP_LOC_FMT.format(alpha, parts)
            print('loading from: {}'.format(pkl_loc))
            self.yor_dict = load_pkl(pkl_loc)

    def str_to_irrep(self, cube_str):
        '''
        cube_str: string representation of 2-cube state
        Returns: numpy matrix
        '''
        otup, gtup = get_wreath(cube_str)
        return self.tup_to_irrep(otup, gtup)

    def tup_to_irrep(self, otup, ptup):
        '''
        otup: tuple of orientations in Z/3Z
        ptup: tuple of permutation of S_8
        Returns: numpy matrix
        '''
        #mult_yor_block(perm_rep, dist, block_cyclic_rep, save_dict)
        block_scalars = block_cyclic_irreps(otup, self.cos_reps, self.cyc_irrep_func)
        rep = get_mat(ptup, self.yor_dict, block_scalars)
        return rep

    def tup_irrepv(self, otup, gtup):
        '''
        otup: tuple of orientations in Z/3Z
        ptup: tuple of permutation of S_8
        Returns: numpy array (unraveled numpy matrix)
        '''
        return self.tup_irrep(otup, gtup).ravel()

def test():
    alpha = (8, 0, 0)
    parts = ((6,2), (), ())
    cube = Cube2Irrep(alpha, parts)

def time_irreps(cube, n):
    c2 = init_2cube()
    cubes = [scramble(c2, 100) for c in range(n)]
    wreath_tups = [get_wreath(c) for c in cubes]

    npstart = time.time()
    for ot, pt in wreath_tups:
        cube.tup_to_irrep(ot, pt)
    npend = time.time()
    print('Elapsed: {:4.3f}'.format(npend - npstart))

if __name__ == '__main__':
    alpha = (2, 3, 3)
    parts = ((2,), (1,1,1), (1,1,1))
    cube = Cube2Irrep(alpha, parts)
    time_irreps(cube, 1)
