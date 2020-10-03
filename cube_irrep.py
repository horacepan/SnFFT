import sys
import os
import pdb
import numpy as np
from perm2 import sn
from coset_utils import young_subgroup_perm, coset_reps
from wreath import wreath_rep, get_mat, cyclic_irreps, block_cyclic_irreps, get_sparse_mat, WreathCycSn
from utils import load_pkl, load_sparse_pkl, check_memory
from young_tableau import wreath_dim
import time
import torch
sys.path.append('./cube/')
from str_cube import get_wreath, init_2cube, render, scramble
from cube_utils import cube2_orientations

if os.path.exists('/local/hopan/cube'):
    IRREP_LOC_FMT = '/local/hopan/cube/pickles/{}/{}.pkl'
    IRREP_SP_LOC_FMT = '/local/hopan/cube/pickles_sparse/{}/{}.pkl'
elif os.path.exists('/scratch/hopan/cube'):
    IRREP_LOC_FMT = '/scratch/hopan/cube/pickles/{}/{}.pkl'
    IRREP_SP_LOC_FMT = '/scratch/hopan/cube/pickles_sparse/{}/{}.pkl'
else:
    IRREP_LOC_FMT = '/project2/risi/cube/pickles/{}/{}.pkl'
    IRREP_SP_LOC_FMT = '/project2/risi/cube/pickles_sparse/{}/{}.pkl'

# This should maybe be split into two classes. One for numpy reps, one for torch sparse reps
class Cube2Irrep(object):
    def __init__(self, alpha, parts, numpy=False, sparse=True):
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

        # cache orientation tuple -> cyclic irrep
        self.cyclic_irreps_re = {}
        self.cyclic_irreps_im = {}
        st = time.time()
        self.fill_cyclic_irreps()
        mem = check_memory(verbose=False)
        # print(f'Done caching cyclic irreps: {time.time() - st:.2f}s | Mem used: {mem:.2f}mb')

        # also cache the cyclic irreps
        if numpy:
            pkl_loc = IRREP_LOC_FMT.format(alpha, parts)
            self.yor_dict = load_pkl(pkl_loc)
        elif sparse:
            pkl_loc = IRREP_SP_LOC_FMT.format(alpha, parts)
            self.yor_dict = load_sparse_pkl(pkl_loc)
        else:
            # TODO: deprecate
            print('neither sparse nor numpy')
            pkl_loc = IRREP_LOC_FMT.format(alpha, parts)
            self.np_yor_dict = load_pkl(pkl_loc)

    def block_pad(self, arr):
        '''
        Given an array of length(self.cos_reps)
        Block extend this to length full group / len(self.cos_reps)
        '''
        block_size = wreath_dim(self.parts) ** 2
        output = np.zeros(len(arr) * block_size)
        for i in range(arr.size):
            output[i * block_size: (i+1) * block_size] =  arr[i]
        return output

    def fill_cyclic_irreps(self):
        '''
        Stores the cyclic irrep for every single 2-cube orientation
        so fetching a 2-cube state representation becomes
        two dictionary lookups and a sparse elementwise multiplication.
        '''
        for tup in cube2_orientations():
            cyc_irrep = block_cyclic_irreps(tup, self.cos_reps, self.cyc_irrep_func)
            self.cyclic_irreps_re[tup] = torch.FloatTensor(self.block_pad(cyc_irrep.real))
            self.cyclic_irreps_im[tup] = torch.FloatTensor(self.block_pad(cyc_irrep.imag))

    def str_to_irrep_np(self, cube_str):
        '''
        cube_str: string representation of 2-cube state
        Returns: numpy matrix
        '''
        otup, gtup = get_wreath(cube_str)
        return self.tup_to_irrep_np(otup, gtup)

    def str_to_irrep_th(self, cube_str):
        otup, gtup = get_wreath(cube_str)
        return self.tup_to_irrep_th(otup, gtup)

    def tup_to_irrep_inv_np(self, otup, ptup):
        wreath_el = WreathCycSn.from_tup(otup, ptup, 3)
        oinv, ginv = wreath_el.inv_tup_rep()
        return self.tup_to_irrep_np(oinv, ginv)

    def tup_to_irrep_np(self, otup, ptup):
        '''
        otup: tuple of orientations in Z/3Z
        ptup: tuple of permutation of S_8
        Returns: numpy matrix
        '''
        block_scalars = block_cyclic_irreps(otup, self.cos_reps, self.cyc_irrep_func)
        rep = get_mat(ptup, self.yor_dict, block_scalars)
        return rep

    def tup_to_irrep_th(self, otup, ptup):
        '''
        otup: int tuple of the orientations
        ptup: permutation tuple
        First cmpus n
        '''
        block_scalars = block_cyclic_irreps(otup, self.cos_reps, self.cyc_irrep_func)
        rep_re, rep_im = get_sparse_mat(ptup, self.np_yor_dict, block_scalars)
        return rep_re, rep_im

    def tup_to_irrep_sp(self, otup, ptup):
        re = self.yor_dict[ptup]
        rem = re.mul(torch.sparse.FloatTensor(re.indices(), self.cyclic_irreps_re[otup], re.size()))
        imm = re.mul(torch.sparse.FloatTensor(re.indices(), self.cyclic_irreps_im[otup], re.size()))
        return rem, imm

    def tup_to_irrep_sp_noimag(self, otup, ptup):
        rem = self.yor_dict[ptup]
        imm = rem * 0
        return rem, imm

    def str_to_irrep_sp(self, cube_str):
        '''
        cube_str: string of cube
        This is only useable if the irrep was loaded with the sparse pickle dict. This means
        that the keys to this dict are tuples and the values are sparse matrices
        '''
        otup, gtup = get_wreath(cube_str)
        re = self.yor_dict[gtup]
        rem = re.mul(torch.sparse.FloatTensor(re.indices(), self.cyclic_irreps_re[otup], re.size()))
        imm = re.mul(torch.sparse.FloatTensor(re.indices(), self.cyclic_irreps_im[otup], re.size()))
        return rem, imm

    def str_to_irrep_sp_inv(self, cube_str):
        otup, gtup = get_wreath(cube_str)
        wreath_el = WreathCycSn.from_tup(otup, gtup, 3)
        oinv, ginv = wreath_el.inv_tup_rep()
        re = self.yor_dict[ginv]
        rem = re.mul(torch.sparse.FloatTensor(re.indices(), self.cyclic_irreps_re[oinv], re.size()))
        imm = re.mul(torch.sparse.FloatTensor(re.indices(), self.cyclic_irreps_im[oinv], re.size()))
        return rem, imm

    def tup_to_irrep_inv(self, otup, ptup):
        wreath_el = WreathCycSn.from_tup(otup, ptup, 3)
        # want ginv tuple, want to dot ginv with oinv
        oinv, ginv = wreath_el.inv_tup_rep()
        re = self.yor_dict[ginv]
        rem = re.mul(torch.sparse.FloatTensor(re.indices(), self.cyclic_irreps_re[oinv], re.size()))
        imm = re.mul(torch.sparse.FloatTensor(re.indices(), self.cyclic_irreps_im[oinv], re.size()))
        return rem, imm


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
        cube.tup_to_irrep_np(ot, pt)
    npend = time.time()
    print('Elapsed: {:4.3f}'.format(npend - npstart))

if __name__ == '__main__':
    alpha = (2, 3, 3)
    parts = ((2,), (1,1,1), (1,1,1))
    cube = Cube2Irrep(alpha, parts)
    time_irreps(cube, 1)
