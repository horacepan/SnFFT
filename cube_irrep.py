import sys
import os
import pdb
import numpy as np
from perm2 import sn
from coset_utils import young_subgroup_perm, coset_reps
from wreath import wreath_rep, get_mat, cyclic_irreps, block_cyclic_irreps
from utils import load_pkl
from yor import yor

sys.path.append('./cube/')
from str_cube import get_wreath


class Cube2Irrep(object):
    def __init__(self, alpha, parts, cached_loc=None):
        self.alpha = alpha
        self.parts = parts
        self.cos_reps = coset_reps(sn(8), young_subgroup_perm(alpha))
        self.cyc_irrep_func = cyclic_irreps(alpha)
        self.yor_dict = None

        if cached_loc:
            self.yor_dict = load_pkl(cached_loc)
        else:
            pass

    def cube_irrep(self, cube_str):
        otup, gtup = get_wreath(cube_str)
        return self.irrep(otup, gtup)

    def irrep(self, otup, ptup):
        #mult_yor_block(perm_rep, dist, block_cyclic_rep, save_dict)
        block_scalars = block_cyclic_irreps(otup, self.cos_reps, self.cyc_irrep_func)
        rep = get_mat(ptup, self.yor_dict, block_scalars)
        return rep

    def irrepv(self, otup, gtup):
        return self.irrep(otup, gtup).ravel()
