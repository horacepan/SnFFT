import time
import pdb
import sys
import os
import numpy as np
from scipy.sparse import csr_matrix
from perm2 import sn
from coset_utils import young_subgroup_perm, coset_reps
from wreath import wreath_rep, cyclic_irreps, block_cyclic_irreps, wreath_rep_sp
from utils import load_pkl, cube2_orientations

sys.path.append('./cube')
from str_cube import get_wreath, init_2cube, scramble_fixedcore

class Cube2SpIrrep(object):
    def __init__(self, alpha, parts, pickledir=None):
        self.alpha = alpha
        self.parts = parts

        self.cos_reps = coset_reps(sn(8), young_subgroup_perm(alpha)) # num blocks = num cosets
        self.cyc_irrep_func = cyclic_irreps(alpha)

        # cache cyclic irreps
        self.cyc_irreps = self.compute_cyclic_irreps()

        # load induced perm irrep dict
        try: 
            fname = os.path.join(pickledir, str(alpha), str(parts) + '.pkl')
            print('Loading pkl from: {}'.format(fname))
            self.ind_irrep_dict = load_pkl(fname)
            self.block_size = self.ind_irrep_dict[(1,2,3,4,5,6,7,8)].shape[0] // len(self.cos_reps)
            end = time.time()
        except:
            raise Exception(f'Cant load cube irrep {alpha}, {parts}')

    def compute_cyclic_irreps(self):
        cyc_irreps= {}
        for otup in cube2_orientations():
            o_irrep = block_cyclic_irreps(otup, self.cos_reps, self.cyc_irrep_func)
            cyc_irreps[otup] = o_irrep
        
        return cyc_irreps

    def str_irrep(self, cube_str):
        otup, ptup = get_wreath(cube_str)
        return self.irrep(otup, ptup)

    def irrep(self, otup, ptup):
        cyc_irrep = self.cyc_irreps[otup] 
        perm_irrep = self.ind_irrep_dict[ptup]

        # block multiply
        nblocks = len(self.cos_reps)
        sp_mat = self.ind_irrep_dict[ptup]

        new_data = np.zeros(sp_mat.data.shape, dtype=np.complex128)
        for idx, c in enumerate(cyc_irrep):
            try:
                st = sp_mat.indptr[idx * self.block_size]
                end = sp_mat.indptr[idx * self.block_size + self.block_size]
            except:
                pdb.set_trace()
            new_data[st: end] = c * sp_mat.data[st: end]

        mat = csr_matrix((new_data, sp_mat.indices, sp_mat.indptr), shape=sp_mat.shape)
        return mat

    # test
    def _str_irrep_test(self, cube_str):
        otup, ptup = get_wreath(cube_str)
        return self._irrep_test(otup, ptup)

    def _irrep_test(self, otup, ptup):
        return wreath_rep_sp(otup, ptup, self.ind_irrep_dict, self.cos_reps,
                             self.cyc_irrep_func, self.cyc_irreps)

if __name__ == '__main__':
    st = time.time()
    alpha = (0, 4, 4)
    parts = ((), (2, 2), (3, 1))
    pickledir = '/scratch/hopan/cube/pickles_sparse'
    irrep = Cube2SpIrrep(alpha, parts, pickledir)
    end = time.time()
    print('Load time: {:.2f}s'.format(end - st))
    c = init_2cube()
    w1 = irrep.str_irrep(c)
