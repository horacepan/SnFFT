import os
import time
import pdb
import sys
import pickle

sys.path.append('../')
import numpy as np
from wreath import young_subgroup_yor, wreath_yor_par, block_cyclic_irreps, dot_tup_inv, get_mat
from coset_utils import tup_set, coset_reps, young_subgroup_perm, young_subgroup
from perm2 import sn
from utils import check_memory
from px_wreath import pyraminx_dists

def fourier_transform(irrep_dict, dist_dict):
    mat = None
    for k, irrep_mat in irrep_dict.items():
        otup, ptup = k
        dist = dist_dict[k]

        if mat is None:
            mat = dist * irrep_mat
        else:
            mat += (dist * irrep_mat)

    return mat

if __name__ == '__main__':
    alpha = eval(sys.argv[1])
    parts = eval(sys.argv[2])
    pkl_name = sys.argv[3]
    savedir = sys.argv[4]
    savename = os.path.join(savedir, str(parts))

    print('saving in: {}'.format(savename))
    fname = 'dists.txt'
    dist_dict = pyraminx_dists(fname)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(pkl_name, 'rb') as f:
        start = time.time()
        irrep_dict  = pickle.load(f)
        mat = fourier_transform(irrep_dict, dist_dict)
        end = time.time()
        print('Elapsed: {:.2f}'.format(end - start))
        np.save(savename, mat)

    print('Elapsed: {:.2f}'.format(end - start))
    check_memory()
