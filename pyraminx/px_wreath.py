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

YOUNG_SUBGROUP_CACHE = {}
COSET_REPS_CACHE = {}

def pyraminx_dists(fname):
    dist_dict = {}

    with open(fname, 'r') as f:
        for line in f.readlines():
            opart, ppart, dist = line.strip().split(',')
            otup = tuple(int(x) for x in opart)
            perm = tuple(int(x) for x in ppart)
            dist = int(dist)
            dist_dict[(otup, perm)] = dist

    return dist_dict

def cyc_irr_func(weak_partition):
    idx0 = weak_partition[0]
    def f(tup):
        p1 = sum(tup[:idx0])
        p2 = sum(tup[idx0:])
        return (1 if p2 % 2 == 0 else -1)
    return f

def block_irreps(tup, coset_reps, cyc_irr_func):
    scalars = np.zeros(len(coset_reps))
    for idx, rep in enumerate(coset_reps):
        tup_g = dot_tup_inv(rep, tup)
        scalars[idx] = cyc_irr_func(tup_g)
    return scalars

def block_mult(yor_dict, block_scalars):
    res = {}
    for k, block in yor_dict.items():
        res[k] = block * block_scalars[k[0]]
    return res

def full_wreath_rep(alpha, parts, dist_dict, par=8):
    s6 = sn(6)
    s_alpha = young_subgroup_perm(alpha)
    young_subgroup_reps = coset_reps(s6, s_alpha)

    cyc_func = cyc_irr_func(alpha)
    yor_dict = wreath_yor_par(alpha, parts, prefix='/local/hopan/irreps/', par=par)

    wreath_dict = {}
    for otup, ptup in dist_dict.keys():
    # given the cyclic func, apply the block scalar multiplications
        block_cyc_rep = block_irreps(otup, young_subgroup_reps, cyc_func)
        yor_block = yor_dict[ptup]
        mat = get_mat(ptup, yor_dict, block_cyc_rep)
        wreath_dict[(otup, ptup)] = mat
    return wreath_dict
 
def save_dict(alpha, parts, wreath_dict, prefix='/local/hopan/pyraminx/irreps'):
    savedir = os.path.join(prefix, str(alpha))
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    save = os.path.join(savedir, str(parts) + '.pkl')
    with open(save, 'wb') as f:
        pickle.dump(wreath_dict, f)

if __name__ == '__main__':
    alpha = eval(sys.argv[1])
    parts = eval(sys.argv[2])
    npar = int(sys.argv[3])
    savedir = sys.argv[4]

    start = time.time()
    dist_dict = pyraminx_dists('dists.txt')
    print('load dict time: {}'.format(time.time() - start))

    start = time.time()
    res = full_wreath_rep(alpha, parts, dist_dict, npar)
    end = time.time()
    print('full wreath rep computation: {}'.format(end - start))
    save_dict(alpha, parts, res, prefix=savedir)
    check_memory()
