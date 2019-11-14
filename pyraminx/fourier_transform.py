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
from px_utils import pyraminx_dists, alpha_parts, dist_df, load_rep_mat
from multiprocessing import Pool

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

def subsample_ft(alpha_parts, nsample):
    '''
    Strategy: load the rep matrix which is a np matrix of shape (nstates, dim, dim)
    load the df as well
    subsample indices
    get the distances and do a block matrix multiply 
    '''
    alpha, parts = alpha_parts
    DIST_COL = 2
    df = dist_df('/local/hopan/pyraminx/dists.txt')
    rep_mat = load_rep_mat(alpha, parts)

    if nsample == 11520:
        sampled_idx = np.arange(11520)
    else:
        sampled_idx = np.random.choice(len(df), size=nsample, replace=False)
    sampled_dists = df.loc[sampled_idx][DIST_COL].values.reshape(nsample, 1, 1)
    sampled_reps = rep_mat[sampled_idx, :, :]
    ft = np.sum(sampled_dists * sampled_reps, axis=0)

    savepref = '/local/hopan/pyraminx/fourier_sample/{}/'.format(nsample)
    savedir = os.path.join(savepref, str(alpha))
    savename = os.path.join(savedir, str(parts))
    if not os.path.exists(savedir):
        try:
            os.makedirs(savedir)
        except:
            pass

    np.save(savename, ft)
    return ft

def par_sample(ncpu, sample):
    aps = alpha_parts()
    args = [(ap, sample) for ap in aps]
    chunk_size = len(aps) // ncpu

    with Pool(ncpu) as p:
        p.starmap(subsample_ft, args, chunk_size)

def par_sample_main():
    ncpu = 16
    samples = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 11520]
    for s in samples:
        start = time.time()
        par_sample(ncpu, s)
        elapsed = time.time() - start
        print('Done with {} samples | {:.2f}s'.format(s, elapsed))

if __name__ == '__main__':
    par_sample_main()
    '''
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
    '''
