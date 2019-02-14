import sys
import pdb
import random
from tqdm import tqdm
from itertools import permutations
from multiprocessing import Pool
import time
import argparse
import ast
import os
from perm2 import Perm2, sn
from utils import load_pkl, tf, partition_parts, cube2_alphas, check_memory, chunk, CUBE2_SIZE
from fft import cube2_inv_fft_part, cube2_inv_fft_func
from wreath import cyclic_irreps, WreathCycSn
from young_tableau import wreath_dim
import numpy as np
from coset_utils import coset_reps, young_subgroup_perm
import pandas as pd
sys.path.append('./cube')
from str_cube import neighbors, neighbors_fixed_core, get_wreath
from cube_perms import rot_permutations

FOURIER_SUBDIR = 'fourier_unmod'
IRREP_SUBDIR = 'pickles'
SPLIT_SUBDIR = 'split_unmod'
SYM_CUBES = 'cube_sym_mod.txt'
DIST = 'dist'

def load_sym_cubes(prefixdir):
    all_cubes = pd.read_csv(os.path.join(prefixdir, SYM_CUBES), header=None)
    all_cubes.set_index(0, inplace=True)
    all_cubes.rename(columns={1: DIST}, inplace=True)
    return all_cubes

def random_sample(n, df):
    # just load the cube_sym_mod
    sampled = random.sample(range(len(df)), n)
    cubes = df.iloc[sampled].index.values # cube strings are the index
    dists = df.iloc[sampled][DIST].values
    return cubes, dists

def load_fourier(prefix, alpha, parts):
    fourier_npy = os.path.join(prefix, FOURIER_SUBDIR, str(alpha), '{}.npy'.format(parts))
    if os.path.exists(fourier_npy):
        return np.load(fourier_npy)
    else:
        return None

def load_irrep(prefix, alpha, parts):
    irrep_path = os.path.join(prefix, IRREP_SUBDIR, str(alpha), '{}.pkl'.format(parts))
    if os.path.exists(irrep_path):
        return load_pkl(irrep_path, 'rb')
    else:
        return None

def main():
    '''
    Loads the fourier and irrep dict and then computes the inverse fourier transform
    portion for this irrep.
    '''
    print("Starting inverse fft test")
    parser = argparse.ArgumentParser(description='Test drive inverse fft')
    parser.add_argument('--prefix', type=str, default='/local/hopan/cube/')
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--npar', type=int, default=1)
    parser.add_argument('--bandlimit', action='store_true')
    parser.add_argument('--smallbandlimit', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    start = time.time()
    n_correct = 0
    idx = 0
    tot = 0
    peak_mem_usg = 0
    s8 = sn(8)
    ift_dists = {}
    true_dists = {}
    all_cubes = load_sym_cubes(args.prefix)
    cube_strs, dists = random_sample(args.samples, all_cubes)

    for c, true_dist in zip(cube_strs, dists):
        for cnbr in neighbors_fixed_core(c):
            cyc_tup, perm = get_wreath(cnbr)
            ift_dists[(perm, cyc_tup)] = 0
            true_dists[(perm, cyc_tup)] = all_cubes.loc[cnbr][DIST]

    if args.bandlimit:
        bl_irreps = big_irreps()
        print('Using irreps: {}'.format(bl_irreps.keys()))
    elif args.smallbandlimit:
        bl_irreps = med_irreps()
        print('Using irreps: {}'.format(bl_irreps.keys()))

    for alpha in cube2_alphas():
        cyc_irrep_func = cyclic_irreps(alpha)
        cos_reps = coset_reps(s8, young_subgroup_perm(alpha))
        for parts in partition_parts(alpha):
            if args.bandlimit or args.smallbandlimit:
                if alpha not in bl_irreps:
                    continue
                elif alpha in bl_irreps and parts not in bl_irreps[alpha]:
                    continue

            print('About to load: {} | {}'.format(alpha, parts))
            _s = time.time()
            fourier_mat = load_fourier(args.prefix, alpha, parts)
            if fourier_mat is None:
                print('Fourier missing. Skipping {} | {}'.format(alpha, parts))
                continue
            irreps = load_irrep(args.prefix, alpha, parts)
            if irreps is None:
                print('Irreps missing. Skipping {} | {}'.format(alpha, parts))
                continue
            idx += 1
            print('{:3}/270 | Elapsed: {:.2f}min'.format(idx+1, (time.time() - start) / 60.))

            ift = cube2_inv_fft_func(irreps, fourier_mat, cos_reps, cyc_irrep_func)

            peak_mem_usg = max(check_memory(False), peak_mem_usg)
            for k in ift_dists.keys():
                _perm, _tup = k
                ift_dists[k] += (ift(_tup, _perm) / CUBE2_SIZE)

            del irreps
            del fourier_mat
            del ift
            idx += 1

    for cube, d_pred in ift_dists.items():
        d_true = true_dists[cube]
        print('{} | pred dist: {:.2f} | true dist: {:.2f}'.format(cube, d_pred, d_true))
        if np.allclose(d_true, d_pred):
            n_correct += 1

    end = time.time()
    print('{} / {} correct.'.format(n_correct, len(ift_dists)))
    print('Total time: {:.2f}'.format(end - start))
    print('Peak memory usage: {:.2f}mb'.format(peak_mem_usg))

def med_irreps():
    irreps = {
        (8, 0, 0): [((8,), (), ())]
    }
    return irreps

def big_irreps():
    irreps = {
        (2, 3, 3): partition_parts((2, 3, 3)),
        (4, 2, 2): partition_parts((4, 2, 2)),
        (3, 1, 4): partition_parts((3, 1, 4)),
        (3, 4, 1): partition_parts((3, 4, 1)),
        (8, 0, 0): [((8,), (), ())]
    }

    return irreps

if __name__ == '__main__':
    tf(main)
    #tf(test_dists) 
