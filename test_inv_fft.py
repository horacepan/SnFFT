import sys
import pdb
import random
import time
import argparse
import os
import numpy as np
import pandas as pd
from perm2 import sn
from utils import load_pkl, tf, partition_parts, cube2_alphas, check_memory, CUBE2_SIZE
from fft import cube2_inv_fft_func
from wreath import cyclic_irreps
from coset_utils import coset_reps, young_subgroup_perm

sys.path.append('./cube')
from str_cube import neighbors_fixed_core, get_wreath

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

def random_sample(n, maxdist, df):
    # just load the cube_sym_mod
    df = df[df[DIST] <= maxdist]
    sampled = random.sample(range(len(df)), n)
    cubes = df.iloc[sampled].index.values # cube strings are the index
    dists = df.iloc[sampled][DIST].values
    return cubes, dists

def load_fourier(prefix, alpha, parts):
    fourier_npy = os.path.join(prefix, FOURIER_SUBDIR, str(alpha), '{}.npy'.format(parts))
    if os.path.exists(fourier_npy):
        return np.load(fourier_npy)
    return None

def load_irrep(prefix, alpha, parts):
    irrep_path = os.path.join(prefix, IRREP_SUBDIR, str(alpha), '{}.pkl'.format(parts))
    if os.path.exists(irrep_path):
        return load_pkl(irrep_path, 'rb')
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
    parser.add_argument('--bandlimit', type=str, default='')
    parser.add_argument('--maxdist', type=int)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    start = time.time()
    peak_mem_usg = 0
    s8 = sn(8)
    ift_dists = {}
    true_dists = {}
    bl_irreps = None
    df_cubes = load_sym_cubes(args.prefix)
    cube_strs, _ = random_sample(args.samples, args.maxdist, df_cubes)

    for c in cube_strs:
        for cnbr in neighbors_fixed_core(c):
            wr = get_wreath(cnbr)
            ift_dists[wr] = 0
            true_dists[wr] = df_cubes.loc[cnbr][DIST]

    if args.bandlimit:
        bl_func = eval(args.bandlimit)
        bl_irreps = bl_func()
        print('Using irreps: {}'.format(bl_irreps.keys()))

    for alpha in cube2_alphas():
        if args.bandlimit and alpha not in bl_irreps:
            continue

        cyc_irrep_func = cyclic_irreps(alpha)
        cos_reps = coset_reps(s8, young_subgroup_perm(alpha))
        for parts in partition_parts(alpha):
            if args.bandlimit and parts not in bl_irreps[alpha]:
                continue

            fourier_mat = load_fourier(args.prefix, alpha, parts)
            if fourier_mat is None:
                continue
            irreps = load_irrep(args.prefix, alpha, parts)
            if irreps is None:
                continue
            ift = cube2_inv_fft_func(irreps, fourier_mat, cos_reps, cyc_irrep_func)

            peak_mem_usg = max(check_memory(False), peak_mem_usg)
            for wreath in ift_dists.keys():
                ift_dists[wreath] += (ift(*wreath) / CUBE2_SIZE)

            del irreps
            del fourier_mat
            del ift
            elapsed = (time.time() - start) / 60.
            print('Elapsed: {:.2f}min | Done with {} | {}'.format(elapsed, alpha, parts))

    for c in cube_strs:
        print('Neighbors of {}:'.format(c))
        for cnbr in neighbors_fixed_core(c):
            wr = get_wreath(cnbr)
            print('{} | pred: {:.2f} | true: {:.2f}'.format(cnbr, ift_dists[wr].real, true_dists[wr]))
        print('=============================================')

    correct_cubes = compute_correct_moves(ift_dists, true_dists, cube_strs)
    print('{} / {} correct.'.format(len(correct_cubes), len(cube_strs)))
    print('Peak memory usage: {:.2f}mb'.format(peak_mem_usg))

def compute_correct_moves(ift_dists, true_dists, cube_states):
    '''
    ift_dists: dictionary of perm + cyclic tuple --> dist
    true_dists: dictionary of perm + cyclic tuple --> dist
    cube_states: list of cube states. Keys of ift_dists + true_dists contain neighbors of
                 these cube states
    Returns: number correct, incorrect
    '''
    n_correct = 0
    correct_cubes = []

    for cube in cube_states:
        opt_nbrs = []
        opt_dist = float('inf')
        opt_pol_nbr = None
        opt_pol_dst = float('inf')

        # loop over neighbors, find best neighbor
        for nbr in neighbors_fixed_core(cube):
            # get the opt nbr
            wr_nbr = get_wreath(nbr)
            if true_dists[wr_nbr] < opt_dist:
                opt_dist = true_dists[wr_nbr]
                opt_nbrs.append(wr_nbr)

            if ift_dists[wr_nbr].real < opt_pol_dst:
                opt_pol_dst = ift_dists[wr_nbr].real
                opt_pol_nbr = wr_nbr

        if opt_pol_nbr in opt_nbrs:
            n_correct += 1
            correct_cubes.append(opt_pol_nbr)

    return correct_cubes

def top():
    irreps = {
        (2, 3, 3): partition_parts((2, 3, 3)),
        (8, 0, 0): [((8,), (), ())]
    }
    return irreps

def top2():
    irreps = {
        (2, 3, 3): partition_parts((2, 3, 3)),
        (4, 2, 2): partition_parts((4, 2, 2)),
        (8, 0, 0): [((8,), (), ())]
    }
    return irreps

def top4():
    irreps = {
        (2, 3, 3): partition_parts((2, 3, 3)),
        (4, 2, 2): partition_parts((4, 2, 2)),
        (3, 1, 4): partition_parts((3, 1, 4)),
        (3, 4, 1): partition_parts((3, 4, 1)),
        (8, 0, 0): [((8,), (), ())]
    }
    return irreps

def t341():
    irreps = {
        (3, 4, 1): partition_parts((3, 4, 1)),
        (8, 0, 0): [((8,), (), ())]
    }
    return irreps

def t314():
    irreps = {
        (3, 1, 4): partition_parts((3, 1, 4)),
        (8, 0, 0): [((8,), (), ())]
    }
    return irreps

def mode():
    irreps = {
        (8, 0, 0): [((8,), (), ())]
    }
    return irreps


if __name__ == '__main__':
    tf(main)
    #tf(test_dists)
