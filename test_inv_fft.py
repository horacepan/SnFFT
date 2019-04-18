import sys
import logging
import pdb
import random
import time
import ast
from datetime import datetime
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
FULL_CUBES = 'cube_sym_mod.txt'
DIST = 'dist'

#logger = logging.getLogger(__name__)
#print = logger.info
def get_kth_irreps(k):
    df = pd.read_csv('data/cube2_scaled_norms.csv')
    # sort by the scaled norm, grab the top k alphas + parts
    topk_df = df.nlargest(k, columns=['scaled_norm'])[['alpha', 'parts', 'scaled_norm']]
    alpha = ast.literal_eval(topk_df['alpha'].values[k-1].replace('.', ','))
    parts = ast.literal_eval(topk_df['parts'].values[k-1].replace('.', ','))
    return alpha, parts

def load_sym_cubes(prefixdir):
    all_cubes = pd.read_csv(os.path.join(prefixdir, SYM_CUBES), header=None)
    all_cubes.set_index(0, inplace=True)
    all_cubes.rename(columns={1: DIST}, inplace=True)
    return all_cubes

def load_full_cubes(prefixdir):
    all_cubes = pd.read_csv(os.path.join(prefixdir, FULL_CUBES), header=None)
    all_cubes.set_index(0, inplace=True)
    all_cubes.rename(columns={1: DIST}, inplace=True)
    return all_cubes

def random_sample(n, maxdist, mindist, df):
    # just load the cube_sym_mod
    df = df[df[DIST] <= maxdist]
    df = df[df[DIST] >= mindist]
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
    parser = argparse.ArgumentParser(description='Test drive inverse fft')
    parser.add_argument('--prefix', type=str, default='/local/hopan/cube/')
    parser.add_argument('--fullprefix', type=str, default='/local/hopan/cube/split_unmod/'
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--maxdist', type=int, default=15)
    parser.add_argument('--mindist', type=int, default=0)
    parser.add_argument('--minnorm', type=float, default=0.)
    parser.add_argument('--maxnorm', type=float, default=2.)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--kth', type=int, default=None)
    parser.add_argument('--logfile', type=str, default='ift.log')
    args = parser.parse_args()

    #logging.basicConfig(
    #     filename='{}_{}.log'.format(args.logfile,
    #                                 datetime.now().strftime("%m_%d_%H_%M_%S")),
    #     level=logging.INFO,
    #     format= '[%(asctime)s][%(module)s][%(funcName)s]: %(message)s',
    #     datefmt='%H:%M:%S'
    #)
    print("Starting inverse fft test")

    random.seed(args.seed)
    np.random.seed(args.seed)
    start = time.time()
    peak_mem_usg = 0
    irreps_used = 0
    used_irreps = []
    s8 = sn(8)
    ift_dists = {}
    true_dists = {}
    kth_irrep = get_kth_irreps(args.kth)
    df_cubes = load_sym_cubes(args.prefix)
    cube_strs, _ = random_sample(args.samples, args.maxdist, args.mindist, df_cubes)

    for c in cube_strs:
        for cnbr in neighbors_fixed_core(c):
            wr = get_wreath(cnbr)
            ift_dists[cnbr] = 0
            true_dists[cnbr] = df_cubes.loc[cnbr][DIST]

    for alpha in cube2_alphas():
        cyc_irrep_func = cyclic_irreps(alpha)
        cos_reps = coset_reps(s8, young_subgroup_perm(alpha))
        for parts in partition_parts(alpha):
            if args.kth and (alpha, parts) != kth_irrep:
                continue

            fourier_mat = load_fourier(args.prefix, alpha, parts)
            if fourier_mat is None:
                continue

            mat_sn = np.linalg.norm(fourier_mat) * fourier_mat.shape[0] / (CUBE2_SIZE * 14.)
            if mat_sn < args.minnorm or mat_sn > args.maxnorm:
                continue

            irreps = load_irrep(args.prefix, alpha, parts)
            if irreps is None:
                continue
            ift = cube2_inv_fft_func(irreps, fourier_mat, cos_reps, cyc_irrep_func)

            peak_mem_usg = max(check_memory(False), peak_mem_usg)
            for cube in ift_dists.keys():
                wr = get_wreath(cube)
                ift_dists[cube] += (ift(*wr) / CUBE2_SIZE)

            del irreps
            del fourier_mat
            del ift
            irreps_used += 1
            used_irreps.append((alpha, parts))
            elapsed = (time.time() - start) / 60.
            print('Done with {} | {:30} | Fourier norm: {:.2f}'.format(alpha, str(parts), mat_sn))


    correct_cubes = compute_correct_moves(ift_dists, true_dists, cube_strs)
    print('Used {} irreps:'.format(irreps_used))
    for a, p in  used_irreps:
        print('{}, {}'.format(a, p))
    print('{} / {} correct.'.format(len(correct_cubes), len(cube_strs)))
    print('Peak memory usage: {:.2f}mb'.format(peak_mem_usg))

def pp_diff(cubes, ift_dists, true_dists, result):
    for c in cubes:
        nbrs = neighbors_fixed_core(c)
        nbrs.sort(key=lambda x: ift_dists[x].real)

        for nbr in nbrs:
            wr_nbr = get_wreath(nbr)
            print('{} | pred: {:5.2f} | true: {:5.2f}'.format(nbr, ift_dists[nbr].real, true_dists[nbr]))
        print("Neighbors of {}  | {}".format(c, result))
        print('====================================================')

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
    incorrect_cubes = []
    base_rate = 0
    for cube in cube_states:
        nbrs = neighbors_fixed_core(cube)
        nbrs.sort(key=lambda x: ift_dists[x].real)

        min_dist = min([true_dists[x] for x in nbrs])
        opt_nbrs = [x for x in nbrs if true_dists[x] == min_dist]
        opt_pol_nbr = nbrs[0]
        base_rate += len(opt_nbrs) / len(nbrs)
        if opt_pol_nbr in opt_nbrs:
            n_correct += 1
            correct_cubes.append(cube)
        else:
            incorrect_cubes.append(cube)
    base_rate = base_rate / len(cube_states)
    pp_diff(incorrect_cubes, ift_dists, true_dists, "incorrect")
    print('Base rate for randomg guessing: {}'.format(base_rate))
    return correct_cubes

if __name__ == '__main__':
    main()
