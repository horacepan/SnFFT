import sys
import pdb
import random
from tqdm import tqdm
from itertools import permutations
import time
import pdb
import argparse
import ast
import os
from perm2 import Perm2, sn
from utils import load_pkl, tf, partition_parts, cube2_alphas, check_memory, CUBE2_SIZE
from fft import cube2_inv_fft_part, cube2_inv_fft_func
from wreath import cyclic_irreps, WreathCycSn
from young_tableau import wreath_dim
import numpy as np
from coset_utils import coset_reps, young_subgroup_perm

FOURIER_SUBDIR = 'fourier'
IRREP_SUBDIR = 'pickles'
SPLIT_SUBDIR = 'split_or'

def get_random_elements(n, prefix_dir, max_n=None):
    start = time.time()
    split_dir = os.path.join(prefix_dir, SPLIT_SUBDIR)
    pickles = [p for p in os.listdir(split_dir) if '.pkl' in p]
    output = []

    for pf in pickles:
        pkl_file = os.path.join(split_dir, pf)
        dist_dict = load_pkl(pkl_file)
        chosen_perms = random.sample(list(dist_dict.keys()), n)

        for p in chosen_perms:
            rand_cycle = random.choice(list(dist_dict[p].keys()))
            output.append((p, rand_cycle, dist_dict[p][rand_cycle]))

        if max_n and len(output) > max_n:
            break

    end = time.time()
    print('Sampling time for {} items: {:.2f}s'.format(n * len(pickles), end - start))
    return output

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
    parser.add_argument('--maxsamples', type=int, default=1)
    args = parser.parse_args()

    start = time.time()
    idx = 0
    tot = 0
    peak_mem_usg = 0
    s8 = sn(8)
    random_elements = get_random_elements(args.samples, args.prefix, args.maxsamples)
    items = {(perm, cyc_tup): {'d_true': dlist[0], 'd_ifft': 0} for perm, cyc_tup, dlist in random_elements}

    for k, v in items.items():
        _perm, _tup = k
        print('{} | {}'.format(k, v))

    for alpha in cube2_alphas():
        cyc_irrep_func = cyclic_irreps(alpha)
        cos_reps = coset_reps(s8, young_subgroup_perm(alpha))

        for parts in partition_parts(alpha):
            print('About to load: {} | {}'.format(alpha, parts))
            _s = time.time()
            fourier_mat = load_fourier(args.prefix, alpha, parts)
            irreps = load_irrep(args.prefix, alpha, parts)
            if (fourier_mat is None) or (irreps is None):
                print('Skipping the missing {} | {}'.format(alpha, parts))
                idx += 1
                continue
            print('{:3}/270 | Elapsed: {:.2f}min'.format(idx+1, (time.time() - start) / 60.))
            ift = cube2_inv_fft_func(irreps, fourier_mat, cos_reps, cyc_irrep_func)

            peak_mem_usg = max(check_memory(False), peak_mem_usg)
            for k in items.keys():
                _perm, _tup = k
                items[k]['d_ifft'] += (ift(_tup, _perm) / CUBE2_SIZE)

            del irreps
            del fourier_mat
            del ift
            idx += 1

    print('Group element | true dist | computed dist | normed dist | close?')
    for k, v in items.items():
        print('{} | {} | {} | {} | {}'.format(
            k,
            v['d_true'],
            v['d_ifft'],
            np.linalg.norm(v['d_ifft']),
            np.allclose(v['d_true'], v['d_ifft']))
        )

    end = time.time()
    print('Total time: {:.2f}'.format(end - start))
    print('Peak memory usage: {:.2f}mb'.format(peak_mem_usg))
    #print('Total time for s8: {:.2f}'.format(pend - pstart))

if __name__ == '__main__':
    tf(main, [])
