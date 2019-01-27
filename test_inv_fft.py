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
from perm2 import Perm2
from utils import load_pkl, tf, partition_parts, cube2_alphas, check_memory
from fft import cube2_inv_fft_part, cube2_inv_fft_func
from wreath import cyclic_irreps
from young_tableau import wreath_dim
import numpy as np

FOURIER_SUBDIR = 'fourier'
IRREP_SUBDIR = 'pickles'
SPLIT_SUBDIR = 'split'

def get_random_elements(n, split_prefix_dir):
    split_dir = os.path.join(split_prefix_dir, SPLIT_SUBDIR)
    random_split_file = random.choice([p for p in os.listdir(split_dir) if '.pkl' in p])

    dist_dict = load_pkl(os.path.join(split_dir, random_split_file))

    # different keys for each?
    chosen_perms = random.sample(list(dist_dict.keys()), n)
    output = []

    for p in chosen_perms:
        rand_cycle = random.choice(list(dist_dict[p].keys()))
        output.append((p, rand_cycle, dist_dict[p][rand_cycle]))

    return output

def load_fourier(prefix, alpha, parts):
    fourier_npy = os.path.join(prefix, FOURIER_SUBDIR, str(alpha), '{}.npy'.format(parts))
    return np.load(fourier_npy)

def load_irrep(prefix, alpha, parts):
    irrep_path = os.path.join(prefix, IRREP_SUBDIR, str(alpha), '{}.pkl'.format(parts))
    return load_pkl(irrep_path, 'rb')

def main():
    '''
    Loads the fourier and irrep dict and then computes the inverse fourier transform
    portion for this irrep.
    '''
    parser = argparse.ArgumentParser(description='Test drive inverse fft')
    parser.add_argument('--prefix', type=str, default='/local/hopan/cube/')
    args = parser.parse_args()

    start = time.time()
    # evaluate inverse fourier transform for a single group element
    # get a random group element
    g = Perm2.from_tup((1, 2, 3, 5, 6, 7, 8))
    g = Perm2.eye(8)
    tot = 0
    i = 0
    for alpha in cube2_alphas():
        cyc_irrep_func = cyclic_irreps(alpha)
        for parts in partition_parts(alpha):
            print('About to load: {} | {}'.format(alpha, parts))
            _s = time.time()
            fourier_mat = load_fourier(args.prefix, alpha, parts)
            irreps = load_irrep(args.prefix, alpha, parts)
            print('Done loading: {} | {}'.format(alpha, parts))
            print('Load time: {:.2f}s'.format(time.time() - _s))
            ift = cube2_inv_fft_func(irreps, fourier_mat, alpha, parts)

            check_memory()
            tot += ift(g) * cyc_irrep_func()
            del irreps
            del fourier_mat
            del ift

    end = time.time()
    print('Load time: {:.2f}'.format(end - start))
    print('Memory usage: | ', end='')
    check_memory()
    #print('Total time for s8: {:.2f}'.format(pend - pstart))

if __name__ == '__main__':
    #print(get_random_elements(200, '/local/hopan/cube/'))
    tf(main, [])
