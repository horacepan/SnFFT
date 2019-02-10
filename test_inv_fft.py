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

def all_alpha_parts(alphas=None):
    if alphas is None:
        base_reps = cube2_alphas()

    output = []
    for alpha in alphas:
        for parts in partition_parts(alpha):
            output.append((alpha, parts))
    return output

# TODO: need to pass along the dict of values here too!
def handle_irreps(alpha_parts):
    alpha, parts = alpha_parts
    results = {}

    for alpha, parts in alpha_parts:
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
        for k in results.keys():
            _perm, _tup = k
            results[k]['d_ifft'] += (ift(_tup, _perm) / CUBE2_SIZE)

        del irreps
        del fourier_mat
        del ift
        idx += 1
    return results

def do_parallel(args):
    alpha_parts = all_alpha_parts()
    chunked_irreps = chunk(alpha_parts, args.npar)
    if args.npar > 1:
        with Pool(args.npar) as p:
            outputs = p.map(handle_irreps, chunked_irreps)

        results = aggregate_outputs(outputs)

        print('Group element | true dist | computed dist | normed dist | close?')
        for k, v in results.items():
            print('{} | {} | {} | {} | {}'.format(
                k,
                v['d_true'],
                v['d_ifft'],
                np.linalg.norm(v['d_ifft']),
                np.allclose(v['d_true'], v['d_ifft']))
            )

def aggregate_outputs(dicts):
    output = {}
    for d in dicts:
        for k, v in d.items():
            # v is a dict too. want to aggregate the stuff inv across all keys
            d[k] = v
    return output

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
    parser.add_argument('--npar', type=int, default=1)
    parser.add_argument('--bandlimit', action='store_true')
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

    bl_irreps = big_irreps()
    if args.bandlimit:
        print('Using irreps: {}'.format(bl_irreps))

    for alpha in cube2_alphas():
        cyc_irrep_func = cyclic_irreps(alpha)
        cos_reps = coset_reps(s8, young_subgroup_perm(alpha))
        for parts in partition_parts(alpha):
            if args.bandlimit:
                if alpha not in bl_irreps:
                    continue
                elif alpha in bl_irreps and parts not in bl_irreps[alpha]:
                    continue
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
    n_correct = 0
    for k, v in items.items():
        correct = np.allclose(v['d_true'], v['d_ifft'])
        print('{} | {:2} | {:15.2f} | {:5.2f} | {}'.format(
            k,
            v['d_true'],
            v['d_ifft'],
            np.linalg.norm(v['d_ifft']),
            correct
        ))
        if correct:
            n_correct += 1

    end = time.time()
    print('{} / {} correct.'.format(n_correct, len(items)))
    print('Total time: {:.2f}'.format(end - start))
    print('Peak memory usage: {:.2f}mb'.format(peak_mem_usg))
    #print('Total time for s8: {:.2f}'.format(pend - pstart))

def big_irreps():
    irreps = {
        #(2, 3, 3): partition_parts((2, 3, 3)),
        #(4, 2, 2): partition_parts((4, 2, 2)),
        #(3, 1, 4): partition_parts((3, 1, 4)),
        #(3, 4, 1): partition_parts((3, 4, 1)),
        (0, 1, 7): [((), (1,), (7,))],
        (0, 7, 1): [((), (7,), (1,))],
        (8, 0, 0): [((8,), (), ()), ((7, 1), (), ())]
    }

    return irreps

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    tf(main, [])
