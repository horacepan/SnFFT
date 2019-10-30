import sys
from tqdm import tqdm
import ast
import math
import argparse
import os
import time
import pdb
import numpy as np
from multiprocessing import Pool, Manager
from utils import tf, chunk, check_memory, load_pkl
from wreath import cyclic_irreps, block_cyclic_irreps, wreath_rep
from young_tableau import wreath_dim
from perm2 import sn
from coset_utils import young_subgroup_perm, coset_reps

TWO_CUBE_SIZE = 88179840
def clean_line(line):
    ostr, pstr, dist = line.strip().split(',')
    otup = tuple(map(int, ostr))
    ptup = tuple(map(int, pstr))
    dist = int(dist)
    return otup, ptup, dist

def mult_yor_block(irrep, scalar, block_scalars, save_dict):
    for k in irrep.keys():
        i, j = k
        if k not in save_dict:
            save_dict[k] = scalar * block_scalars[i] * irrep[k]
        else:
            save_dict[k] += scalar * block_scalars[i] * irrep[k]

def convert_yor_matrix(irrep_dict, block_size, cosets):
    '''
    Alternative:
        compute block size via young_tableau(parts)
        compute cosets via (8!) / (alpha!)
    '''
    shape = (block_size * cosets, block_size * cosets)
    full_mat = np.zeros(shape, dtype=np.complex128)

    for tup_idx, mat in irrep_dict.items():
        i, j = tup_idx
        full_mat[block_size * i: block_size * (i+1), block_size * j : block_size * (j + 1)] = mat

    return full_mat

def coset_size(alpha):
    alpha_size = math.factorial(alpha[0]) * math.factorial(alpha[1]) * math.factorial(alpha[2])
    return int(math.factorial(8) // alpha_size)

def split_transform(fsplit_lst, irrep_dict, alpha, parts, mem_dict=None):
    '''
    fsplit_pkl: list of pkl file names of the distance values for a chunk of the total distance values
    irrep_dict: irrep dict 
    alpha: weak partition
    parts: list/iterable of partitions of the parts of alpha
    '''
    print('     Computing transform on splits: {}'.format(fsplit_lst))
    cos_reps = coset_reps(sn(8), young_subgroup_perm(alpha))
    save_dict = {}
    cyc_irrep_func = cyclic_irreps(alpha)
    pid = os.getpid()

    for fsplit_pkl in fsplit_lst:
        with open(fsplit_pkl, 'r') as f:
            # dict of function values
            pkl_dict = load_pkl(fsplit_pkl)
            for perm_tup, tup_dict in pkl_dict.items():
                for tup, dists in tup_dict.items():
                    dist_tot = sum(dists)
                    perm_rep = irrep_dict[perm_tup]  # perm_rep is a dict of (i, j) -> matrix
                    block_cyclic_rep = block_cyclic_irreps(tup, cos_reps, cyc_irrep_func)
                    mult_yor_block(perm_rep, dist_tot, block_cyclic_rep, save_dict)
            if mem_dict is not None:
                mem_dict[pid] = max(check_memory(verbose=False), mem_dict.get(pid, 0))
            del pkl_dict

    block_size = wreath_dim(parts)
    n_cosets = coset_size(alpha)
    mat = convert_yor_matrix(save_dict, block_size, n_cosets)
    return mat

def text_split_transform(fsplit_lst, irrep_dict, alpha, parts, mem_dict=None):
    '''
    fsplit_pkl: list of split file names of the distance values for a chunk of the total distance values
    irrep_dict: irrep dict
    alpha: weak partition
    parts: list/iterable of partitions of the parts of alpha
    '''
    print('     Computing transform on splits: {}'.format(fsplit_lst))
    cos_reps = coset_reps(sn(8), young_subgroup_perm(alpha))
    save_dict = {}
    cyc_irrep_func = cyclic_irreps(alpha)
    pid = os.getpid()

    for split_f in fsplit_lst:
        with open(split_f, 'r') as f:
            for line in tqdm(f):
                otup, perm_tup, dist  = clean_line(line)
                perm_rep = irrep_dict[perm_tup]  # perm_rep is a dict of (i, j) -> matrix
                block_cyclic_rep = block_cyclic_irreps(otup, cos_reps, cyc_irrep_func)
                mult_yor_block(perm_rep, dist, block_cyclic_rep, save_dict)

        if mem_dict is not None:
            mem_dict[pid] = max(check_memory(verbose=False), mem_dict.get(pid, 0))

    block_size = wreath_dim(parts)
    n_cosets = coset_size(alpha)
    mat = convert_yor_matrix(save_dict, block_size, n_cosets)
    return mat


def full_transform(args, alpha, parts, split_chunks):
    print('Computing full transform for alpha: {} | parts: {}'.format(alpha, parts))
    savedir_alpha = os.path.join(args.savedir, args.alpha)
    savename = os.path.join(savedir_alpha, '{}'.format(parts))
    print('Savename: {}'.format(savename))
    if os.path.exists(savename + '.npy'):
        print('Skipping. Already computed fourier matrix for: {} | {}'.format(alpha, parts))
        exit()

    manager = Manager()
    irrep_dict = load_pkl(os.path.join(args.pkldir, args.alpha, '{}.pkl'.format(parts)))
    mem_usg = check_memory(verbose=False)

    if not os.path.exists(savedir_alpha):
        print('Making: {}'.format(savedir_alpha))
        os.makedirs(savedir_alpha)

    if args.par > 1:
        print('Par process with {} processes...'.format(len(split_chunks)))
        mem_dict = manager.dict()
        with Pool(len(split_chunks)) as p:
            arg_tups = [(_fn, irrep_dict, alpha, parts, mem_dict) for _fn in split_chunks]
            matrices = p.starmap(text_split_transform, arg_tups)
            np.save(savename, sum(matrices))
    else:
        print('Single thread...')
        matrices = []
        block_size = wreath_dim(parts)
        n_cosets = coset_size(alpha)
        shape = (block_size * n_cosets, block_size * n_cosets)
        result = np.zeros(shape, dtype=np.complex128)
        mem_dict = {}
        for _fn in split_chunks:
            res = text_split_transform(_fn, irrep_dict, alpha, parts)
            matrices.append(res)
            result += res
        np.save(savename, sum(matrices))

    print('Post loading pickle mem usg: {:.4}mb | Final mem usg: {:.4f}mb'.format(mem_usg, check_memory(False)))
    print('Processes')
    for pid, usg in mem_dict.items():
        print('{} | {:.4f}mb'.format(pid, usg))
    print('Done!')

def main(args):
    print('args: {}'.format(args))
    print('split dir: {}'.format(args.splitdir))
    print('pkl dir  : {}'.format(args.pkldir))
    print('save dir : {}'.format(args.savedir))
    print('Evaluating irrep: {}'.format(args.alpha))

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    split_files = [os.path.join(args.splitdir, f) for f in os.listdir(args.splitdir) if args.suffix in f]
    split_chunks = chunk(split_files, args.par)
    parts = ast.literal_eval(args.parts)
    alpha = ast.literal_eval(args.alpha)
    #assert all(sum(parts[i]) == alpha[i] for i in range(len(parts))), 'Invalid partition for alpha!'
    print('About to full transform: {}'.format(split_chunks))
    full_transform(args, alpha, parts, split_chunks)

def test_partial():
    start = time.time()
    irrep_dict = load_pkl('/local/hopan/cube/pickles/(0, 1, 7)/((), (1,), (4, 3)).pkl')
    split_transform('/local/hopan/cube/split/xaa.pkl', irrep_dict, (0,1,7), ((), (1,), (4,3)))
    end = time.time()
    print('Elapsed: {:.2f}'.format(end - start))

if __name__ == '__main__':
    #test_partial()
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--pkldir', type=str, default='/local/hopan/cube/pickles')
    parser.add_argument('--splitdir', type=str, default='/local/hopan/cube/split_unmod')
    parser.add_argument('--savedir', type=str, default='/local/hopan/cube/fourier')
    parser.add_argument('--alpha', type=str, default='(9, 0, 0)')
    parser.add_argument('--parts', type=str, default='((7,1),(),())')
    parser.add_argument('--par', type=int, default=1, help='Amount of parallelism')
    parser.add_argument('--suffix', type=str, default='split', help='special suffix for split files')
    args = parser.parse_args()
    tf(main, [args])
