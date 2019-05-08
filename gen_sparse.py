import sys
import os
import random
import pickle
import pdb
import numpy as np
import torch
import torch.nn
import glob
from tqdm import tqdm
from utils import load_sparse_pkl, load_pkl, check_memory, partition_parts
from wreath import get_mat

def convert_idx(idx, in_cols, out_cols):
    '''
    idx: tuple of ints for the index of an element in a (nrows x in_cols) matrix
    in_cols: number of cols of matrix containing the given index
    out_cols: number of cols of output matrix we are converting the idx with respect to
    Returns: tuple
    '''
    raw_idx = (idx[0] * in_cols) + idx[1] # flat index
    out_idx = (raw_idx // out_cols, raw_idx % out_cols)
    return out_idx

def block_indices(idx, bs):
    '''
    idx: tuple of ints
    bs: block size
    Returns a list of the indices of the block that the given index would refer to, where
        each index points to a bs x bs sized submatrix.
    '''
    x, y = idx
    return [(x*bs + i, y*bs + j) for i in range(bs) for j in range(bs)]

def to_block_sparse(mat_dict, outshape=None):
    ncosets = len(list(mat_dict.keys()))
    mat_0 = list(mat_dict.values())[0]
    bs = mat_0.shape[0]
    in_cols = ncosets * bs

    if outshape is None:
        out_cols = (ncosets * bs)**2
    else:
        out_cols = outshape[1]

    idxs = []
    vals_re = []

    for idx, mat in mat_dict.items():
        indices = [convert_idx(i, in_cols, out_cols) for i in block_indices(idx, bs)]
        idxs.extend(indices)
        vals_re = np.concatenate((vals_re, mat.real.ravel()))

    torch_i = torch.LongTensor(idxs).t()
    torch_v_re = torch.FloatTensor(vals_re)
    size = (bs**2 * ncosets**2 // out_cols, out_cols)
    return torch_i, torch_v_re, size

def gen_th_pkl(np_pkl, th_pkl):
    if os.path.exists(th_pkl):
        print('Skipping pkl: {}'.format(th_pkl))
        #return
    else:
        print('Not skipping pkl: {}'.format(th_pkl))

    if not os.path.exists(np_pkl):
        print(np_pkl, 'doesnt exist! Exiting!')
        exit()
    else:
        dirname = os.path.dirname(th_pkl)
        try:
            os.makedirs(dirname) # rp
        except:
            print('makedirs: Director already exists {}? {}'.format(dirname, os.path.exists(dirname)))
    print('trying to open: {}'.format(np_pkl))
    with open(np_pkl, 'rb') as f:
        ydict = pickle.load(f) 

    check_memory()
    print('after loading {}'.format(np_pkl))

    sparse_tdict = {}
    for perm_tup, rep_dict in tqdm(ydict.items()):
        idx, vreal, size = to_block_sparse(rep_dict)
        sparse_tdict[perm_tup] = {
            'idx': idx,
            'real': vreal,
        }

    check_memory()
    print('making the sparse dict loading {}'.format(th_pkl))
    del ydict

    # hacky way to assign this
    sparse_tdict['size'] = size

    #with open(th_pkl, 'wb') as f:
    #    pickle.dump(sparse_tdict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Created:', th_pkl)
    del sparse_tdict

def compare(np_dict, th_dict):
    for perm, ydict in (np_dict.items()):
        if perm not in th_dict:
            print('torch pkl doesnt have permutation: {}'.format(perm))
            pdb.set_trace()
        mat = get_mat(perm, np_dict)
        th_mat_re = th_dict[perm].to_dense().numpy()
        if not np.allclose(mat.ravel(), th_mat_re):
            print('Inconsistency between numpy and torch versions!')
            pdb.set_trace()

def test_th_pkl(np_pkl, th_pkl):
    print('Testing equivalence')
    np_dict = load_pkl(np_pkl)
    th_dict = load_sparse_pkl(th_pkl)
    compare(np_dict, th_dict)
    print('All equal between numpy and torch versions!!')
    check_memory()

def gen_pickle_name(suffix, alpha, p):
    if os.path.exists('/local/hopan'):
        return '/local/hopan/cube/{}/{}/{}.pkl'.format(suffix, alpha, p)
    elif os.path.exists('/scratch/hopan/'):
        return '/scratch/hopan/cube/{}/{}/{}.pkl'.format(suffix, alpha, p)
    elif os.path.exists('/project2/risi'):
        return '/project2/risi/cube/{}/{}/{}.pkl'.format(suffix, alpha, p)
    else:
        raise Exception

def test():
    alphas = [(2, 3, 3)]
    parts = [((2,), (1, 1, 1), (1, 1, 1))]
    pset = set()
    pkls = []
    mem_usg = [0]
    for alpha in alphas:
        print('Computing sparse pickles for: {}'.format(alpha))
        #parts = partition_parts(alpha)
        for idx, p in enumerate(parts):
            other = (p[0], p[2], p[1])
            if other in pset:
                continue

            np_pkl = gen_pickle_name('pickles', alpha, p)
            th_pkl = gen_pickle_name('pickles_sparse', alpha, p)
            gen_th_pkl(np_pkl, th_pkl)

            print('Done with {} | {}'.format(alpha, p))
            curr = check_memory()
            print('{:2} irreps | {:30}: {:9.2f} | '.format(idx+1, str(p), curr - mem_usg[-1]), end=' | ')
            mem_usg.append(curr)
            pset.add(p)
    print('Done')
    check_memory()

if __name__ == '__main__':
    alphas = [
        (2, 3, 3),
        (4, 2, 2),
        (3, 1, 4),
        (1, 2, 5),
        (0, 4, 4),
        (5, 0, 3),
        (6, 1, 1),
        (2, 0, 6),
        (0, 1, 7),
        (8, 0, 0),
    ]
    test()
