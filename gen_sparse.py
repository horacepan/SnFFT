import os
import random
import pickle
import pdb
import numpy as np
import torch
import torch.nn
import glob
from tqdm import tqdm
from utils import load_sparse_pkl, load_pkl, check_memory
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
    vals_im = []

    for idx, mat in mat_dict.items():
        indices = [convert_idx(i, in_cols, out_cols) for i in block_indices(idx, bs)]
        idxs.extend(indices)
        vals_re = np.concatenate((vals_re, mat.real.ravel()))
        vals_im = np.concatenate((vals_im, mat.imag.ravel()))

    torch_i = torch.LongTensor(idxs).t()
    torch_v_re = torch.FloatTensor(vals_re)
    torch_v_im = torch.FloatTensor(vals_im)
    size = (bs**2 * ncosets**2 // out_cols, out_cols)
    return torch_i, torch_v_re, torch_v_im, size

def gen_th_pkl(np_pkl, th_pkl):
    print('Does {} exist? {}'.format(os.path.exists(os.path.dirname(th_pkl)), os.path.dirname(th_pkl)))
    if not os.path.exists(np_pkl):
        print(np_pkl, 'doesnt exist! Exiting!')
        exit()
    else:
        dirname = os.path.dirname(th_pkl)
        try:
            os.makedirs(dirname) # rp
        except:
            print('Exists {}? {}'.format(dirname, os.path.exists(dirname)))
        print('AlreadyExists {}? {}'.format(dirname, os.path.exists(dirname)))

    with open(np_pkl, 'rb') as f:
        ydict = pickle.load(f) 

    sparse_tdict = {}
    for perm_tup, rep_dict in tqdm(ydict.items()):
        idx, vreal, vimag, size = to_block_sparse(rep_dict)
        sparse_tdict[perm_tup] = {
            'idx': idx,
            'real': vreal,
            'imag': vimag,
        }

    # hacky way to assign this
    sparse_tdict['size'] = size

    with open(th_pkl, 'wb') as f:
        pickle.dump(sparse_tdict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Created:', th_pkl)
    print('size' in sparse_tdict)

def compare(np_dict, th_dict):
    for perm, ydict in tqdm(np_dict.items()):
        if perm not in th_dict:
            print('torch pkl doesnt have permutation: {}'.format(perm))
            pdb.set_trace()
        mat = get_mat(perm, np_dict)
        th_mat_re = th_dict[perm]['real'].to_dense().numpy()
        th_mat_im = th_dict[perm]['imag'].to_dense().numpy()
        if not (np.allclose(mat.real.ravel(), th_mat_re) and np.allclose(mat.imag.ravel(), th_mat_im)):
            print('Inconsistency between numpy and torch versions!')
            pdb.set_trace()

def test_th_pkl(np_pkl, th_pkl):
    print('Testing equivalence')
    np_dict = load_pkl(np_pkl)
    th_dict = load_sparse_pkl(th_pkl)
    compare(np_dict, th_dict)
    print('All equal between numpy and torch versions!!')
    check_memory()

def test():
    np_pkl = '/local/hopan/cube/pickles/(2, 3, 3)/((2,), (1, 1, 1), (1, 1, 1)).pkl'
    th_pkl = '/local/hopan/cube/pickles_sparse/(2, 3, 3)/((2,), (1, 1, 1), (1, 1, 1)).pkl'
    gen_th_pkl(np_pkl, th_pkl)
    test_th_pkl(np_pkl, th_pkl)

if __name__ == '__main__':
    test()
