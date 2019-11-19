import os
import pdb
import time
import random
import pickle
import argparse
from tqdm import tqdm
import numpy as np
from wreath import get_mat
from utils import load_irrep, check_memory
from scipy.sparse import csr_matrix

def convert(alpha, parts, prefix='/local/hopan/cube/', irrep_dict=None):
    if irrep_dict is None: 
        irrep_dict = load_irrep(prefix, alpha, parts)
    sp_dict = {}

    keys = list(irrep_dict.keys())
    for k in keys:
        matdict = irrep_dict[k]
        sp_dict[k] = convert_mat_dict(matdict, alpha, parts)

    return sp_dict

def convert_row_idx(rowidxs, i, j, block_size):
    '''
    Convert the row indices to match 
    '''
    return i * block_size  + rowidxs

def convert_col_idx(colidxs, i, j, block_size):
    return j * block_size  + colidxs 
    
def convert_mat_dict(matdict, alpha, parts):
    # need block size, number of blocks (num cosets)
    rows = []
    cols = []
    vals = []
    num_blocks = len(matdict) # (i, j) -> block
    block_size = 0

    for (i, j), mat in matdict.items():
        block_size = mat.shape[0]
        nzrows, nzcols = mat.nonzero()
        nzvals = [mat[tup] for tup in zip(nzrows, nzcols)]
        glob_rows = convert_row_idx(nzrows, i, j, block_size)
        glob_cols = convert_col_idx(nzcols, i, j, block_size)

        vals.extend(nzvals) 
        rows.extend(glob_rows)
        cols.extend(glob_cols)
        # convert these to global indices
    return csr_matrix((vals, (rows, cols)), shape=(num_blocks * block_size, num_blocks * block_size))

def save_sp_pkl(sp_dict, savedir_top, alpha, parts):
    savedir = os.path.join(savedir_top, 'pickles_sparse', str(alpha))
    savename = os.path.join(savedir, str(parts) + '.pkl')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    print('Saving in: {}'.format(savename))
    f = open(savename, 'wb')
    pickle.dump(sp_dict, f)

def main(alpha, parts, savedir):
    st = time.time()
    irrep_dict = load_irrep(savedir, alpha, parts)
    end = time.time()
    check_memory()

    print('Load time {:.2f}s | {} {}'.format(end - st, alpha, parts))
    sp_dict = convert(alpha, parts, irrep_dict=irrep_dict)
    print('Convert time {:.2f}s'.format(time.time() - end))
    check_memory()
   
    save_sp_pkl(sp_dict, savedir, alpha, parts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=str, default='(8, 0, 0)')
    parser.add_argument('--parts', type=str, default='((7,1),(),())')
    parser.add_argument('--savedir', type=str, default='/local/hopan/cube/')
    args = parser.parse_args()
    main(args.alpha, args.parts, args.savedir)
