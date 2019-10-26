import pdb
import sys
import math
import os
import time
import numpy as np
import pandas as pd
from mpi4py import MPI

from multiprocessing import Pool
from tqdm import tqdm
from utils import load_pkl, load_irrep, check_memory, chunk
from wreath import cyclic_irreps, block_cyclic_irreps, wreath_rep
from young_tableau import wreath_dim
from perm2 import sn
from coset_utils import young_subgroup_perm, coset_reps
from multi import mult_yor_block, coset_size, convert_yor_matrix, clean_line

def load_np_data():
    fname = '/local/hopan/cube/cube_sym_mod_tup.npy'
    return np.load(fname)

def load_df(prefix='/local/hopan/cube/'):
    fname = os.path.join(prefix, 'cube_sym_mod_tup.txt')
    df = pd.read_csv(fname, header=None, dtype={0:str, 1:str, 2:int})
    return df

def load_csv():
    f = open('/local/hopan/cube/cube_sym_mod_tup.txt')
    return [clean_line(l) for l in f.readlines()] 

def par_cube_ft(rank, size, alpha, parts):
    start = time.time()
    try:
        df = load_df()
        irrep_dict = load_irrep('/local/hopan/cube/', alpha, parts)
    except:
        df = load_df('/scratch/hopan/cube/')
        irrep_dict = load_irrep('/scratch/hopan/cube/', alpha, parts)
    print('rank {:2d} | load irrep: {:.2f}s'.format(rank, time.time() - start))

    cos_reps = coset_reps(sn(8), young_subgroup_perm(alpha))
    save_dict = {}
    cyc_irrep_func = cyclic_irreps(alpha)

    chunk_size = len(df) // size
    start_idx  = chunk_size * rank
    for idx in range(start_idx, start_idx + chunk_size):
        row = df.loc[idx]
        otup = tuple(int(i) for i in row[0])
        perm_tup = tuple(int(i) for i in row[1])
        dist = int(row[2])
 
        perm_rep = irrep_dict[perm_tup]  # perm_rep is a dict of (i, j) -> matrix
        block_cyclic_rep = block_cyclic_irreps(otup, cos_reps, cyc_irrep_func)
        mult_yor_block(perm_rep, dist, block_cyclic_rep, save_dict)

    block_size = wreath_dim(parts)
    n_cosets = coset_size(alpha)
    mat = convert_yor_matrix(save_dict, block_size, n_cosets)
    return mat 

def mpi_main():
    comm = MPI.COMM_WORLD
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    _start = time.time()
    alpha = (8, 0, 0)
    parts = ((5, 3), (), ())

    start = time.time()
    # mat = par_cube_ft(alpha, parts, irrep_dict, lst)
    mat = par_cube_ft(rank, size, alpha, parts)

    start = time.time()
    all_mats = comm.gather(mat, root=0)
    if rank == 0:
        print('Elapsed for gather: {:.2f}s'.format(time.time() - start))

    res_mat = np.sum(all_mats, axis=0)
    if rank == 0:
        print('All done | {:.2f}s | shape {}'.format(time.time() - _start, res_mat.shape))

def test():
    start = time.time()
    alpha = (2, 3, 3)
    parts = ((2,), (3,), (3,))
    df = load_df('/scratch/hopan/cube/')
    irrep_dict = load_irrep('/scratch/hopan/cube/', alpha, parts)
    end = time.time()
    check_memory()

if __name__ == '__main__':
    mpi_main()
