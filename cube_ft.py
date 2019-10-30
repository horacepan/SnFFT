import pdb
import sys
import math
import os
import time
import numpy as np
import pandas as pd
from mpi4py import MPI
import argparse

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
        df = load_df('/scratch/hopan/cube/')
        irrep_dict = load_irrep('/scratch/hopan/cube/', alpha, parts)
    except Exception as e:
        print('rank {} | memory usg: {} | exception {}'.format(rank, check_memory(verbose=False), e))

    print('Rank {:3d} / {} | load irrep: {:.2f}s | mem: {}mb'.format(rank, size, time.time() - start, check_memory(verbose=False)))

    cos_reps = coset_reps(sn(8), young_subgroup_perm(alpha))
    save_dict = {}
    cyc_irrep_func = cyclic_irreps(alpha)

    chunk_size = len(df) // size
    start_idx  = chunk_size * rank
    #print('Rank {} | {:7d}-{:7d}'.format(rank, start_idx, start_idx + chunk_size))
    if rank == 0:
        print('Rank {} | elapsed: {:.2f}s | {:.2f}mb | done load'.format(rank, time.time() - start, check_memory(verbose=False)))

    for idx in range(start_idx, start_idx + chunk_size):
        row = df.loc[idx]
        otup = tuple(int(i) for i in row[0])
        perm_tup = tuple(int(i) for i in row[1])
        dist = int(row[2])
 
        perm_rep = irrep_dict[perm_tup]  # perm_rep is a dict of (i, j) -> matrix
        block_cyclic_rep = block_cyclic_irreps(otup, cos_reps, cyc_irrep_func)
        mult_yor_block(perm_rep, dist, block_cyclic_rep, save_dict)

    if rank == 0:
        print('Rank {} | elapsed: {:.2f}s | {:.2f}mb | done add'.format(rank, time.time() - start, check_memory(verbose=False)))

    del irrep_dict
    block_size = wreath_dim(parts)
    n_cosets = coset_size(alpha)
    mat = convert_yor_matrix(save_dict, block_size, n_cosets)
    if rank == 0:
        print('Rank {} | elapsed: {:.2f}s | {:.2f}mb | done matrix conversion'.format(rank, time.time() - start, check_memory(verbose=False)))

    return mat 

def mpi_main(alpha, parts):
    savename = '/scratch/hopan/cube/fourier/{}/{}'.format(alpha, parts)
    if os.path.exists(savename):
        print('File {} exists! Skipping'.format(savename))
        exit()
        #print('File {} exists! Running anyway!'.format(savename))

    comm = MPI.COMM_WORLD
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()
    if rank == 0:
        print('starting {} | {}'.format(alpha, parts))

    _start = time.time()
    start = time.time()
    # mat = par_cube_ft(alpha, parts, irrep_dict, lst)
    mat = par_cube_ft(rank, size, alpha, parts)
    #all_mats = comm.gather(mat, root=0)
    if rank == 0:
        print('post par cube ft: {:.2f}s | mem: {:.2f}mb'.format(time.time() - start, check_memory(verbose=False)))

    sendmat = mat
    recvmat = None
    if rank == 0:
        recvmat = np.empty([size, *sendmat.shape], dtype=sendmat.dtype)
    comm.Gather(sendmat, recvmat, root=0)

    if rank == 0:
        print('Elapsed for gather: {:.2f}s | mem {:.2f}mb'.format(time.time() - start, check_memory(verbose=False)))
        res_mat = np.sum(recvmat, axis=0)
        print('All done | {:.2f}s | shape {} | mem {:.2f}mb'.format(time.time() - _start, res_mat.shape, check_memory(verbose=False)))

        # save dir
        if not os.path.exists('/scratch/hopan/cube/fourier/{}'.format(alpha)):
            os.makedirs('/scratch/hopan/cube/fourier/{}'.format(alpha))
        savename = '/scratch/hopan/cube/fourier/{}/{}'.format(alpha, parts)
        np.save(savename, res_mat)
        print('Done saving in {}!'.format(savename))

def test():
    start = time.time()
    alpha = (2, 3, 3)
    parts = ((2,), (3,), (3,))
    df = load_df('/scratch/hopan/cube/')
    irrep_dict = load_irrep('/scratch/hopan/cube/', alpha, parts)
    end = time.time()
    check_memory()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=str, default='(8, 0, 0)')
    parser.add_argument('--parts', type=str, default='((7,1),(),())')
    args = parser.parse_args()
    alpha = eval(args.alpha)
    parts = eval(args.parts)
    mpi_main(alpha, parts)
