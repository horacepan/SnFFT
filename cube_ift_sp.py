import pdb
import sys
import math
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from mpi4py import MPI
import argparse

from multiprocessing import Pool
from tqdm import tqdm
from utils import load_pkl, load_irrep, check_memory, chunk, load_pkl
from wreath import cyclic_irreps, block_cyclic_irreps, wreath_rep, wreath_rep_sp
from young_tableau import wreath_dim
from perm2 import sn
from coset_utils import young_subgroup_perm, coset_reps
from multi import mult_yor_block, coset_size, convert_yor_matrix, clean_line
import torch
from itertools import product

def all_cyc_irreps(cos_reps, cyc_irrep_func):
    otups = []
    irreps = {}
    xs = [(0, 1, 2) for _ in range(8)]
    opts = product(*xs)
    opts = [o for o in opts if sum(o) % 3 == 0]
    for otup in opts:
        irreps[otup] = block_cyclic_irreps(otup, cos_reps, cyc_irrep_func)
    return irreps

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

def par_cube_ift(rank, size, alpha, parts):
    start = time.time()
    try:
        df = load_df('/scratch/hopan/cube/')
        sp_irrep_dict = load_pkl('/scratch/hopan/cube/pickles_sparse/{}/{}.pkl'.format(alpha, parts))
        fhat = np.load('/scratch/hopan/cube/fourier/{}/{}.npy'.format(alpha, parts))
    except Exception as e:
        print('rank {} | memory usg: {} | exception {}'.format(rank, check_memory(verbose=False), e))

    print('Rank {:3d} / {} | load irrep: {:.2f}s | mem: {:.2f}mb | {} {}'.format(rank, size, time.time() - start, check_memory(verbose=False), alpha, parts))

    cos_reps = coset_reps(sn(8), young_subgroup_perm(alpha))
    cyc_irrep_func = cyclic_irreps(alpha)
    cyc_irrs = all_cyc_irreps(cos_reps, cyc_irrep_func)

    chunk_size = len(df) // size
    start_idx = chunk_size * rank
    mat = np.zeros(chunk_size, dtype=fhat.dtype)
    fhat_t_ravel = fhat.T.ravel()
    #print('Rank {} | {:7d}-{:7d}'.format(rank, start_idx, start_idx + chunk_size))
    if rank == 0:
        print('Rank {} | elapsed: {:.2f}s | {:.2f}mb | mat shape: {} | done load | {} {}'.format(rank, time.time() - start, check_memory(verbose=False), fhat.shape, alpha, parts))

    st = time.time()
    for idx in range(start_idx, start_idx + chunk_size):
        row = df.loc[idx]
        otup = tuple(int(i) for i in row[0])
        perm_tup = tuple(int(i) for i in row[1])
        #dist = int(row[2])
        # actually want the inverse

        wmat_sp = wreath_rep_sp(otup, perm_tup, sp_irrep_dict, cos_reps, cyc_irrep_func, cyc_irrs)
        wmat_inv_sp = wmat_sp.conj().T
        # trace(rho(ginv) fhat) = trace(fhat rho(ginv)) = vec(fhat.T).dot(vec(rho(ginv)))
        #feval = np.dot(fhat.T.ravel(), wmat_inv.ravel())
        #feval = np.dot(fhat_t_ravel, wmat_inv.ravel())
        feval_sp = (wmat_inv_sp.multiply(fhat.T)).sum()
        mat[idx - start_idx] = fhat.shape[0] * feval_sp

    if rank == 0:
        elapsed = time.time() - st
        avg_t = elapsed / chunk_size
        print('Rank {} | elapsed: {:.2f}s | {:.2f}mb | done add | ift time: {:.2f}s | avg time: {:.6f}s'.format(rank, time.time() - start, check_memory(verbose=False), elapsed, avg_t))

    del sp_irrep_dict
    if rank == 0:
        print('Rank {} | elapsed: {:.2f}s | {:.2f}mb | done matrix conversion'.format(rank, time.time() - start, check_memory(verbose=False)))

    return mat 

def mpi_main(alpha, parts):
    savename = '/scratch/hopan/cube/fourier_sym_eval/{}/{}.npy'.format(alpha, parts)
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
    mat = par_cube_ift(rank, size, alpha, parts)
    #all_mats = comm.gather(mat, root=0)
    if rank == 0:
        print('post par cube ft: {:.2f}s | mem: {:.2f}mb'.format(time.time() - start, check_memory(verbose=False)))

    sendmat = mat
    recvmat = None
    if rank == 0:
        recvmat = np.empty([size, *sendmat.shape], dtype=sendmat.dtype)
        print('Before gather: {:.2f}s | mem {:.2f}mb'.format(time.time() - start, check_memory(verbose=False)))
    comm.Gather(sendmat, recvmat, root=0)

    if rank == 0:
        print('Elapsed for gather: {:.2f}s | mem {:.2f}mb'.format(time.time() - start, check_memory(verbose=False)))
        #res_mat = np.sum(recvmat, axis=0)
        res_mat = recvmat.reshape(-1)
        print('All done | {:.2f}s | shape {} | mem {:.2f}mb'.format(time.time() - _start, res_mat.shape, check_memory(verbose=False)))

        # save dir
        if not os.path.exists('/scratch/hopan/cube/fourier_sym_eval/{}'.format(alpha)):
            os.makedirs('/scratch/hopan/cube/fourier_sym_eval/{}'.format(alpha))
        savename = '/scratch/hopan/cube/fourier_sym_eval/{}/{}'.format(alpha, parts)
        np.save(savename, res_mat)
        print('Done saving in {}! | Total time: {:.2f}s'.format(savename, time.time() - _start))

def test_main(alpha, parts):
    '''
    Computes the ft via the sparse wreath rep and the non-sparse wreath rep
    to double check that the sparse wreath rep is actually correct.
    '''
    _start = time.time()
    st = time.time()
    sp_irrep_dict = load_pkl('/scratch/hopan/cube/pickles_sparse/{}/{}.pkl'.format(alpha, parts))
    end = time.time()
    print('Loading sparse irrep dict: {:.2f}s'.format(time.time() - st))
    check_memory()

    st = time.time()
    irrep_dict = load_irrep('/scratch/hopan/cube/', alpha, parts)
    print('Loading irrep dict: {:.2f}s'.format(time.time() - st))
    check_memory()

    # generate a random group element?
    st = time.time()
    df = load_df('/scratch/hopan/cube/')
    fhat = np.load('/scratch/hopan/cube/fourier/{}/{}.npy'.format(alpha, parts))
    print('Loading df: {:.2f}s'.format(time.time() - st))
    check_memory()

    cyc_irrep_func = cyclic_irreps(alpha)
    cos_reps = coset_reps(sn(8), young_subgroup_perm(alpha))
    st = time.time()
    cyc_irrs = all_cyc_irreps(cos_reps, cyc_irrep_func)
    print('Time to compute all cyc irreps: {:.5f}s'.format(time.time() - st))

    sp_times = []
    sp_mult_times = []
    sp_results = np.zeros(len(df), dtype=np.complex128)

    coo_times = []
    th_sp_times = []
    times = []
    mult_times = []
    z3_irreps = []
    results = np.zeros(len(df), dtype=np.complex128)
    fhat_t_ravel = fhat.T.ravel()
    loop_start = time.time()
    for idx in range(len(df)):
        row = df.loc[idx]
        otup = tuple(int(i) for i in row[0])
        perm_tup = tuple(int(i) for i in row[1])

        # compute wreath rep
        st = time.time()
        wmat = wreath_rep(otup, perm_tup, irrep_dict, cos_reps, cyc_irrep_func)
        reg_time = time.time() - st

        # compute wreath rep multiply
        st = time.time()
        wmat_inv = wmat.conj().T
        feval = np.dot(fhat_t_ravel, wmat_inv.ravel())
        reg_mult_time = time.time() - st
        results[idx] = feval

        # compute sparse wreath rep
        st = time.time()
        wmat_sp = wreath_rep_sp(otup, perm_tup, sp_irrep_dict, cos_reps, cyc_irrep_func, cyc_irrs)
        sp_time = time.time() - st

        if not np.allclose(wmat, wmat_sp.todense()):
            print('unequal! | idx = {}'.format(idx))
            pdb.set_trace()

        # compute sparse wreath rep multiply
        st = time.time()
        wmat_inv_sp = wmat_sp.conj().T
        feval_sp = (wmat_inv_sp.multiply(fhat.T)).sum()
        sp_mult_time = time.time() - st
        sp_results[idx] = feval_sp

        times.append(reg_time)
        sp_times.append(sp_time)
        mult_times.append(reg_mult_time)
        sp_mult_times.append(sp_mult_time)

        st = time.time()
        coo = wmat_sp.tocoo()
        end = time.time()
        coo_times.append(end - st)

        st = time.time()
        ix = torch.LongTensor([coo.row, coo.col])
        th_sp_re = torch.sparse.FloatTensor(ix, torch.FloatTensor(coo.data.real), torch.Size(coo.shape))
        th_sp_cplx = torch.sparse.FloatTensor(ix, torch.FloatTensor(coo.data.imag), torch.Size(coo.shape))
        end = time.time()
        th_sp_times.append(end - st)


        st = time.time()
        block_scalars = block_cyclic_irreps(otup, cos_reps, cyc_irrep_func)
        end = time.time()
        z3_irreps.append(end - st) 
        if idx > 200:
            break

    print('Normal time: {:.6f}s | Sparse time: {:.6f}s'.format(np.mean(times), np.mean(sp_times)))
    print('Mult time:   {:.6f}s | Spmult time: {:.6f}s'.format(np.mean(mult_times), np.mean(sp_mult_times)))
    print('To coo time: {:.6f}s | Torchsptime: {:.6f}s'.format(np.mean(coo_times), np.mean(th_sp_times)))
    print('irrep time:  {:.6f}s'.format(np.mean(z3_irreps)))
    print('Loop time: {:.2f}s'.format(time.time() - loop_start))
    print('Total time: {:.2f}s'.format(time.time() - _start))
    #assert np.allclose(sp_results, results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=str, default='(2, 3, 3)')
    parser.add_argument('--parts', type=str, default='((2,), (1, 1, 1), (2, 1))')
    args = parser.parse_args()
    alpha = eval(args.alpha)
    parts = eval(args.parts)
    #test_main(alpha, parts)
    mpi_main(alpha, parts)
