import os
import argparse

import time
import pdb
import sys
sys.path.append('./cube')

import pickle
import pandas as pd
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from utils import check_memory
from str_cube import *

TOL = 1e-4

if os.path.exists('/local/hopan'):
    PREFIX = '/local/hopan/cube'
elif os.path.exists('/scratch/hopan'):
    PREFIX = '/scratch/hopan/cube'
else:
    PREFIX = '/project2/risi/cube/'

def load_cube_df(nrows=None):
    df = pd.read_csv(os.path.join(PREFIX, 'cube_sym_mod.txt'), header=None, dtype={0:str, 1:int}, nrows=nrows)
    df.columns = ['cube', 'distance']
    return df

def load_cube_df_indexed(nrows=None):
    df = pd.read_csv(os.path.join(PREFIX, 'cube_sym_mod.txt'), header=None, dtype={0:str, 1:int}, nrows=nrows)
    df.columns = ['cube', 'distance', ]
    df['index'] = df.index
    df = df.set_index('cube')
    return df

def load_pkls():
    with open('/local/hopan/cube/cube2_pkls/idx_to_nbrs.pkl', 'rb') as f:
        idx_to_nbrs = pickle.load(f)
    with open('/local/hopan/cube/cube2_pkls/idx_to_cube.pkl', 'rb') as f:
        idx_to_cube = pickle.load(f)
    with open('/local/hopan/cube/cube2_pkls/idx_to_dist.pkl', 'rb') as f:
        idx_to_dist = pickle.load(f)
    with open('/local/hopan/cube/cube2_pkls/cube_to_idx.pkl', 'rb') as f:
        cube_to_idx = pickle.load(f)

    return idx_to_nbrs, idx_to_cube, idx_to_dist, cube_to_idx

def par_main(par_f, ncpu):
    global all_df
    all_df = load_cube_df_indexed()
    start = time.time()
    df_chunk = np.array_split(all_df, ncpu)
    idx_to_nbrs, idx_to_cube, idx_to_dist, cube_to_idx = load_pkls()
    arg_tups = [(idx, _d) for idx, _d in enumerate(df_chunk)]

    print('Starting par proc with {} processes...'.format(ncpu))
    check_memory()
    with Pool(ncpu) as p:
        map_res = p.starmap(par_f, arg_tups)

    print('Elapsed proc time: {:.2f}min'.format( (time.time() - start) / 60. ))
    return map_res

def irrep_feval(alpha, parts):
    fname_fmt = os.path.join(PREFIX, 'fourier_eval_sym_all/{}/{}.npy')
    print(fname_fmt)
    return np.load(fname_fmt.format(alpha, parts))

def par_irrep_main(par_f, alpha, parts, ncpu):
    global all_df
    all_df = load_cube_df_indexed()
    df_chunk = np.array_split(all_df[:20000], ncpu)
    real_mat = irrep_feval(alpha, parts).real
    arg_tups = [(idx, _d, real_mat) for idx, _d in enumerate(df_chunk)]

    print('Before pool | ', end='')
    check_memory()

    with Pool(ncpu) as p:
        map_res = p.starmap(par_f, arg_tups)
        par_correct, par_chosen_cubes = zip(*map_res) 

        cat_correct = np.concatenate(par_correct)
        cat_chosen = np.concatenate(par_chosen_cubes)

    return cat_correct, cat_chosen

def proc_baseline(idx, df):
    '''
    df: dataframe whose index is cube strings
    '''
    global all_df
    props = np.zeros(len(df))
    i = 0
    for c in tqdm(df.index):
        nbrs = neighbors_fixed_core_small(c)
        nbr_df = all_df.loc[nbrs]
        nbr_idx = nbr_df['index'] # need this for indexing into mat
        min_dist = nbr_df.distance.min()
        min_cubes = nbr_df[nbr_df.distance == min_dist]

        props[i] = len(min_cubes) / len(nbrs)
        i += 1

    print('Proc {:2} done'.format(idx))
    return props

def proc_baseline_df(idx, df, mat):
    global all_df
    i = 0
    start =time.time()
    correct = np.zeros(len(df))
    chosen_cubes = np.zeros(len(df), dtype=int)

    print('Proc {:2} started | '.format(idx), end='')
    check_memory()
 
    for c in (df.index):
        nbrs = neighbors_fixed_core_small(c)
        nbr_df = all_df.loc[nbrs]
        nbr_idx = nbr_df['index'] # need this for indexing into mat
        min_dist = nbr_df.distance.min()
        min_cubes = nbr_df[nbr_df.distance == min_dist]

        vals = mat[nbr_idx]
        n_idx = np.argmin(vals)
        min_irrep_cube = nbrs[n_idx] # this gives the index
        correct[i] = (min_irrep_cube in min_cubes.index)
        chosen_cubes[i] = all_df.loc[min_irrep_cube]['index']
        i += 1

    end = time.time()
    print('Proc {:2} done: {:.2f}mins'.format(idx, (end - start) / 60.))
    return correct, chosen_cubes


def maybe_mkdir(d):
    try:
        os.makedirs(d)
    except:
        print('Already exists: {}'.format(d))

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=str, default='(2, 3, 3)')
    parser.add_argument('--parts', type=str, default='((2,), (1, 1, 1), (1, 1, 1))')
    parser.add_argument('--ncpu', type=int, default=8, help='Amount of parallelism')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--savedir', type=str, default=os.path.join(PREFIX, 'fourier_eval_results'))
    args = parser.parse_args()

    if args.baseline:
        random_baseline_fname = os.path.join(args.savedir, 'baseline')
        res = par_main(proc_baseline, args.ncpu)
        print('Avg: {:.5f}'.format(np.mean(res)))
        np.save(random_baseline_fname, res)
    else:
        alpha = eval(args.alpha)
        parts = eval(args.parts)
        correct, chosen = par_irrep_main(proc_baseline_df, alpha, parts, args.ncpu)
        print('{:.8f}|{}|{}'.format(np.mean(correct), alpha, parts))

        corr_dir = os.path.join(args.savedir, str(alpha), str(parts))
        chos_dir = os.path.join(args.savedir, str(alpha), str(parts))
        maybe_mkdir(corr_dir)
        maybe_mkdir(chos_dir)

        corr_fname = os.path.join(corr_dir, 'correct.npy')
        chos_fname = os.path.join(chos_dir, 'chosen.npy')
        print('Saving to: {}'.format(corr_fname))
        print('Saving to: {}'.format(chos_fname))
        np.save(corr_fname, correct)
        np.save(chos_fname, chosen)
        elapsed_time = (time.time() - start) / 60.
        print('Done! Total time: {:.2f}mins'.format(elapsed_time))
