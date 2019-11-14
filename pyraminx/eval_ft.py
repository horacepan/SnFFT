import os
import time
import sys
sys.path.append('../')

from px_utils import *
from test_ft import ift
from test_ft import load_ft, load_wreaths
from tqdm import tqdm
import numpy as np
from utils import check_memory
from multiprocessing import Pool
SAVEPREFIX = '/local/hopan/pyraminx/irreps_mat/'

# loop over the data, evaluatea
def save_mats_bfs_order(alpha, parts):
    # for a given irrep, load all the matrices
    # also want the group elements in some order
    start = time.time()
    dists = dist_df('dists.txt')
    reps_dict = load_wreaths(alpha, parts)
    ft_mat = load_ft(alpha, parts)
    dim = reps_dict[((0, 0, 0, 0, 0, 0), (1, 2, 3, 4, 5, 6))].shape[0]
    output = np.zeros((len(dists), dim, dim))

    for idx in range(len(dists)):
        row = dists.loc[idx]
        ori = row[0]
        perm = row[1]
        output[idx] = reps_dict[(ori, perm)]

    savedir = os.path.join(SAVEPREFIX, str(alpha))
    savename = os.path.join(SAVEPREFIX, str(alpha), str(parts))
    if not os.path.exists(savedir):
        try:
            os.makedirs(savedir)
        except:
            pass

    np.save(savename, output)
    print('Done with {} | {}'.format(alpha, parts))

def convert_to_mat(procid, aps):
    start = time.time()
    print('Starting: {}'.format(procid))
    for alpha, parts in aps:
        save_mats_bfs_order(alpha, parts)
    print('Done {} | Elapsed: {:.2f}s'.format(procid, time.time() - start))

def eval_ft(ap):
    alpha, parts = ap
    ft = load_ft(alpha, parts)
    reps = load_rep_mat(alpha, parts)

    scale = ft.shape[0] / PYRAMINX_GROUP_SIZE
    ift = np.trace(np.matmul(ft.T, reps), axis1=1, axis2=2) * scale

    savepref = '/local/hopan/pyraminx/fourier_eval/'
    savedir = os.path.join(savepref, str(alpha))
    savename = os.path.join(savedir, str(parts))
    if not os.path.exists(savedir):
        try:
            os.makedirs(savedir)
        except:
            pass

    np.save(savename, ift)
    print('Done with {} | {}'.format(alpha, parts))

def eval_subsampled_ft(ap, nsample):
    '''
    ap: tuple of alpha, parts where alpha is a weak partition of a number
        parts is tuple of partitions of each segment of the alpha
    '''   
    alpha, parts = ap
    ft = load_ft_sample(alpha, parts, nsample) # this should load the subsampled_ft
    reps = load_rep_mat(alpha, parts)

    scale = ft.shape[0] / (nsample * 4)
    ift = np.trace(np.matmul(ft.T, reps), axis1=1, axis2=2) * scale

    savepref = '/local/hopan/pyraminx/fourier_eval_sample/{}/'.format(nsample)
    savedir = os.path.join(savepref, str(alpha))
    savename = os.path.join(savedir, str(parts))
    if not os.path.exists(savedir):
        try:
            os.makedirs(savedir)
        except:
            pass

    np.save(savename, ift)
    #print('Done with {} | {}'.format(alpha, parts))
 
def par_main(ncpu):
    aps = alpha_parts()
    cnt = len(aps) // ncpu
    chunked = [aps[i:i+cnt] for i in range(0, len(aps), len(aps) // ncpu)]
    if len(chunked) > ncpu:
        last = chunked.pop(-1)
        chunked[-1] = chunked[-1] + last

    vals = [(idx, v) for idx, v in enumerate(chunked)]
    with Pool(ncpu) as p:
        p.starmap(convert_to_mat, vals)
    check_memory()

def par_ft_eval(ncpu):
    aps = alpha_parts()
    with Pool(ncpu) as p:
        p.map(eval_ft, aps)

def par_ft_sample_eval(ncpu, nsample):
    np.random.seed(0)
    aps = alpha_parts()
    chunk_size = len(aps) // ncpu
    args = [(ap, nsample) for ap in aps]
    with Pool(ncpu) as p:
        p.starmap(eval_subsampled_ft, args)

if __name__ == '__main__':
    np.random.seed(26)
    ncpu = 16
    #par_ft_eval(ncpu)

    samples = [11000, 10000, 9000]
    for nsample in samples:
        start = time.time()
        par_ft_sample_eval(ncpu, nsample)
        elapsed = time.time() - start
        print('Done with {} samples | Elapsed: {:.2f}s'.format(nsample, elapsed))
