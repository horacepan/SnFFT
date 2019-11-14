import time
import random
import pickle
import pdb
import os
from itertools import product
import numpy as np
import sys
from tqdm import tqdm

sys.path.append('/home/hopan/github/SnFFT/')
from utils import partitions, weak_partitions
from px_utils import *

PX_GROUP_SIZE = 11520 * 4

def load_ft(alpha, parts, prefix='/local/hopan/pyraminx/fourier/'):
    fname = os.path.join(prefix, str(alpha), str(parts) + '.npy')
    return np.load(fname)

def load_wreaths(alpha, parts, prefix='/local/hopan/pyraminx/irreps/'):
    fname = os.path.join(prefix, str(alpha), str(parts) + '.pkl')
    with open(fname, 'rb') as f:
        return pickle.load(f)

def load_all():
    fts = {}
    max_dim = 0
    tot = 0

    for alpha in weak_partitions(6, 2):
        for parts in product(partitions(alpha[0]), partitions(alpha[1])):
            #print(alpha, parts)
            mat = load_ft(alpha, parts)
            max_dim = max(max_dim, mat.shape[0])
            fts[(alpha, parts)] = mat
            tot += (mat.shape[0] ** 2)

    #print('max dim: {}'.format(max_dim))
    #print('Dim', tot)
    return fts

def ift(g, wreath_dict, fourier_mat):
    # trace(A^T B) = vec(A).dot(vec(B))
    # trace(rho(g inv) fhat) = trace(fhat rho(ginv)) = trace(fhat.T rho(g))
    # we can do rho(g inv).T = rho(g) in this situation because real reps
    rep = wreath_dict[g]
    dim = fourier_mat.shape[0]
    return dim * fourier_mat.T.ravel().dot(rep.ravel()) / PX_GROUP_SIZE

def inv_fft_check(nsample):
    start = time.time()
    dist_dict = pyraminx_dists('dists.txt')
    elements = list(dist_dict.keys())
    #sampled_items = random.sample(elements, nsample)
    sampled_items = elements
    ft_sum = np.zeros(len(sampled_items))
    true_sum = np.array([dist_dict[g] for g in sampled_items])

    cnt = 0

    for alpha in weak_partitions(6, 2):
        for parts in product(partitions(alpha[0]), partitions(alpha[1])):
            wreath_dict = load_wreaths(alpha, parts)
            fourier_mat = load_ft(alpha, parts)
            for idx, g in enumerate(sampled_items):
                ft_sum[idx] += ift(g, wreath_dict, fourier_mat)
            print('{:2s} of 65 | Elapsed: {:.2f}s'.format(str(cnt), time.time() - start))
            cnt += 1

    end = time.time() 
    print('Elapsed: {:.2f}s'.format(end - start))
    print('All same: {}'.format(np.allclose(ft_sum, true_sum)))

def inv_fft_check2():
    df = dist_df('dists.txt')
    dists_np = df[2].values
    ift = np.zeros(dists_np.shape)
    res = {}
    for alpha, parts in tqdm(alpha_parts()):
        ift_mat = load_mat_ift(alpha, parts)
        res[(alpha, parts)] = ift_mat
        ift += ift_mat

    print('Same: {}'.format(np.allclose(ift, dists_np)))

def inv_fft_sample(nsample):
    ift = np.zeros(11520)
    res = {}
    for alpha, parts in tqdm(alpha_parts()):
        ift_mat = load_mat_ift_sample(alpha, parts, nsample)
        res[(alpha, parts)] = ift_mat
        ift += ift_mat

    return ift
     
if __name__ == '__main__':
    #inv_fft_check(100)
    inv_fft_check2() 
