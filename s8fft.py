import pdb
import time
import pickle
from tqdm import tqdm
from yor import yor
from perm2 import Perm2
from young_tableau import FerrersDiagram
from utils import partitions
from multiprocessing import Pool
import numpy as np
import pandas as pd

def stotup(s):
    return tuple(int(i) for i in s)

def read_file(fname):
    with open(fname, 'r') as f:
        ptups = []
        dists = []
        for line in f:
            stup, dist = line.strip().split(',')
            dist = int(dist)
            tup = stotup(stup)
            ptups.append(tup)
            dists.append(dist)

        return ptups, dists

def get_irrep_dict(irrep):
    tfmt = '_'.join(str(i) for i in irrep)
    fname = f'/local/hopan/irreps/s_8/{tfmt}.pkl'
    return pickle.load(open(fname, 'rb'))

def fft(perm_tups, dists, irrep):
    f = FerrersDiagram(irrep)
    irrep_dict = get_irrep_dict(irrep)
    fhat = np.zeros((f.n_tabs(), f.n_tabs()))

    for p, d in zip(perm_tups, dists):
        fhat += (irrep_dict[p] * d)

    return fhat

def non_pkl_fft(perm_tups, dists, irrep):
    f = FerrersDiagram(irrep)
    fhat = np.zeros((f.n_tabs(), f.n_tabs()))

    for p, d in tqdm(zip(perm_tups, dists)):
        fhat += (yor(f, Perm2.from_tup(p), use_cache=False) * d)

    np.save(f'/scratch/hopan/s9puzzle/fourier/{irrep}.npy')
    return fhat

def do_all_ffts(fname):
    st = time.time()
    perm_tups, dists = read_file(fname)
    fhats = []
    s8parts = partitions(8)
    for part in s8parts:
        fhat = fft(perm_tups, dists, part)

        np.save(f'/local/hopan/s8cube/fourier/{part}.npy', fhat)
        fhats.append(fhat)
        print('Done {:2d} / {:2d} | Elapsed: {:.2f}mins'.format(len(fhats), len(s8parts), (time.time() - st) / 60.))

def par_ft(perms, dists):
    p9s = list(partitions(9))
    args = [(perms, dists, p) for p in p9s]
    st = time.time()
    with Pool(30) as pool:
        pool.starmap(non_pkl_fft, args)
    print('Done all: {:.2f}min'.format((end - st) / 60.))

if __name__ == '__main__':
    fname = '/home/hopan/github/idastar/s9_dists.txt'
    ptups, dists = read_file(fname)
    par_ft(ptups, dists)
    #fname = '/home/hopan/github/idastar/s8_dists_red.txt'
    #do_all_ffts(fname)
