import pdb
import time
import pickle
from yor import yor
from young_tableau import FerrersDiagram
from utils import partitions
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

if __name__ == '__main__':
    fname = '/home/hopan/github/idastar/s8_dists_red.txt'
    do_all_ffts(fname)
