import os
import argparse
import time
import pdb
import sys
sys.path.append('../')
from yor import yor
from young_tableau import FerrersDiagram
from multiprocessing import Pool
from perm2 import Perm2
from utils import check_memory
import pandas as pd
import numpy as np

def fts(df, ferrers):
    res = None
    for idx, row in df.iterrows():
        ptup = tuple(int(x) for x in row[0])
        perm = Perm2.from_tup(ptup)
        if res is None:
            res = row[1] * yor(ferrers, perm, use_cache=False)
        else:
            res += (row[1] * yor(ferrers, perm, use_cache=False))
    return res 

def par_ft(partition, fname, savedir, ncpu=16):
    if not os.path.exists(savedir):
        try:
            print('Directory {} doesnt exist. creating it now'.format(savedir))
            os.makedirs(savedir)
        except:
            print('Directory {} didnt exist. Tried to make it. Already made. Continuing...'.format(savedir))

    ferrers = FerrersDiagram(partition)
    print('Ferrers:')
    print(ferrers)
    df = pd.read_csv(fname, header=None, dtype={0: str, 1:int})
    check_memory()

    df_chunk = np.array_split(df, ncpu)
    arg_tups = [(chunk, ferrers) for chunk in df_chunk]
    savename = os.path.join(savedir, str(partition))
    print('Saving in: {}'.format(savename))
    if os.path.exists(savename):
        print('{} exists. Not running'.format(savename))

    with Pool(ncpu) as p:
        map_res = p.starmap(fts, arg_tups)
        # sum of these matrices is what we wnat
        fourier_mat = sum(map_res)
        np.save(savename, fourier_mat)
        return fourier_mat

def test_one():
    fname = '/local/hopan/tile/tile3.txt'
    df = pd.read_csv(fname, header=None, dtype={0: str, 1:int})
    df.columns = ['perm', 'distance']
    df = df[df['distance'] < 4]
    print(len(df))
    partition = (8, 1)
    ferrers = FerrersDiagram(partition)
    res = fts(df, ferrers)
    pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', type=str, default='(9,)')
    parser.add_argument('--ncpu', type=int, default=8)
    parser.add_argument('--fname', type=str, default='/local/hopan/tile/tile3.txt') 
    parser.add_argument('--savedir', type=str, default='/local/hopan/tile/fourier_eval') 
    args = parser.parse_args()
    args.partition = eval(args.partition)
    print('Arguments:')
    print(args)

    start =time.time()
    res = par_ft(args.partition, args.fname, args.savedir, args.ncpu)
    print('Done | Fourier matrix size: {}'.format(res.shape))
    end = time.time()
    print('Elapsed time: {:.2f}s'.format(end - start))
