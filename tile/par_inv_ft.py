import socket
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

S9 = 362880.0

def inv_ft_part2(g, ferrers, fhat):
    '''
    g: tuple of ints
    '''
    rep = yor(ferrers, g, use_cache=False) 
    dim = mat.shape[0]
    result = dim * fhat.T.ravel().dot(rep.ravel())
    return result

def inv_transform(perm_df, ferrers):
    # need to load the appropriate matrix
    try:
        fhat_loc = '/local/hopan/tile/fourier_eval/{}.npy'.format(ferrers.partition)
        fhat = np.load(fhat_loc)
    except:
        fhat_loc = '/scratch/hopan/tile/fourier_eval/{}.npy'.format(ferrers.partition)
        fhat = np.load(fhat_loc)

    ifts = []
    for perm_num in perm_df[0]:
        # g is a number, turn it into a tuple
        g = Perm2.from_tup(tuple(int(x) for x in str(perm_num)))
        rep = yor(ferrers, g, use_cache=False)
        dim = fhat.shape[0]
        result = dim * (fhat * rep).sum() / S9
        ifts.append(result)
 
    return ifts

def par_inv_ft(partition, fname, savedir, ncpu=16):
    if not os.path.exists(savedir):
        try:
            print('Directory {} doesnt exist. creating it now'.format(savedir))
            os.makedirs(savedir)
        except:
            print('Directory {} didnt exist. Tried to make it. Already made. Continuing...'.format(savedir))

    ferrers = FerrersDiagram(partition)
    df = pd.read_csv(fname, header=None, dtype={0: str, 1:int})
    check_memory()

    df_chunk = np.array_split(df, ncpu)
    arg_tups = [(chunk, ferrers) for chunk in df_chunk]
    savename = os.path.join(savedir, str(partition)) + '.csv'
    print('Saving in: {}'.format(savename))
    if os.path.exists(savename):
        print('{} exists. Not running'.format(savename))
        return

    with Pool(ncpu) as p:
        results = p.starmap(inv_transform, arg_tups)
        concat_results = sum(results, [])
        df[1] = concat_results
        df.to_csv(savename, header=None, index=False)

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', type=str, default='(9,)')
    parser.add_argument('--ncpu', type=int, default=8)
    parser.add_argument('--fname', type=str, default='/local/hopan/tile/tile3.txt') 
    parser.add_argument('--savedir', type=str, default='/local/hopan/tile/dist_inv_ft') 
    args = parser.parse_args()
    args.partition = eval(args.partition)
    print('Arguments:')
    print(args)

    start = time.time()
    res = par_inv_ft(args.partition, args.fname, args.savedir, args.ncpu)
    end = time.time()
    print('Elapsed time: {:.2f}s'.format(end - start))
