import os
import sys
import time
import argparse
from tqdm import tqdm
from fourier_policy import FourierPolicy
from perm_df import PermDF, nbrs

sys.path.append('../')
from utils import check_memory

def main(args):
    fname = '/home/hopan/github/idastar/s8_dists_fixed.txt'
    if os.path.exists('/local/hopan/irreps'):
        pkl_prefix = '/local/hopan/irreps/s_8'
    else:
        pkl_prefix = '/scratch/hopan/irreps/s_8'

    irreps = [(3, 2, 2, 1), (4, 2, 2), (4, 2, 1, 1), (3, 3, 1, 1)]
    #irreps = [(7,1), (6,2), (6, 1,1), (5,3)]
    test_ratio = args.testratio

    perm_df = PermDF(fname, nbrs)
    print('Doing the split')
    split_st = time.time()
    train_p, train_y, test_p, test_y = perm_df.train_test_split(test_ratio)
    print('Done split: {:.2f}s'.format(time.time() - split_st))
    print('Test ratio: {} | Test items: {}'.format(len(test_p) / len(perm_df.df), len(test_p)))

    print('Random policy prop correct: {}'.format(perm_df.benchmark(test_p)))

    policy = FourierPolicy(irreps[:args.topk], pkl_prefix)
    print('Starting the fit | Fitting {} perms'.format(len(train_p)))
    fit_st = time.time()
    policy.fit_perms(train_p, train_y)
    print('Fit time: {:.2f}s'.format(time.time() - fit_st))

    ncorrect = 0
    for gtup in tqdm(test_p):
        ncorrect += int(perm_df.opt_nbr(gtup, policy))
    print('Correct rate: {} / {} | {}'.format(ncorrect, len(test_p), ncorrect / len(test_p)))

    ncorrect = 0
    for gtup in tqdm(train_p):
        ncorrect += int(perm_df.opt_nbr(gtup, policy))
    print('Correct rate: {} / {} | {}'.format(ncorrect, len(train_p), ncorrect / len(train_p)))
    print('Random policy prop correct: {}'.format(perm_df.benchmark(test_p)))
    check_memory()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--testratio', type=float, default=0.05)
    args = parser.parse_args()
    main(args)
