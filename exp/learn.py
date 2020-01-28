import os
import sys
import time
import argparse
from tqdm import tqdm
from fourier_policy import FourierPolicy
from perm_df import PermDF, nbrs
from logger import get_logger
sys.path.append('../')
from utils import check_memory

def main(args):
    log = get_logger(f'./{args.logfile}')
    irreps = [(3, 2, 2, 1), (4, 2, 2), (4, 2, 1, 1), (3, 3, 1, 1)]

    perm_df = PermDF(args.fname, nbrs)
    log.info('Splitting data')
    train_p, train_y, test_p, test_y = perm_df.train_test_split(args.testratio)
    log.info('Test ratio: {} | Test items: {}'.format(len(test_p) / len(perm_df.df), len(test_p)))

    random_correct = perm_df.benchmark(test_p)
    log.info('Random policy prop correct: {}'.format(random_correct))

    policy = FourierPolicy(irreps[:args.topk], args.pklprefix)
    log.info('Starting the fit | Fitting {} perms'.format(len(train_p)))
    policy.fit_perms(train_p, train_y)
    log.info('Done fitting')

    ncorrect = 0
    for gtup in tqdm(test_p):
        ncorrect += int(perm_df.opt_nbr(gtup, policy))
    log.info('Correct rate: {} / {} | {}'.format(ncorrect, len(test_p), ncorrect / len(test_p)))

    ncorrect = 0
    for gtup in tqdm(train_p):
        ncorrect += int(perm_df.opt_nbr(gtup, policy))
    log.info('Correct rate: {} / {} | {}'.format(ncorrect, len(train_p), ncorrect / len(train_p)))
    log.info('Random policy prop correct: {}'.format(perm_df.benchmark(test_p)))
    log.info(f'Mem footprint: {check_memory()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--testratio', type=float, default=0.05)
    parser.add_argument('--fname', type=str, default='/home/hopan/github/idastar/s8_dists_fixed.txt')
    parser.add_argument('--pklprefix', type=str, default='/local/hopan/irreps/s_8')
    parser.add_argument('--logfile', type=str, default=f'{time.time()}.log')
    args = parser.parse_args()
    main(args)
