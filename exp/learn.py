import pdb
import os
import sys
import time
import argparse
from tqdm import tqdm
import numpy as np

from fourier_policy import FourierPolicy, FourierPolicyTorch, FourierPolicyCG
from perm_df import PermDF, nbrs
from logger import get_logger
sys.path.append('../')
from utils import check_memory
from utility import S8_GENERATORS

# TODO: place elsewhere
def get_batch(xs, ys, size):
    idx = np.random.choice(len(xs), size=size)
    return [xs[i] for i in idx], np.array([ys[i] for i in idx]).reshape(-1, 1)

def main(args):
    log = get_logger(f'./{args.logfile}')
    irreps = [(3, 2, 2, 1), (4, 2, 2), (4, 2, 1, 1), (3, 3, 1, 1)]

    perm_df = PermDF(args.fname, nbrs)
    log.info('Using irreps: {}'.format(irreps[:args.topk]))

    if args.splitmode == 'train_test':
        log.info('Using split func: train_test')
        train_p, train_y, test_p, test_y = perm_df.train_test(args.testratio)
    else:
        train_p, train_y, test_p, test_y = perm_df.train_test_split(args.testratio)

    log.info('Test ratio: {:.4f} | Test items: {}'.format(len(test_p) / len(perm_df.df), len(test_p)))
    if args.mode == 'numpy':
        log.info('Using numpy policy')
        policy = FourierPolicy(irreps[:args.topk], args.pklprefix)
    elif args.mode == 'cg':
        log.info('Using cg torch policy')
        policy = FourierPolicyCG(irreps[:args.topk], args.pklprefix, lr=args.lr)
    else:
        log.info('Using torch policy')
        policy = FourierPolicyTorch(irreps[:args.topk], args.pklprefix, lr=args.lr)

    losses = []
    cg_losses = []
    for e in range(args.maxiters + 1):
        bx, by = get_batch(train_p, train_y, args.minibatch)
        loss = policy.train_batch(bx, by, lr=args.lr, epoch=e)
        losses.append(loss)

        if args.mode == 'cg' and e % args.cgiters == 0 and e > 0:
            cg_loss = policy.train_cg_loss(S8_GENERATORS)
            train_score = perm_df.benchmark_policy(train_p, policy)
            log.info(f'|    Train iter {e:3d}: batch mse: {loss:.4f} | Policy train score: {train_score:.4f} | CG loss: {cg_loss:.5f}')
            cg_losses.append(cg_loss)

        if e % args.logiters == 0:
            train_score = perm_df.benchmark_policy(train_p, policy)
            log.info(f'Train iter {e:3d}: batch mse: {loss:.4f} | Policy test score: {train_score:.4f} | last loss: {loss:.5f}')


    log.info('Done training')
    train_score = perm_df.benchmark_policy(train_p, policy)
    test_score = perm_df.benchmark_policy(test_p, policy)
    log.info('Train Correct rate: {:.4f} | Size: {}'.format(train_score, len(train_p)))
    log.info('Test Correct rate:  {:.4f} | Size: {}'.format(test_score, len(test_p)))
    log.info('Random policy prop correct: {:.4f}'.format(perm_df.benchmark(test_p)))
    log.info(f'Mem footprint: {check_memory()}mb')
    log.info(f'Log saved: {args.logfile}')
    # inspect policy's w[-1]
    pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--testratio', type=float, default=0.05)
    parser.add_argument('--fname', type=str, default='/home/hopan/github/idastar/s8_dists_red.txt')
    parser.add_argument('--pklprefix', type=str, default='/local/hopan/irreps/s_8')
    parser.add_argument('--fhatprefix', type=str, default='/local/hopan/s8cube/fourier/')
    parser.add_argument('--logfile', type=str, default=f'/logs/{time.time()}.log')
    parser.add_argument('--minibatch', type=int, default=128)
    parser.add_argument('--maxiters', type=int, default=1000)
    parser.add_argument('--cgiters', type=int, default=250)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logiters', type=int, default=100)
    parser.add_argument('--mode', type=str, default='numpy')
    parser.add_argument('--splitmode', type=str, default='train_test_split')

    args = parser.parse_args()
    main(args)
