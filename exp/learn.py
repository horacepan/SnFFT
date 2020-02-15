import pdb
import os
import sys
import time
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fourier_policy import FourierPolicyTorch, FourierPolicyCG
from perm_df import PermDF, nbrs
from logger import get_logger
sys.path.append('../')
from utils import check_memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: place elsewhere
def get_batch(xs, ys, size):
    idx = np.random.choice(len(xs), size=size)
    return [xs[i] for i in idx], np.array([ys[i] for i in idx]).reshape(-1, 1)

def main(args):
    _st = time.time()
    log = get_logger(f'./{args.logfile}')
    log.info('Starting...')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    irreps = [(3, 2, 2, 1), (4, 2, 2), (4, 2, 1, 1), (3, 3, 1, 1)]

    perm_df = PermDF(args.fname, nbrs)
    train_p, train_y, test_p, test_y = perm_df.train_test(args.testratio)
    all_perms = train_p + test_p # cache these

    log.info('Using irreps: {}'.format(irreps[:args.topk]))
    log.info('Test ratio: {:.4f} | Test items: {}'.format(len(test_p) / len(perm_df.df), len(test_p)))
    if args.mode == 'cg':
        log.info('Using cg policy')
        policy = FourierPolicyCG(irreps[:args.topk], args.pklprefix, args.lr, all_perms)
        optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    else:
        log.info('Using torch policy')
        policy = FourierPolicyTorch(irreps[:args.topk], args.pklprefix, args.lr, all_perms)
        optim = torch.optim.Adam(policy.parameters(), lr=args.lr)

    policy.to(device)
    st = time.time()
    losses = []
    cg_losses = []

    for e in range(args.maxiters + 1):
        optim.zero_grad()
        bx, by = get_batch(train_p, train_y, args.minibatch)
        y_pred = policy.forward_tup(bx)
        by_th = torch.from_numpy(by).float().reshape(y_pred.shape).to(device)
        loss = F.mse_loss(by_th, ypred)
        loss.backward()
        optim.step()

        if args.mode == 'cg' and e % args.cgiters == 0 and e > 0:
            cgst = time.time()
            cg_loss = policy.train_cg_loss_cached(optim)
            train_mse = policy.compute_loss(train_p, train_y)
            cgt = (time.time() - cgst) / 60.
            post_train_mse = policy.compute_loss(train_p, train_y)
            log.info(f'|    Iter {e:5d}: train mse: {train_mse:.2f} | CG loss: {cg_loss:.2f} | cg time: {cgt:.2f}min | post update batch mse: {post_train_mse:.2f}')
            cg_losses.append(cg_loss)

        if e % args.logiters == 0:
            #train_score = perm_df.benchmark_policy(train_p, policy)
            with torch.no_grad():
                train_mse = policy.compute_loss(train_p, train_y)
                test_score = perm_df.benchmark_policy(test_p, policy)

                st = time.time()
                tpred, tnbrs = policy.nbr_deltas(test_p, nbrs)
                nbr_deltas = tnbrs - tpred
                nbr_delta_means = nbr_deltas.abs().mean(dim=1)
                nbr_deltas_std = nbr_deltas.abs().std(dim=1)

                log.info(f'Iter {e:5d}: train mse: {train_mse:.2f} | Policy test score: {test_score:.2f} | ' + \
                         f'Test loss: {loss:.2f} | Test mean: {tpred.mean().item():.2f} | std: {tpred.std().item():.2f} | ' + \
                          'Test Nbr abs diff mean: {:.2f} | std: {:.2f}'.format(nbr_deltas.abs().mean().item(), nbr_deltas.abs().std()))

    if args.fulldebug:
        train_score = perm_df.benchmark_policy(train_p, policy)
        test_score = perm_df.benchmark_policy(test_p, policy)
        train_mse = policy.compute_loss(train_p, train_y)
        test_mse = policy.compute_loss(test_p, test_y)
        log.info('Train Correct rate: {:.4f} | Size: {}'.format(train_score, len(train_p)))
        log.info('Test Correct rate:  {:.4f} | Size: {}'.format(test_score, len(test_p)))
        log.info('Train MSE: {:.4f} | Test MSE: {:.4f}'.format(train_mse, test_mse))
        log.info('Random policy prop correct: {:.4f}'.format(perm_df.benchmark(test_p)))

        tpred, tnbrs = policy.nbr_deltas(test_p, nbrs)
        nbr_deltas = tnbrs - tpred
        nbr_delta_means = nbr_deltas.abs().mean(dim=1)
        nbr_deltas_std = nbr_deltas.abs().std(dim=1)
        log.info('Nbr abs diff mean: {:.2f} | std: {:.2f}'.format(nbr_deltas.abs().mean().item(), nbr_deltas.abs().std()))

    #log.info('Random policy prop correct: {:.4f}'.format(perm_df.benchmark(test_p)))
    log.info(f'Mem footprint: {check_memory()}mb')
    log.info(f'Log saved: {args.logfile}')
    log.info('Total time: {:.2f}mins'.format((time.time() - _st) / 60.))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--testratio', type=float, default=0.05)
    parser.add_argument('--fname', type=str, default='/home/hopan/github/idastar/s8_dists_red.txt')
    parser.add_argument('--pklprefix', type=str, default='/local/hopan/irreps/s_8/')
    parser.add_argument('--fhatprefix', type=str, default='/local/hopan/s8cube/fourier/')
    parser.add_argument('--logfile', type=str, default=f'/logs/{time.time()}.log')
    parser.add_argument('--minibatch', type=int, default=128)
    parser.add_argument('--maxiters', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--cgiters', type=int, default=250)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logiters', type=int, default=1000)
    parser.add_argument('--mode', type=str, default='torch')
    parser.add_argument('--fulldebug', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
