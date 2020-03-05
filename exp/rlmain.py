from GPUtil import showUtilization as gpu_usage
import pdb
import os
import time
import random
from itertools import permutations
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from perm_df import PermDF
from rlmodels import MLP
from fourier_policy import FourierPolicyCG
from utility import nbrs, perm_onehot, ReplayBuffer, update_params, str_val_results, test_model, test_all_states, log_grad_norms
from logger import get_logger
from tensorboardX import SummaryWriter
import sys
sys.path.append('../')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_exp_rate(epoch, explore_epochs, min_exp):
    #return 1
    return max(min_exp, 1 - (epoch / explore_epochs))

def full_benchmark(policy, perm_df, env, log):
    score, dists, stats = test_all_states(policy, 20, perm_df, env)
    log.info(f'Full Prop solves: {score:.4f} | stats: {str_val_results(stats)} | dists: {dists}')
    return {'score': score, 'dists': dists, 'stats': stats}

def main(args):
    log = get_logger(args.logfile, stdout=args.stdout, tofile=args.savelog)
    sumdir = os.path.join(f'./logs/nb_summary2/{args.notes}')
    if not os.path.exists(sumdir):
        os.makedirs(sumdir)
    else:
        sumdir += ('_' + str(random.random())[2:4])

    if args.savelog:
        swr = SummaryWriter(sumdir)

    log.info(f'Starting ... Saving logs in: {args.logfile} | summary writer: {sumdir}')
    log.info('Args: {}'.format(args))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    target = None
    seen_states = set()
    perms = list(permutations(tuple(i for i in range(1, 9))))

    perm_df = PermDF(args.fname, 6)
    env = perm_df
    irreps = eval(args.irreps)

    if args.convert == 'onehot':
        to_tensor = perm_onehot
    elif args.convert == 'irrep':
        to_tensor = None
    else:
        raise Exception('Must pass in convert string')

    if args.model == 'linear':
        log.info(f'Policy using Irreps: {irreps}')
        policy = FourierPolicyCG(irreps, args.yorprefix, perms)
        target = FourierPolicyCG(irreps, args.yorprefix, perms, rep_dict=policy.rep_dict, pdict=policy.pdict)
        to_tensor = lambda g: policy.to_tensor(g)
    elif args.model == 'dvn':
        log.info('Using MLP DVN')
        policy = MLP(to_tensor([perms[0]]).numel(), args.nhid, 1, layers=args.layers, to_tensor=to_tensor, std=args.std)
        target = MLP(to_tensor([perms[0]]).numel(), args.nhid, 1, layers=args.layers, to_tensor=to_tensor, std=args.std)
    elif args.model == 'dqn':
        log.info('Using MLP DQN')
        nactions = 6
        policy = MLP(to_tensor([perms[0]]).numel(), args.nhid, nactions, layers=args.layers, to_tensor=to_tensor, std=args.std)
        target = MLP(to_tensor([perms[0]]).numel(), args.nhid, nactions, layers=args.layers, to_tensor=to_tensor, std=args.std)

    policy.to(device)
    target.to(device)
    if not args.skipvalidate:
        baseline_corr, corr_dict = perm_df.benchmark()
        log.info('Baseline correct: {}'.format(baseline_corr))
        log.info(str_val_results(corr_dict))
        res, distr, distr_stats = test_model(policy, 1000, 1000, 20, perm_df, env)

    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    replay = ReplayBuffer(to_tensor([perms[0]]).numel(), args.capacity)

    max_benchmark = 0
    icnt = 0
    updates = 0
    bps = 0
    nactions = 6
    dist_vals = {i: [] for i in range(0, 10)}

    for e in range(args.epochs + 1):
        states = env.random_walk(args.eplen)
        for state in states:
            _nbrs = env.nbrs(state)
            if random.random() < get_exp_rate(e, args.epochs / 2, args.minexp):
                move = np.random.choice(6) # TODO
            elif hasattr(policy, 'nout') and policy.nout == 1:
                nbrs_tens = to_tensor(_nbrs).to(device)
                move = policy.forward(nbrs_tens).argmax().item()
            elif hasattr(policy, 'nout') and policy.nout > 1:
                move = policy.forward(to_tensor([state])).argmax().item()
            next_state = _nbrs[move]

            done = 1 if (env.is_done(state)) else 0
            reward = 1 if done else -1
            replay.push(to_tensor([state]), move, to_tensor([next_state]), reward, done, state, next_state)

            icnt += 1
            if icnt % args.update == 0 and icnt > 0:
                optim.zero_grad()
                bs, ba, bns, br, bd, bs_tups, bns_tups = replay.sample(args.minibatch, device)
                if args.model == 'linear' or args.model == 'dvn':
                    bs_nbrs = [n for tup in bs_tups for n in env.nbrs(tup)]
                    bs_nbrs_tens = to_tensor(bs_nbrs)
                    if args.doubleq:
                        all_nbr_vals = policy.forward(bs_nbrs_tens).reshape(-1, nactions)
                        _, opt_nbr_idx = all_nbr_vals.max(dim=1, keepdim=True)
                        opt_nbr_vals = target.forward(bs_nbrs_tens).reshape(-1, nactions).gather(1, opt_nbr_idx).detach()
                    else:
                        nbr_vals = policy.forward(bs_nbrs_tens).detach().reshape(-1, nnbrs)
                        opt_nbr_vals, idx = nbr_vals.max(dim=1, keepdim=True)

                    loss = F.mse_loss(policy.forward(bs),
                                      args.discount * opt_nbr_vals + br)
                elif args.model ==  'dqn':
                    qs = policy.forward(bs)
                    qsa = qs.gather(1, ba.long())

                    # if double q, get the next value from target using online's actions
                    if args.doubleq:
                        nxt_vals, nxt_actions = policy.forward(bns).detach().max(dim=1, keepdim=True)
                        qsan = target.forward(bns).gather(1, nxt_actions).detach()
                    else:
                        nxt_vals, nxt_actions = target.forward(bns).detach().max(dim=1, keepdim=True)
                        qsan = policy.forward(bns).gather(1, nxt_actions).detach()
                    loss = F.mse_loss(qsa, args.discount * qsan + br)

                loss.backward()
                if args.lognorms and bps % args.normiters == 0:
                    log_grad_norms(swr, policy, e)
                optim.step()
                bps += 1
                if args.savelog:
                    swr.add_scalar('loss', loss.item(), bps)

        #if e % 200 == 0:
        #    for ii in range(0, 9):
        #        ii_val = policy.forward_tup(perm_df.random_states(ii, 100)).mean().item()
        #        dist_vals[ii].append(ii_val)

        seen_states.update(states)
        if e % args.targetupdate == 0 and e > 0:
            update_params(target, policy)
            updates += 1

        if e % args.logiters == 0:
            exp_rate = get_exp_rate(e, args.epochs / 2, args.minexp)
            benchmark, val_results = perm_df.prop_corr_by_dist(policy)
            max_benchmark = max(max_benchmark, benchmark)
            str_dict = str_val_results(val_results)
            if args.savelog:
                swr.add_scalar('prop_correct/overall', benchmark, e)
                for ii in range(1, 9):
                    # sample some number of states that far away, evaluate them, report mean + std
                    rand_states = perm_df.random_states(ii, 100)
                    rand_tensors = to_tensor(rand_states)
                    vals = policy.forward(rand_tensors)
                    swr.add_scalar(f'values_median/states_{ii}', vals.median().item(), e)
                    swr.add_scalar(f'values_mean/states_{ii}', vals.mean().item(), e)
                    swr.add_scalar(f'values_std/states_{ii}', vals.std().item(), e)
                    swr.add_scalar(f'prop_correct/dist_{ii}', val_results[ii], e)

            log.info(f'Epoch {e:5d} | exp rate: {exp_rate:.2f} | val: {str_dict} | ' + \
                     f'Dist corr: {benchmark:.4f} | Updates: {updates}, bps: {bps} | seen: {len(seen_states)} | icnt: {icnt}')
        if e % args.benchlog == 0 and e > 0:
            bench_results = full_benchmark(policy, perm_df, env, log)

    log.info('Max benchmark prop corr move attained: {:.4f}'.format(max_benchmark))
    log.info(f'Done training | log saved in: {args.logfile}')
    if not args.skipvalidate:
        bench_results = full_benchmark(policy, perm_df, env, log)
        return bench_results

if __name__ == '__main__':
    _prefix = 'local' if os.path.exists('/local/hopan/irreps') else 'scratch'
    parser = argparse.ArgumentParser()
    # log params
    parser.add_argument('--stdout', action='store_true', default=True)
    parser.add_argument('--savelog', action='store_true', default=False)
    parser.add_argument('--logfile', type=str, default=f'./logs/rl/{time.time()}.log')
    parser.add_argument('--skipvalidate', action='store_true', default=False)
    parser.add_argument('--benchlog', type=int, default=5000)
    parser.add_argument('--lognorms', action='store_true', default=False)
    parser.add_argument('--normiters', type=int, default=10)
    parser.add_argument('--logiters', type=int, default=1000)

    # file related params
    parser.add_argument('--fname', type=str, default='/home/hopan/github/idastar/s8_dists_red.txt')
    parser.add_argument('--fhatdir', type=str, default='/local/hopan/s8cube/fourier/')
    parser.add_argument('--yorprefix', type=str, default=f'/{_prefix}/hopan/irreps/s_8/')
    parser.add_argument('--loadfhats', action='store_true', default=False)
    parser.add_argument('--notes', type=str, default='')

    # model params
    parser.add_argument('--convert', type=str, default='irrep')
    parser.add_argument('--irreps', type=str, default='[]')
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--nhid', type=int, default=32)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--std', type=float, default=0.1)

    # hparams
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--capacity', type=int, default=10000)
    parser.add_argument('--eplen', type=int, default=15)
    parser.add_argument('--minexp', type=float, default=0.05)
    parser.add_argument('--update', type=int, default=50)
    parser.add_argument('--minibatch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--discount', type=float, default=1)
    parser.add_argument('--doubleq', action='store_true', default=False)
    parser.add_argument('--targetupdate', type=int, default=100)
    args = parser.parse_args()
    main(args)
