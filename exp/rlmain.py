from GPUtil import showUtilization as gpu_usage
import pdb
import os
import time
import random
from itertools import permutations
import argparse

import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from perm_df import PermDF, WreathDF
from rlmodels import MLP, ResidualBlock, MLPResBlock
from wreath_fourier import WreathPolicy
from fourier_policy import FourierPolicyCG
from utility import nbrs, perm_onehot, ReplayBuffer, update_params, str_val_results, test_model, test_all_states, log_grad_norms, check_memory, wreath_onehot
from logger import get_logger
from tensorboardX import SummaryWriter
import sys
sys.path.append('../')

import pyr_irreps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_exp_rate(epoch, explore_epochs, min_exp):
    return max(min_exp, 1 - (epoch / explore_epochs))

def full_benchmark(policy, perm_df, to_tensor, log):
    score, dists, stats = test_all_states(policy, 20, perm_df, to_tensor)
    log.info(f'Full Prop solves: {score:.4f} | stats: {str_val_results(stats)} | dists: {dists}')
    return {'score': score, 'dists': dists, 'stats': stats}

def main(args):
    log = get_logger(args.logfile, stdout=not args.nostdout, tofile=args.savelog)
    sumdir = os.path.join(f'./logs/{args.sumdir}/{args.notes}/seed_{args.seed}')
    if not os.path.exists(sumdir) and args.savelog:
        os.makedirs(sumdir)

    if args.savelog:
        swr = SummaryWriter(sumdir)

    log.info(f'Starting ... Saving logs in: {args.logfile} | summary writer: {sumdir}')
    log.info('Args: {}'.format(args))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    seen_states = set()
    perms = list(permutations(tuple(i for i in range(1, 9))))
    if args.env == 's8':
        ident = tuple(range(1, 9))
        ident = perms[0]
    elif args.env == 'pyraminx':
        ident = ((0, 0, 0,0, 0, 0), tuple(range(1, 7)))
    else:
        raise Exception(f'{args.env} is not a supported env!')

    if args.env == 's8':
        perm_df = PermDF(args.fname, 6)
        irreps = eval(args.irreps)
    elif args.env == 'pyraminx':
        perm_df = WreathDF(args.fname, 8, cyc_size=2)
        irreps = pyr_irreps.get_topk_irreps(args.num_pyr_irreps)

    if args.convert == 'onehot' and args.env == 's8':
        to_tensor = perm_onehot
    elif args.convert == 'onehot' and args.env == 'pyraminx':
        to_tensor = lambda g: wreath_onehot(g, 2)
    elif args.convert == 'irrep':
        to_tensor = None
    else:
        raise Exception('Must pass in convert string')

    if args.model == 'linear':
        log.info(f'Policy using Irreps: {irreps}')
        if args.env == 's8':
            policy = FourierPolicyCG(irreps, args.yorprefix, perms)
            target = FourierPolicyCG(irreps, args.yorprefix, perms, rep_dict=policy.rep_dict, pdict=policy.pdict)
        else:
            policy = WreathPolicy(irreps, args.pyrprefix)
            target = WreathPolicy(irreps, args.pyrprefix, rep_dict=policy.rep_dict, pdict=policy.pdict)
        to_tensor = lambda g: policy.to_tensor(g)
    elif args.model == 'dvn':
        log.info('Using MLP DVN')
        policy = MLP(to_tensor([ident]).numel(), args.nhid, 1, layers=args.layers, to_tensor=to_tensor, std=args.std)
        target = MLP(to_tensor([ident]).numel(), args.nhid, 1, layers=args.layers, to_tensor=to_tensor, std=args.std)
    elif args.model == 'res':
        log.info('Using MLP DVN')
        policy = ResidualBlock(to_tensor([ident]).numel(), args.nhid, nout=1, to_tensor=to_tensor, std=args.std)
        target = ResidualBlock(to_tensor([ident]).numel(), args.nhid, nout=1, to_tensor=to_tensor, std=args.std)
    elif args.model == 'mlp_res':

        log.info('Using MLPRes')
        policy = MLPResBlock(to_tensor([ident]).numel(), args.nhid, 1, to_tensor=to_tensor, std=args.std)
        target = MLPResBlock(to_tensor([ident]).numel(), args.nhid, 1, to_tensor=to_tensor, std=args.std)
    elif args.model == 'dqn':
        log.info('Using MLP DQN')
        nactions = 6
        policy = MLP(to_tensor([ident]).numel(), args.nhid, nactions, layers=args.layers, to_tensor=to_tensor, std=args.std)
        target = MLP(to_tensor([ident]).numel(), args.nhid, nactions, layers=args.layers, to_tensor=to_tensor, std=args.std)

    policy.to(device)
    target.to(device)
    if not args.skipvalidate:
        baseline_corr, corr_dict = perm_df.benchmark()
        log.info('Baseline correct: {}'.format(baseline_corr))
        log.info(str_val_results(corr_dict))
        res, distr, distr_stats = test_model(policy, 1000, 1000, 20, perm_df, to_tensor)

    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    replay = ReplayBuffer(to_tensor([ident]).numel(), args.capacity)
    max_benchmark = 0
    icnt = 0
    updates = 0
    bps = 0
    nactions = perm_df.num_nbrs
    dist_vals = {i: [] for i in range(0, 10)}
    update_pairs = {}
    exp_moves = 0
    pol_moves = 0
    pushes = {}
    npushes = 0

    save_epochs = []
    save_prop_corr = []
    save_solve_corr = []
    save_seen = []
    save_updates = []
    check_memory()

    for e in range(args.epochs + 1):
        states = perm_df.random_walk(args.eplen)
        #for state in states:
        for idx, state in enumerate(states):
            _nbrs = perm_df.nbrs(state)
            if random.random() < get_exp_rate(e, args.epochs * args.exp_prop, args.minexp):
                exp_moves += 1
                move = np.random.choice(nactions) # TODO
            elif hasattr(policy, 'nout') and policy.nout == 1:
                nbrs_tens = to_tensor(_nbrs).to(device)
                move = policy.forward(nbrs_tens).argmax().item()
                pol_moves += 1
            elif hasattr(policy, 'nout') and policy.nout > 1:
                move = policy.forward(to_tensor([state])).argmax().item()
            next_state = _nbrs[move]

            done = 1 if (perm_df.is_done(state)) else 0
            reward = 1 if done else -1
            #replay.push(to_tensor([state]), move, to_tensor([next_state]), reward, done, state, next_state)
            if done:
                replay.push(to_tensor([next_state]), 0, to_tensor([state]), -1, 0, next_state, state, idx + 1)
            replay.push(to_tensor([state]), move, to_tensor([next_state]), reward, done, state, next_state, idx+1)
            d1 = perm_df.distance(state)
            d2 = perm_df.distance(next_state)
            pushes[(d1, d2, done)] = pushes.get((d1, d2, done), 0) + 1
            npushes += 1

            icnt += 1
            if icnt % args.update == 0 and icnt > 0:
                optim.zero_grad()
                bs, ba, bns, br, bd, bs_tups, bns_tups, bidx = replay.sample(args.minibatch, device)
                seen_states.update(bs_tups)
                #for b1, b2 in zip(bs_tups, bns_tups):
                if args.model == 'linear':
                    bs_nbrs = [n for tup in bs_tups for n in perm_df.nbrs(tup)]
                    bs_nbrs_tens = to_tensor(bs_nbrs)
                    opt_nbr_vals, _ = target.forward(bs_nbrs_tens).detach().reshape(-1, nactions).max(dim=1, keepdim=True)
                    for index, nbr_tup in enumerate(bns_tups):
                        d1 = perm_df.distance(bs_tups[index])
                        d2 = perm_df.distance(nbr_tup)
                        don = bd[index].item()
                        update_pairs[d1, d2, don]  = update_pairs.get((d1, d2, don), 0) + 1

                    if args.use_done:
                        loss = F.mse_loss(policy.forward(bs),
                                          args.discount * (1 - bd) * opt_nbr_vals + br)
                    else:
                        loss = F.mse_loss(policy.forward(bs),
                                          args.discount * opt_nbr_vals + br)
                elif args.model == 'dvn' or args.model == 'mlp_res':
                    bs_nbrs_tens = to_tensor([n for tup in bs_tups for n in perm_df.nbrs(tup)])
                    opt_nbr_vals, _ = target.forward(bs_nbrs_tens).reshape(-1, nactions).max(dim=1, keepdim=True)
                    loss = F.mse_loss(policy.forward(bs),
                                      args.discount * (1 - bd) * opt_nbr_vals.detach() + br)
                elif args.model ==  'dqn':
                    qs = policy.forward(bs)
                    qsa = qs.gather(1, ba.long())

                    nxt_vals, nxt_actions = target.forward(bns).detach().max(dim=1, keepdim=True)
                    qsan = target.forward(bns).gather(1, nxt_actions).detach()
                    loss = F.mse_loss(qsa, args.discount * (1 - bd) * qsan + br)

                loss.backward()
                if args.lognorms and bps % args.normiters == 0:
                    log_grad_norms(swr, policy, e)
                optim.step()
                bps += 1
                if args.savelog:
                    swr.add_scalar('loss', loss.item(), bps)

        #if e % 1000 == 0:
        #    vals = {}
        #    for ii in range(0, 9):
        #        ii_val = policy.forward(to_tensor(perm_df.random_states(ii, 100))).mean().item()
        #        vals[ii] = ii_val
        #        #dist_vals[ii].append(ii_val)
        #    log.info('      Dist vals: {}'.format(str_val_results(vals)))

        if e % args.targetupdate == 0 and e > 0:
            update_params(target, policy)
            updates += 1

        if e % args.logiters == 0:
            exp_rate = get_exp_rate(e, args.epochs / 2, args.minexp)
            benchmark, val_results = perm_df.prop_corr_by_dist(policy, to_tensor)
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

                val_corr, distr, distr_stats = test_model(policy, 1000, 1000, 20, perm_df, to_tensor)
                save_epochs.append(e)
                save_prop_corr.append(benchmark)
                save_solve_corr.append(val_corr)
                save_seen.append(len(seen_states))
                save_updates.append(updates)


            log.info(f'Epoch {e:5d} | exp rate: {exp_rate:.2f} | val: {str_dict} | ' + \
                     f'Dist corr: {benchmark:.4f} | Updates: {updates}, bps: {bps} | seen: {len(seen_states)} | icnt: {icnt}')
        if e % args.benchlog == 0 and e > 0:
            bench_results = full_benchmark(policy, perm_df, to_tensor, log)

    log.info('Max benchmark prop corr move attained: {:.4f}'.format(max_benchmark))
    log.info(f'Done training | log saved in: {args.logfile}')
    bench_results = full_benchmark(policy, perm_df, to_tensor, log)
    _prop_corr, _ = perm_df.prop_corr_by_dist(policy, to_tensor)
    bench_results['prop_correct'] = _prop_corr
    stats_dict = {'epochs': save_epochs,
                  'updates': save_updates,
                  'prop_corr': save_prop_corr,
                  'solve_corr': save_solve_corr,
                  'seen': save_seen,
                  'updates': save_updates}
    bench_results.update(stats_dict)

    if not args.skipvalidate and args.savelog:
        json.dump(bench_results, open(os.path.join(sumdir, 'stats.json'), 'w'))
        json.dump(args.__dict__, open(os.path.join(sumdir, 'args.json'), 'w'))
    print(f'Done with: {irreps}')
    return bench_results

def get_args():
    _prefix = 'local' if os.path.exists('/local/hopan/irreps') else 'scratch'
    parser = argparse.ArgumentParser()
    # log params
    parser.add_argument('--nostdout', action='store_true', default=False)
    parser.add_argument('--sumdir', type=str, default='test')
    parser.add_argument('--savelog', action='store_true', default=False)
    parser.add_argument('--logfile', type=str, default=f'./logs/rl/{time.time()}.log')
    parser.add_argument('--skipvalidate', action='store_true', default=False)
    parser.add_argument('--benchlog', type=int, default=50000)
    parser.add_argument('--lognorms', action='store_true', default=False)
    parser.add_argument('--normiters', type=int, default=10)
    parser.add_argument('--logiters', type=int, default=1000)

    # file related params
    parser.add_argument('--fname', type=str, default='/home/hopan/github/idastar/s8_dists_red.txt')
    parser.add_argument('--yorprefix', type=str, default=f'/{_prefix}/hopan/irreps/s_8/')
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--env', type=str, default='s8')
    parser.add_argument('--pyrprefix', type=str, default=f'/{_prefix}/hopan/pyraminx/irreps/')

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
    parser.add_argument('--minexp', type=float, default=1.0)
    parser.add_argument('--update', type=int, default=50)
    parser.add_argument('--minibatch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--discount', type=float, default=1)
    parser.add_argument('--targetupdate', type=int, default=100)

    parser.add_argument('--use_done', action='store_true', default=False)
    parser.add_argument('--exp_prop', type=float, default=0.4)

    # env specific
    parser.add_argument('--num_pyr_irreps', type=int, default=1)
    parser.add_argument('--ncpu', type=int, default=2)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)
