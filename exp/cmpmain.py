from GPUtil import showUtilization as gpu_usage
import pdb
import os
import time
import random
from itertools import permutations
import argparse

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from perm_df import PermDF
from yor_dataset import YorConverter
from rlmodels import MLP, LinearPolicy
from fourier_policy import FourierPolicyCG
from utility import nbrs, perm_onehot, ReplayBuffer, update_params, str_val_results, val_model, can_solve
from s8puzzle import S8Puzzle
import group_puzzle
from logger import get_logger
from tensorboardX import SummaryWriter
import sys
sys.path.append('../')
from utils import check_memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_exp_rate(epoch, explore_epochs, min_exp):
    return max(min_exp, 1 - (epoch / explore_epochs))

def get_reward(done):
    if done:
        return 0
    else:
        return -1

def main(args):
    log = get_logger(args.logfile)
    sumdir = os.path.join(f'./logs/test/{args.notes}')
    if not os.path.exists(sumdir):
        os.makedirs(sumdir)
    swr = SummaryWriter(sumdir) # this should maybe

    log.info(f'Starting ... Saving logs in: {args.logfile}')
    log.info('Args: {}'.format(args))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    target = None
    seen_states = set()
    irreps = [(3, 2, 2, 1), (4, 2, 2), (4, 2, 1, 1)][:args.topk]
    if args.irreps:
        try:
            irreps = eval(args.irreps)
        except:
            log.info('Invalid input for args.irreps: {}'.format(args.irreps))
            exit()

    env = S8Puzzle()
    to_tensor = None
    vec_size = env.size # this isnt an ideal interface

    if args.model == 'linear':
        log.info(f'Policy using Irreps: {irreps}')
        policy = LinearPolicy(vec_size, 1)
        if args.doubleq:
            target = LinearPolicy(vec_size, 1)
            target.to(device)
    elif args.model == 'dvn':
        log.info('Using MLP DVN')
        policy = MLP(to_tensor(vec_size, args.nhid, 1, layers=args.layers, to_tensor=to_tensor, std=args.std)
        target = MLP(to_tensor(vec_size, args.nhid, 1, layers=args.layers, to_tensor=to_tensor, std=args.std)
        target.to(device)
    elif args.model == 'dqn':
        log.info('Using MLP DQN')
        nout = env.num_nbrs()
        policy = MLP(vec_size, args.nhid, nout, layers=args.layers, to_tensor=to_tensor, std=args.std)
        target = MLP(vec_size, args.nhid, nout, layers=args.layers, to_tensor=to_tensor, std=args.std)
        target.to(device)

    policy.to(device)
    perm_df = PermDF(args.fname, nbrs)
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)

    log.info('Memory used pre replay creation: {:.2f}mb'.format(check_memory(False)))
    replay = ReplayBuffer(vec_size, args.capacity)
    log.info('Memory used post replay creation: {:.2f}mb'.format(check_memory(False)))

    max_benchmark = 0
    icnt = 0
    updates = 0
    bps = 0
    losses = []

    for e in range(args.epochs + 1):
        states = env.random_walk(args.eplen)
        for state in states:
            _nbrs = env.nbrs(state)
            if random.random() < get_exp_rate(e, args.epochs / 2, args.minexp):
                move = env.random_move(state)
            elif hasattr(policy, 'nout') and policy.nout == 1:
                nbrs_tens = to_tensor(_nbrs).to(device)
                move = policy.forward(nbrs_tens).argmax().item()
            elif hasattr(policy, 'nout') and policy.nout > 1:
                move = policy.forward(to_tensor([state])).argmax().item()
            next_state = _nbrs[move]

            # reward is pretty flexible though
            done = int(env.is_done(next_state))
            reward = get_reward(done)
            replay.push(to_tensor([state]), move, to_tensor([next_state]), reward, done, state, next_state)

            icnt += 1
            if icnt % args.update == 0 and icnt > 0:
                optim.zero_grad()
                bs, ba, bns, br, bd, bs_tups, bns_tups = replay.sample(args.minibatch, device)
                bs_nbrs = [n for tup in bs_tups for n in env.nbrs(tup)]
                bs_nbr_tensor = to_tensor(bs_nbrs)

                # update actually should not be different in the linear/nonlinear case...
                # only diff if we use doubleq/dont use doubleq!
                if args.doubleq:
                    all_nbr_vals = policy.forward(bs_nbr_tensor).reshape(-1, policy.nout)
                    opt_nbr_idx = all_nbr_vals.max(dim=1, keepdim=True)[1]
                    opt_nbr_vals = target.forward(bs_nbr_tensor).reshape(-1, target.nout).gather(1, opt_nbr_idx).detach()
                else:
                    opt_nbr_vals = policy.eval_opt_nbr(bs_nbr_tensor, env.num_nbrs()).detach()
                loss = F.mse_loss(policy.forward(bs),
                                  args.discount * (1 - bd) * opt_nbr_vals + br)
                loss.backward()
                losses.append(loss.item())
                optim.step()
                bps += 1
                swr.add_scalar('loss', loss.item(), bps)

        seen_states.update(states)
        if e % args.qqupdate == 0 and e > 0 and target:
            update_params(target, policy)
            updates += 1

        if e % args.logiters == 0:
            exp_rate = get_exp_rate(e, args.epochs / 2, args.minexp)
            benchmark, val_results = perm_df.prop_corr_by_dist(policy, False)
            max_benchmark = max(max_benchmark, benchmark)
            str_dict = str_val_results(val_results)
            swr.add_scalar('prop_correct/overall', benchmark, e)

            for ii in range(1, 9):
                # sample some number of states that far away, evaluate them, report mean + std
                rand_states = perm_df.random_state(ii, 100)
                rand_tensor = to_tensor(rand_states)
                vals = policy.forward(rand_tensor)
                swr.add_scalar(f'values/median/states_{ii}', vals.median().item(), e)
                swr.add_scalar(f'values/mean/states_{ii}', vals.mean().item(), e)
                swr.add_scalar(f'values/std/states_{ii}', vals.std().item(), e)
                swr.add_scalar(f'prop_correct/dist_{ii}', val_results[ii], e)

            log.info(f'Epoch {e:5d} | Last {args.logiters} loss: {np.mean(losses[-args.logiters:]):.3f} | ' + \
                     f'exp rate: {exp_rate:.2f} | val: {str_dict} | Dist corr: {benchmark:.4f} | Updates: {updates}, bps: {bps} | seen: {len(seen_states)} | icnt: {icnt}')

    log.info('Max benchmark prop corr move attained: {:.4f}'.format(max_benchmark))
    log.info(f'Done training | log saved in: {args.logfile}')
    sp_results = val_model(policy, args.valmaxdist, perm_df, env)
    benchmark, val_results = perm_df.prop_corr_by_dist(policy, False)
    str_dict = str_val_results(val_results)
    log.info('Prop correct moves: {:.3f} | Prop correct by distance: {}'.format(benchmark, str_dict))
    log.info('Shortest path results: {}'.format(sp_results))
    pdb.set_trace()

if __name__ == '__main__':
    _prefix = 'local' if os.path.exists('/local/hopan/irreps') else 'scratch'
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, default=f'./logs/rl/{time.time()}.log')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--capacity', type=int, default=10000)
    parser.add_argument('--eplen', type=int, default=15)
    parser.add_argument('--minexp', type=float, default=0.05)
    parser.add_argument('--update', type=int, default=100)
    parser.add_argument('--logiters', type=int, default=1000)
    parser.add_argument('--minibatch', type=int, default=128)
    parser.add_argument('--convert', type=str, default='irrep')
    parser.add_argument('--yorprefix', type=str, default=f'/{_prefix}/hopan/irreps/s_8/')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--discount', type=float, default=1)
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--valmaxdist', type=int, default=8)
    parser.add_argument('--fname', type=str, default='/home/hopan/github/idastar/s8_dists_red.txt')
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--doubleq', action='store_true', default=False)
    parser.add_argument('--qqupdate', type=int, default=100)
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument('--irreps', type=str, default='')
    parser.add_argument('--nhid', type=int, default=32)
    parser.add_argument('--puzzle', type=str, default='S8Puzzle')
    args = parser.parse_args()
    main(args)
