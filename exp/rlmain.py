import pdb
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
from rlmodels import MLP, RlPolicy
from utility import nbrs, perm_onehot, ReplayBuffer
from s8puzzle import S8Puzzle
from logger import get_logger

def get_start_states():
    df = pd.read_csv('/home/hopan/github/idastar/s8_dists_red.txt', header=None,
                     dtype={0: str, 1: int}, nrows=24)
    states = [tuple(int(i) for i in row[0]) for _, row in df.iterrows()]
    return states

def get_exp_rate(epoch, explore_epochs, min_exp):
    return max(min_exp, 1 - (epoch / explore_epochs))

def get_reward(done):
    if done:
        return 10
    else:
        return -1

def can_solve(state, policy, max_moves):
    '''
    state: tuple
    policy: nn policy
    max_moves: int
    Returns: True if policy solves(gets to a finished state) within max_moves
    '''
    curr_state = state
    for _ in range(max_moves):
        opt_move = policy.opt_move_tup(curr_state)
        curr_state = S8Puzzle.step(curr_state, opt_move)
        if S8Puzzle.is_done(curr_state):
            return True

    return False

def val_model(policy, max_dist, max_moves, cnt=100):
    '''
    To validate a model need:
    - transition function
    - means to generate states (or pass them in)
    - policy to evluate
    - check if policy landed us in a done state
    '''
    # generate k states by taking a random walk of length d
    # up to size
    nsolves = {}
    for dist in range(1, max_dist):
        d_states = [S8Puzzle.random_state(dist) for _ in range(cnt)]
        solves = 0
        for state in d_states:
            # try to solve the given state within
            solves += can_solve(state, policy, max_moves)
        nsolves[dist] = solves / cnt
    return nsolves

def main(args):
    log = get_logger(args.logfile)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    perms = list(permutations(tuple(i for i in range(1, 9))))
    tens = perm_onehot
    if args.convert == 'onehot':
        tens = perm_onehot
    elif args.convert == 'irrep':
        irreps = [(3, 2, 2, 1)]
        tens = YorConverter(irreps, args.yorprefix, perms)

    if args.model == 'linear':
        policy = RlPolicy(tens(perms[0]).numel(), 1, tens)
    elif args.model == 'mlp':
        policy = MLP(tens(perms[0]).numel(), 32, 1, tens)

    perm_df = PermDF(args.fname, nbrs)
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    start_states = S8Puzzle.start_states()
    done_states = set(start_states)

    val_results = val_model(policy, 5, 10)
    log.info('Validation results: {}'.format(val_results))
    replay = ReplayBuffer(tens(perms[0]).numel(), args.capacity)
    for _ in range(args.capacity):
        st1 = tens(random.choice(start_states))
        st2 = tens(random.choice(start_states))
        replay.push(st1, st2, 10, 1) # this is sort of hacky
    icnt = 0
    wins = 0
    losses = []

    for e in range(args.epochs):
        states = S8Puzzle.random_walk(args.eplen)

        for state in states:
            _nbrs = S8Puzzle.nbrs(state)
            if random.random() < get_exp_rate(e, args.epochs / 4, args.minexp):
                move = S8Puzzle.random_move()
            else:
                nbrs_tens = torch.cat([tens(n) for n in _nbrs], dim=0).float()
                move = policy.opt_move(nbrs_tens)
            next_state = _nbrs[move]

            # reward is pretty flexible though
            done = int(S8Puzzle.is_done(next_state))
            reward = get_reward(done)
            curr_done = int(S8Puzzle.is_done(state))
            curr_rew = get_reward(curr_done)

            # this is sort of a hack
            if curr_done:
                replay.push(tens(state), tens(next_state), curr_rew, curr_done)
            else:
                replay.push(tens(state), tens(next_state), reward, done)

            icnt += 1
            if done:
                wins += 1

            if icnt % args.update == 0 and icnt > 0:
                optim.zero_grad()
                bs, bns, br, bd = replay.sample(args.minibatch)
                # assumption is that the reward will always be -1 for nonfinished state
                loss = F.mse_loss(policy(bs), (1 - bd) * policy(bns) + br)
                loss.backward()
                losses.append(loss.item())
                optim.step()

            if icnt % args.logiters == 0 and icnt > 0:
                wins = 0
                avg_loss = np.mean(losses)
                loss = np.mean(losses[-10])
                log.info(f'Iter {icnt:4d} | last 10 loss: {loss.item():.3f} | ' + \
                         f'avg loss: {avg_loss:.3f} | wins: {wins/args.logiters}')

    log.info('Done training')
    val_results = val_model(policy, 5, 10)
    log.info('Validation results: {}'.format(val_results))

    eval_net = lambda tup: policy(tens(tup))
    policy_score = perm_df.benchmark_policy(perms, eval_net, rev=True)
    log.info('Policy score: {:.4f}'.format(policy_score))

    # debug
    states = S8Puzzle.random_walk(5)
    all_nbrs = [nbrs(s) for s in states]
    for nlst, st in zip(all_nbrs, states):
        nbrs_tens = torch.cat([tens(n) for n in nlst], dim=0)
        nbr_vals = policy.forward(nbrs_tens)
        st_val = policy.forward(tens(st))
        opt_ = policy.opt_move(nbrs_tens)
        nbr_vals = [n.item() for n in nbr_vals]
        log.info('St: {} | Nbr vals: {} | opt: {}'.format(st_val.item(), nbr_vals, opt_))
    pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, default=f'./logs/rl/{time.time()}.log')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--capacity', type=int, default=10000)
    parser.add_argument('--eplen', type=int, default=15)
    parser.add_argument('--minexp', type=int, default=0.05)
    parser.add_argument('--update', type=int, default=100)
    parser.add_argument('--logiters', type=int, default=2000)
    parser.add_argument('--minibatch', type=int, default=128)
    parser.add_argument('--convert', type=str, default='onehot')
    parser.add_argument('--yorprefix', type=str, default='/local/hopan/irreps/s_8/')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--fname', type=str, default='/home/hopan/github/idastar/s8_dists_red.txt')
    args = parser.parse_args()
    main(args)
