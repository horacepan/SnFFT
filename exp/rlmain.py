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
from rlmodels import MLP, RlPolicy, MLPMini
from fourier_policy import FourierPolicyCG
from utility import nbrs, perm_onehot, ReplayBuffer
from s8puzzle import S8Puzzle
from logger import get_logger
import sys
sys.path.append('../')
from utils import check_memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        neighbors = S8Puzzle.nbrs(curr_state)
        opt_move = policy.opt_move_tup(neighbors)
        curr_state = S8Puzzle.step(curr_state, opt_move)
        if S8Puzzle.is_done(curr_state):
            return True

    return False

def val_model(policy, max_dist, perm_df, cnt=100):
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
    for dist in range(1, max_dist + 1):
        d_states = perm_df.random_state(dist, cnt)
        solves = 0
        for state in d_states:
            solves += can_solve(state, policy, dist + 1)
        nsolves[dist] = solves / cnt
    return nsolves

def main(args):
    log = get_logger(args.logfile)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    perms = list(permutations(tuple(i for i in range(1, 9))))
    irreps = [(3, 2, 2, 1), (4, 2, 2), (4, 2, 1, 1)]

    if args.convert == 'onehot':
        to_tensor = perm_onehot
    elif args.convert == 'irrep':
        #to_tensor = YorConverter(irreps[:args.topk], args.yorprefix, perms)
        to_tensor = None
    else:
        raise Exception('Must pass in convert string')

    if args.model == 'linear':
        log.info('Using linear policy on irreps with cg iterations: {}'.format(args.docg))
        policy = FourierPolicyCG(irreps[:args.topk], args.yorprefix, args.lr, perms)
        to_tensor = lambda g: policy.to_tensor(g)
    elif args.model == 'mlp':
        log.info('Using MLP')
        policy = MLP(to_tensor([perms[0]]).numel(), 32, 1, to_tensor)
    elif args.model == 'mini':
        log.info('Using Mini MLP')
        policy = MLPMini(to_tensor([perms[0]]).numel(), 32, 1, to_tensor)

    policy.to(device)
    perm_df = PermDF(args.fname, nbrs)
    if hasattr(policy, 'optim'):
        optim = policy.optim
    else:
        optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    start_states = S8Puzzle.start_states()

    replay = ReplayBuffer(to_tensor([perms[0]]).numel(), args.capacity)
    for i in range(args.capacity):
        st1 = to_tensor([random.choice(start_states)])
        st2 = to_tensor([random.choice(start_states)])
        replay.push(st1, get_reward(True), 10, 1) # this is sort of hacky

    log.info('Checking true policy...')
    log.info('Sanity check | true sp dist policy: {}'.format(val_model(perm_df, args.valmaxdist, perm_df)))
    val_results = val_model(policy, args.valmaxdist, perm_df)
    log.info('Before training val results: {}'.format(val_results))
    icnt = 0
    losses = []
    cglosses = []
    for e in range(args.epochs + 1):
        states = S8Puzzle.random_walk(args.eplen)
        for state in states:
            _nbrs = S8Puzzle.nbrs(state)
            if random.random() < get_exp_rate(e, args.epochs / 2, args.minexp):
                move = S8Puzzle.random_move()
            else:
                nbrs_tens = to_tensor(_nbrs).to(device)
                move = policy.forward(nbrs_tens).argmax().item()
            next_state = _nbrs[move]

            # reward is pretty flexible though
            done = int(S8Puzzle.is_done(next_state))
            reward = get_reward(done)
            curr_done = int(S8Puzzle.is_done(state))
            curr_rew = get_reward(curr_done)

            # this is sort of a hack
            if curr_done:
                replay.push(to_tensor([state]), to_tensor([next_state]), curr_rew, curr_done)
            else:
                replay.push(to_tensor([state]), to_tensor([next_state]), reward, done)

            icnt += 1
            if icnt % args.update == 0 and icnt > 0:
                optim.zero_grad()
                bs, bns, br, bd = replay.sample(args.minibatch)
                bs = bs.to(device)
                bns = bns.to(device)
                br = br.to(device)
                bd = bd.to(device)

                loss = F.mse_loss(policy.forward(bs), (1 - bd) * policy.forward(bns).detach() + br)
                loss.backward()
                losses.append(loss.item())
                optim.step()

        if args.docg and e % args.cgupdate == 0 and icnt > 0:
            cgst = time.time()
            for cgi in range(args.ncgiters):
                cgloss = policy.train_cg_loss_cached()
                cglosses.append(cgloss)
            val_results = val_model(policy, args.valmaxdist, perm_df)
            log.info('      Completed: {:3d} cg backprops | Last CG loss: {:.2f} | Avg all CG Loss: {:.2f} | Elapsed: {:.2f}min | Val results: {}'.format(
                     len(cglosses), np.mean(cglosses[:-args.ncgiters]), np.mean(cglosses), (time.time() - cgst) / 60., val_results
            ))

        if e % args.logiters == 0:
            val_results = val_model(policy, args.valmaxdist, perm_df)
            exp_rate = get_exp_rate(e, args.epochs / 2, args.minexp)
            if hasattr(policy, 'eval_cg_loss'):
                cgst = time.time()
                cgloss = policy.eval_cg_loss()
                cgt = time.time() - cgst

                log.info(f'Epoch {e:5d} | last 10 loss: {np.mean(losses[-10:]):.3f} | cg loss: {cgloss:.3f}, time: {cgt:.2f}s | ' + \
                         f'exp rate: {exp_rate:.3f} | val: {val_results}')
            else:
                log.info(f'Epoch {e:5d} | last 10 loss: {np.mean(losses[-10:]):.3f} | ' + \
                         f'exp rate: {exp_rate:.3f} | val: {val_results}')


    log.info('Done training')
    val_results = val_model(policy, args.valmaxdist, perm_df)
    log.info('Validation results: {}'.format(val_results))
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
    parser.add_argument('--minexp', type=int, default=0.05)
    parser.add_argument('--update', type=int, default=100)
    parser.add_argument('--logiters', type=int, default=1000)
    parser.add_argument('--ncgiters', type=int, default=10)
    parser.add_argument('--docg', action='store_true', default=False)
    parser.add_argument('--cgupdate', type=int, default=500)
    parser.add_argument('--minibatch', type=int, default=128)
    parser.add_argument('--convert', type=str, default='irrep')
    parser.add_argument('--yorprefix', type=str, default=f'/{_prefix}/hopan/irreps/s_8/')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--valmaxdist', type=int, default=6)
    parser.add_argument('--valmaxmoves', type=int, default=10)
    parser.add_argument('--fname', type=str, default='/home/hopan/github/idastar/s8_dists_red.txt')
    args = parser.parse_args()
    main(args)
