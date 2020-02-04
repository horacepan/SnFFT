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

from yor_dataset import YorConverter
from utility import nbrs, perm_onehot, ReplayBuffer
from logger import get_logger

def get_start_states():
    df = pd.read_csv('/home/hopan/github/idastar/s8_dists_red.txt', header=None,
                     dtype={0: str, 1: int}, nrows=24)
    states = [tuple(int(i) for i in row[0]) for _, row in df.iterrows()]
    return states

def get_exp_rate(epoch, max_epochs, min_exp):
    return max(min_exp, epoch / max_epochs)

def get_reward_done(state, done_states):
    '''
    Returns a tuple of reward and done (1 or 0)
    '''
    if state in done_states:
        return 0, 1
    return -1, 0

def random_walk(start, nbr_func, max_iters):
    states = []
    for _ in range(max_iters):
        snbrs = nbr_func(start)
        nbr = random.choice(snbrs)
        states.append(nbr)

    return states

class RlPolicy(nn.Module):
    def __init__(self, nin, nout):
        super(RlPolicy, self).__init__()
        self.nin = nin
        self.nin = nin
        self.net = nn.Linear(nin, nout)

    def forward(self, x):
        return self.net(x)

    def opt_move(self, x):
        output = self.forward(x)
        return output.argmax(dim=1).item()

    def reset_parameters(self):
        self.net.weight.data.normal(std=0.1)

class MLP(nn.Module):
    def __init__(self, nin, nhid, nout):
        super(MLP, self).__init__()
        self.nin = nin
        self.nhid = nhid
        self.nout = nout
        self.net = nn.Sequential(
            nn.Linear(nin, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nout)
        )

    def forward(self, x):
        return self.net(x)

    def opt_move(self, x):
        output = self.forward(x)
        return output.argmax()

def val_model(net):
    pass

def main(args):
    log = get_logger(args.logfile)
    perms = list(permutations(tuple(i for i in range(1, 9))))
    tens = perm_onehot
    if args.convert == 'onehot':
        tens = perm_onehot
    elif args.convert == 'irrep':
        irreps = [(3, 2, 2, 1)]
        tens = YorConverter(irreps, args.yorprefix, perms)

    #policy = RlPolicy(tens(perms[0]).numel(), 1)
    policy = MLP(tens(perms[0]).numel(), 32, 1)
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    start_states = get_start_states()
    done_states = set(start_states)

    replay = ReplayBuffer(tens(perms[0]).numel(), args.capacity)
    icnt = 0
    losses = []
    wins = 0
    for e in range(args.epochs):
        start = random.choice(start_states)
        states = random_walk(start, nbrs, args.maxep)

        for state in states:
            _nbrs = nbrs(state)
            if random.random() < get_exp_rate(e, args.maxep, args.minexp):
                move = np.random.choice(6)
            else:
                nbrs_tens = torch.cat([tens(n) for n in _nbrs], dim=0)
                move = policy.opt_move(nbrs_tens)
            next_state = _nbrs[move]
            reward, done = get_reward_done(next_state, done_states)
            replay.push(tens(state), tens(next_state), reward, done)
            icnt += 1
            if done:
                wins += 1

            if icnt % args.update == 0 and icnt > 0:
                optim.zero_grad()
                bs, bns, br, bd = replay.sample(args.minibatch)
                # what do we want to minimize?
                loss = F.mse_loss(policy(bs), (1 - bd) * policy(bns) + br)
                loss.backward()
                losses.append(loss.item())
                optim.step()

            if icnt % args.logiters == 0 and icnt > 0:
                avg_loss = np.mean(losses)
                loss = np.mean(losses[-10])
                log.info(f'Iter {icnt:4d} | last 10 loss: {loss.item():.3f} | avg loss: {avg_loss:.3f} | wins: {wins/icnt}')

    log.info('Done training')
    states = random_walk(start, nbrs, 5)
    all_nbrs = [nbrs(s) for s in states]
    for nlst, st in zip(all_nbrs, states):
        nbrs_tens = torch.cat([tens(n) for n in nlst], dim=0)
        nbr_vals = policy.forward(nbrs_tens)
        st_val = policy.forward(tens(st))
        opt_ = policy.opt_move(nbrs_tens)
        nbr_vals = [n.item() for n in nbr_vals]
        log.info('St: {} | Nbr vals: {} | opt: {}'.format(st_val.item(), nbr_vals, opt_))
        pdb.set_trace()
    pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, default=f'./logs/rl/{time.time()}.log')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--capacity', type=int, default=10000)
    parser.add_argument('--maxep', type=int, default=20)
    parser.add_argument('--minexp', type=int, default=0.05)
    parser.add_argument('--update', type=int, default=50)
    parser.add_argument('--logiters', type=int, default=2000)
    parser.add_argument('--minibatch', type=int, default=64)
    parser.add_argument('--convert', type=str, default='irreps')
    parser.add_argument('--yorprefix', type=str, default='/local/hopan/irreps/s_8/')
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    main(args)
