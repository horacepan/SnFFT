#!/usr/bin/env python
# coding: utf-8
import sys
import os
import time
import json
sys.path.append('/home/hopan/github/SnFFT/exp/')

import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from tqdm import tqdm
from logger import get_logger
from cube3 import Cube3, Cube3Edge, Cube3Corner
from wreath_fourier import CubePolicy, CubePolicyLowRank, CubePolicyLowRankEig
from rlmodels import MLPResModel, DVN
from utility import ReplayBuffer, update_params, check_memory
from replay_buffer import ReplayBufferMini
import pdb

from astar import a_star, gen_mlp_model
from cube_main import try_load_weights

sys.path.append('../')
from complex_utils import cmse, cmse_real

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def can_solve(model, env, state, max_steps, argmin=True):
    if env.is_done(state):
        return True

    for _ in range(max_steps):
        nbrs = env.nbrs(state)
        # this is suspect
        nbr_vals = model.forward(model.to_tensor(nbrs)).detach()
        if any(env.is_done(n) for n in nbrs):
            return True
        if argmin:
            best_action = nbr_vals.argmin().item()
        else:
            best_action = nbr_vals.argmax().item()
        state = env.step(state, best_action)

    return False

def astar_bench(policy, env, trials, max_exp, scramble_len, argmin=True):
    correct = 0
    solve_lens = []

    done_state = env.start_state()
    if argmin:
        hfunc = lambda s: policy.forward(policy.to_tensor([s]))[0].item()
    else:
        hfunc = lambda s: -policy.forward(policy.to_tensor([s]))[0].item()

    for _ in range(trials):
        start = env.random_state(scramble_len + (1 if random.random() < 0.5 else 0))
        corr, num = a_star(start, done_state, env, hfunc, max_exp)
        if corr:
            solve_lens.append(num)
            correct += 1

    return correct, solve_lens

def benchmark(model, env, trials, max_steps, scramble_len=1000):
    correct = 0

    for _ in range(args.trials):
        state = env.random_state(scramble_len + (1 if random.random() > 0.5 else 0))
        correct += can_solve(model, env, state, max_steps)
    return correct

def unique_logger(sumdir):
    cnt = 0
    logfile = os.path.join(sumdir, f'output.log')
    while os.path.exists(logfile):
        logfile = os.path.join(sumdir, f'output{cnt}.log')
        cnt += 1
    return logfile

def main(args):
    set_seed(args.seed)
    sumdir = os.path.join(f'{args.savedir}', f'{args.env}', f'{args.notes}/seed_{args.seed}')
    if args.savelog:
        try:
            os.makedirs(sumdir)
        except:
            pass
        logfile = unique_logger(sumdir)
        log = get_logger(fname=logfile, stdout=True, tofile=True)
        json.dump(args.__dict__, open(os.path.join(sumdir, 'args.json'), 'w'), indent=2)
        log.info("Saving in: {} | Starting ...".format(logfile))
    else:
        log = get_logger(fname=None, stdout=True, tofile=False)

    if args.env == 'Cube3':
        env = Cube3()
    elif args.env == 'Cube3Edge':
        env = Cube3Edge()
    elif args.env == 'Cube3Corner':
        env = Cube3Corner()
    else:
        log.info(f'{args.env} is not a valid env')
        exit()

    nactions = len(env.moves)
    nin = env.to_tensor([env.start_state()]).numel()
    nout = 1
    updates = 0
    icnt = 0
    start_ep = 0

    if args.convert == 'onehot':
        replay = ReplayBuffer(nin, args.capacity)
    else:
        log.info('Creating mini replay')
        replay = ReplayBufferMini(args.capacity)

    if args.model == 'resmlp':
        policy = MLPResModel(nin, args.resfc1, args.resfc2, nout, args.nres, to_tensor=env.to_tensor, std=args.std)
        target = MLPResModel(nin, args.resfc1, args.resfc2, nout, args.nres, to_tensor=env.to_tensor, std=args.std)
        to_tensor = lambda g: wreath_onehot(g, 3)
    elif args.model == 'linear':
        irreps = eval(args.cube_irreps)
        log.info(f'Policy using Irreps: {irreps}')
        policy = CubePolicy(irreps, std=args.std)
        target = CubePolicy(irreps, std=args.std, irrep_loaders=policy.irrep_loaders)
        to_tensor = lambda g: policy.to_tensor(g)
        log.info('Cube policy dim: {}'.format(policy.dim))
    elif args.model == 'lowrank':
        irreps = eval(args.cube_irreps)
        log.info(f'Policy using Irreps: {irreps} | Rank: {args.rank}')
        policy = CubePolicyLowRank(irreps, rank=args.rank, std=args.std)
        target = CubePolicyLowRank(irreps, rank=args.rank, std=args.std, irrep_loaders=policy.irrep_loaders)
        to_tensor = lambda g: policy.to_tensor(g)
        log.info('Cube policy dim: {}'.format(policy.dim))
    else:
        log.info('Not a valid model!')
    policy.to(device)
    target.to(device)

    if args.init == 'xavier':
        policy.xinit()
        target.xinit()

    if args.loadmodel:
        loaded, start_ep = try_load_weights(sumdir, policy, target)
        if loaded:
            log.info(f'Loaded models! | Starting at epoch: {start_ep}')
            set_seed(args.seed + start_ep)

    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    log.info("Done setup ...")

    for e in range(start_ep, start_ep + args.epochs + 1):
        states = env.random_walk(args.rw_length)
        for state in states:
            action = random.randint(0, len(env.moves) - 1)
            next_state = env.step(state, action)
            reward = 1 if env.is_done(state) else -1
            #reward = 0 if env.is_done(state) else 1
            done = 1 if env.is_done(state) else 0
            if args.convert == 'onehot':
                replay.push(env.to_tensor([state]), action, env.to_tensor([next_state]), reward, done, state, next_state, icnt+1)
            else:
                replay.push(state, action, next_state, reward, done)

            if icnt % args.update_int == 0:
                optim.zero_grad()
                if args.convert == 'onehot':
                    bs, ba, bns, br, bd, bs_tups, bns_tups, bidx = replay.sample(args.minibatch, device)
                else:
                    bs_tups, ba, bns_tups, br, bd = replay.sample(args.minibatch, device)

                if args.convert == 'onehot':
                    bs_nbrs = [n for tup in bs_tups for n in env.nbrs(tup)]
                    bs_nbrs_tens = env.to_tensor(bs_nbrs)
                    opt_nbr_vals, _ = target.forward(bs_nbrs_tens).detach().reshape(-1, nactions).min(dim=1, keepdim=True)
                    loss = F.mse_loss(policy.forward(bs), args.discount * (1 - bd) * opt_nbr_vals + br)
                    loss.backward()
                    optim.step()
                else:
                    bs_re, bs_im = to_tensor(bs_tups)
                    val_re, val_im = policy.forward_complex(bs_re, bs_im)
                    bs_nbrs = [n for tup in bs_tups for n in env.nbrs(tup)]
                    bs_nbrs_re, bs_nbrs_im = to_tensor(bs_nbrs)
                    nr, ni = target.forward_complex(bs_nbrs_re, bs_nbrs_im)
                    opt_nbr_vals_re, opt_idx = nr.reshape(-1, nactions).max(dim=1, keepdim=True)
                    opt_nbr_vals_im = ni.reshape(-1, nactions).gather(1, opt_idx)
                    loss = cmse(val_re, val_im,
                                    (1 - bd) * args.discount * opt_nbr_vals_re.detach() + br,
                                    (1 - bd) * args.discount * opt_nbr_vals_im.detach())
                    loss.backward()
                    optim.step()

            icnt += 1

        if e % args.target_update == 0 and icnt > 0:
            update_params(target, policy)
            updates += 1

        if e % args.logint == 0:
            #correct = benchmark(policy, env, trials=args.trials, max_steps=args.max_steps, scramble_len=args.scramble_len)
            _astart = time.time()
            correct, solves_lens = astar_bench(policy, env, trials=args.trials, max_exp=args.max_exp, scramble_len=args.scramble_len, argmin=False)
            _atime = (time.time() - _astart) / 60.
            avg = 0
            if correct > 0:
                avg = np.mean(solves_lens)

            log.info(f'Epoch: {e:4d} | A* Solves: {correct/args.trials:.2f} | Avg len: {avg:.2f} | Updates: {updates:4d} | Iters: {icnt:6d} | Mem: {check_memory(verbose=False):.2f}mb | A* time: {_atime:.2f}mins')

            torch.save(policy.state_dict(), os.path.join(sumdir, f'model_last.pt'))

        if e % 10000 == 0 and e > 0:
            torch.save(policy.state_dict(), os.path.join(sumdir, f'model_{e}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='Cube3')
    parser.add_argument('--savedir', type=str, default='/scratch/hopan/cube/irreps/')
    parser.add_argument('--savelog', action='store_true', default=False)
    parser.add_argument('--loadmodel', action='store_true', default=False)
    parser.add_argument('--notes', type=str, default='test')

    # hyperparams
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--capacity', type=int, default=100000)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--minibatch', type=int, default=32)
    parser.add_argument('--discount', type=float, default=1.0)

    # intervals, test params
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--logint', type=int, default=1000)
    parser.add_argument('--target_update', type=int, default=20)
    parser.add_argument('--update_int', type=int, default=50)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--max_exp', type=int, default=100)
    parser.add_argument('--rw_length', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=10)
    parser.add_argument('--scramble_len', type=int, default=100)

    # model params
    parser.add_argument('--convert', type=str, default='onehot')
    parser.add_argument('--model', type=str, default='resmlp')
    parser.add_argument('--cube_irreps', type=str, default='[((4, 2, 2), ((4,), (1, 1), (1, 1))), ((2, 3, 3), ((2,), (1, 1, 1), (1, 1, 1)))]')
    parser.add_argument('--rank', type=int, default=10)
    parser.add_argument('--std', type=float, default=0.01)
    parser.add_argument('--resfc1', type=int, default='1024')
    parser.add_argument('--resfc2', type=int, default='2048')
    parser.add_argument('--nres', type=int, default='1')
    parser.add_argument('--init', type=str, default='default')

    args = parser.parse_args()
    main(args)
