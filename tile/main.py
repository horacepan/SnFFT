import os
import time
import random
import argparse
from collections import namedtuple
import json
import pdb
import sys
sys.path.append('../')
from utils import check_memory, get_logger
from tile_memory import ReplayMemory, ReplayMemory2

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tile_irrep_env import TileIrrepEnv
from tile_env import TileEnv
from tensorboardX import SummaryWriter

from tile_dqn import IrrepDQN, MLP

log = get_logger(None)

def eval_model(model, env, trials, max_iters):
    successes = 0
    move_cnt = []
    for e in range(trials):
        state = env.reset()
        for i in range(max_iters):
            action = get_action(model, env, env.grid, e)
            new_state, reward, done, _ = env.step(action)
            state = new_state
            if done:
                successes += 1
                move_cnt.append(i + 1)
                break
    log.info('Validation | {} Trials | Solves: {:.2f} | Avg Solve: {:.2f}'.format(trials, successes, np.mean(move_cnt)))

def exp_rate(explore_epochs, epoch_num, eps_min):
    return max(eps_min, 1 - (epoch_num / (1 + explore_epochs)))

def get_action(pol_net, env, state, e, all_nbrs=None):
    '''
    pol_net: TileDQN
    env: TileIrrepEnv
    state: not actually used! b/c we need to get the neighbors of the current state!
           Well, we _could_ have the state be the grid state!
    e: int
    '''
    if all_nbrs is None:
        all_nbrs = env.all_nbrs(env.grid) # these are irreps

    invalid_moves = [m for m in TileEnv.MOVES if m not in env.valid_moves()]
    vals = pol_net.forward(torch.from_numpy(all_nbrs).float())
    # TODO: this is pretty hacky
    for m in invalid_moves:
        vals[m] = -float('inf')
    return torch.argmax(vals).item()

def update2(pol_net, targ_net, env, batch, opt, discount, ep):
    rewards = torch.from_numpy(batch.reward)
    dones = torch.from_numpy(batch.done)
    states = torch.from_numpy(batch.state)
    nbrs = torch.from_numpy(batch.nbrs)

    # Recall Q(s_t, a_t) = V(s_{t+1})
    pred_vals = pol_net.forward(states)

    # feed ALL nbrs into this!
    irrep_dim = states.size(-1)
    n_nbrs = len(TileEnv.MOVES)
    #batch_all_nbrs = torch.FloatTensor([env.all_nbrs(grid) for grid in batch.grid]).view(-1, irrep_dim)
    batch_all_nbrs = nbrs.view(-1, irrep_dim)
    all_next_vals = targ_net.forward(batch_all_nbrs)
    all_next_vals = all_next_vals.view(len(states), n_nbrs)
    best_vals = all_next_vals.max(dim=1)[0]
    targ_vals = rewards + discount * (1 - dones) * best_vals

    opt.zero_grad()
    loss = F.mse_loss(pred_vals, targ_vals.detach())
    loss.backward()
    opt.step()

    return loss.item()

def update(pol_net, targ_net, env, batch, opt, discount, ep):
    rewards = torch.from_numpy(batch.reward)
    dones = torch.from_numpy(batch.done)
    states = torch.from_numpy(batch.state)
    next_states = torch.from_numpy(batch.next_state)
    dists = torch.from_numpy(batch.scramble_dist).float()
    pred_vals = pol_net.forward(states)
    targ_vals = (rewards + discount * (1 - dones) * targ_net.forward(next_states))

    opt.zero_grad()
    #errors = (1 / (dists + 1.)) * (pred_vals - targ_vals.detach()).pow(2)
    #errors = (pred_vals - targ_vals.detach()).pow(2)
    #loss = errors.sum() / len(targ_vals)
    loss = F.mse_loss(pred_vals, targ_vals.detach())
    loss.backward()
    opt.step()
    return loss.item()

def main(hparams):
    partitions = eval(hparams['partitions'])
    pol_net = IrrepDQN(partitions)
    targ_net = IrrepDQN(partitions)
    env = TileIrrepEnv(hparams['tile_size'], partitions, hparams['reward'])
    opt = torch.optim.Adam(pol_net.parameters(), hparams['lr'])

    if hparams['update_type'] == 1:
        memory = ReplayMemory(hparams['capacity'],
                              env.observation_space.shape[0])
    else:
        memory2 = ReplayMemory2(hparams['capacity'],
                                env.observation_space.shape[0],
                                (env.n, env.n))
    torch.manual_seed(hparams['seed'])
    np.random.seed(hparams['seed'])
    random.seed(hparams['seed'])

    iters = 0
    losses = []
    dones = []
    for e in range(hparams['epochs'] + 1):
        #state = env.reset()
        states = env.shuffle(hparams['shuffle_len'])
        for dist, (grid_state, _x, _y) in enumerate(states):
            nbrs = env.all_nbrs(grid_state)
            if random.random() < exp_rate(hparams['max_exp_epochs'], e, hparams['min_exp_rate']):
                # we compute neighbors b/c we need to cache this?
                action = random.choice(env.valid_moves())
            else:
                action  = get_action(pol_net, env, state, e, all_nbrs=nbrs)

            #new_state, reward, done, _ = env.step(action)
            state = env.cat_irreps(grid_state)
            new_grid_state, reward, done, _ = env.peek(grid_state, _x, _y, action)
            new_state = nbrs[action]
            if hparams['update_type'] == 1:
                memory.push(state, action, new_state, reward, done, dist)
            else:
                memory2.push(state, nbrs, env.grid, reward, done) # only need the new state
            state = new_state
            iters += 1

            if iters % hparams['update_int'] == 0 and iters > 0:
                if hparams['update_type'] == 1:
                    batch = memory.sample(hparams['batch_size'])
                    loss = update(pol_net, targ_net, env, batch, opt, hparams['discount'], e)
                else:
                    batch2 = memory2.sample(hparams['batch_size'])
                    loss = update2(pol_net, targ_net, env, batch2, opt, hparams['discount'], e)
                losses.append(loss)

            if done:
                break

            if iters % hparams['update_int'] == 0 and e > 0:
                targ_net.load_state_dict(pol_net.state_dict())

        dones.append(done)
        if e % hparams['log_int'] == 0 and e > 0:
            log.info('Ep: {:4} | Last {} avg loss: {:.3f} | Exp rate: {:.4}'.format(
                e, hparams['log_int'], np.mean(losses[-hparams['log_int']:]),
                exp_rate(hparams['max_exp_epochs'], e, hparams['min_exp_rate'])
            ))

        if e % hparams['val_int'] == 0 and e > 0:
            eval_model(pol_net, env, 100, 13)
    try:
        if hparams['savename']:
            torch.save(pol_net, './irrep_models/{}.pt'.format(hparams['savename']))
    except:
        pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--partitions', type=str, default='[(8,1)]')
    parser.add_argument('--partitions', type=str, default='[(4,), (3,1), (2, 1, 1), (2, 2), (1, 1, 1, 1)]')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--tile_size', type=int, default=2)
    parser.add_argument('--capacity', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--max_iters', type=int, default=30)
    parser.add_argument('--max_exp_epochs', type=int, default=500)
    parser.add_argument('--min_exp_rate', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--reward', type=str, default='penalty')
    parser.add_argument('--nhid', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shuffle_len', type=int, default=50)

    parser.add_argument('--log_int', type=int, default=100)
    parser.add_argument('--val_int', type=int, default=100)
    parser.add_argument('--update_int', type=int, default=20)
    parser.add_argument('--target_int', type=int, default=20)
    parser.add_argument('--update_type', type=int, default=1)
    parser.add_argument('--savename', type=str, default=None)

    args = parser.parse_args()
    hparams = vars(args)
    print(args)
    try:
        main(hparams)
    except KeyboardInterrupt:
        print('Keyboard escape!')
        check_memory()
