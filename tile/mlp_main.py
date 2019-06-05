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
from tile_memory import ReplayMemory, ReplayMemory2, SimpleMemory

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tile_irrep_env import TileIrrepEnv
from tile_env import *
from tensorboardX import SummaryWriter

from tile_models import IrrepDQN, MLP, TileBaselineQ

log = get_logger(None, stream=False)

def exp_rate(explore_epochs, epoch_num, eps_min):
    return max(eps_min, 1 - (epoch_num / (1 + explore_epochs)))

def mlp_get_action(model, env, state, e):
        '''
        state: 1 x n_in tensor
        Returns int of the argmax
        '''
        state = torch.from_numpy(state).float().unsqueeze(0)
        vals = model.forward(state)
        return vals.argmax(dim=1).item()

def eval_model(model, env, trials, max_iters):
    successes = 0
    move_cnt = []
    for e in range(trials):
        state = env.reset()
        for i in range(max_iters):
            action = model.get_action(state)
            new_state, reward, done, _ = env.step(action)
            state = new_state
            if done:
                successes += 1
                move_cnt.append(i + 1)
                break
    if len(move_cnt) > 0:
        print('Validation | {} Trials | Solves: {:.2f} | LQ: {:.2f} | Avg Solve: {:.2f}  | UQ: {:2f}'.format(
            trials, successes, np.percentile(move_cnt, 25), np.mean(move_cnt), np.percentile(move_cnt, 75)
        ))
    else:
        print('Validation | {} Trials | Solves: 0')


def main(hparams):
    partitions = eval(hparams['partitions'])
    #env = TileIrrepEnv(hparams['tile_size'], partitions, hparams['reward'])
    env = TileEnv(hparams['tile_size'], one_hot=True, reward=hparams['reward'])
    pol_net = TileBaselineQ(env.observation_space.shape[0], hparams['nhid'], env.actions)
    targ_net = TileBaselineQ(env.observation_space.shape[0], hparams['nhid'], env.actions)

    opt = torch.optim.Adam(pol_net.parameters(), hparams['lr'])

    # this is probably something each individual model should own
    mem_dict = {
        'onehot_state': (env.observation_space.shape[0],),
        'next_onehot_state': (env.observation_space.shape[0],),
        'action': (1,),
        'reward': (1,),
        'done': (1,),
        'dist': (1,),
        'scramble_dist': (1,),
    }
    dtype_dict = {
        'action': int,
        'scramble_dist': int,
    }
    memory = SimpleMemory(hparams['capacity'], mem_dict, dtype_dict)
    torch.manual_seed(hparams['seed'])
    np.random.seed(hparams['seed'])
    random.seed(hparams['seed'])

    print('Before training')
    #eval_model(pol_net, env, 100, 100)

    iters = 0
    losses = []
    dones = []
    tot_dists = []
    for e in range(hparams['epochs'] + 1):
        onehot_state = env.reset()
        #states = env.shuffle(hparams['shuffle_len'])
        for i in range(hparams['max_iters']):
        # states are onehot vectors
        #for dist, (grid_state, _x, _y) in enumerate(states):
            #onehot_state = grid_to_onehot(grid_state)
            _x, _y = env.x, env.y
            if random.random() < exp_rate(hparams['max_exp_epochs'], e, hparams['min_exp_rate']):
                action = random.choice(env.valid_moves(_x, _y))
            else:
                action = pol_net.get_action(onehot_state)

            # need option to do peek instead of step if we want to use a shuffle trajectory!
            new_state, reward, done, _ = env.step(action)
            #new_grid, reward, done, info = env.peek(grid_state, _x, _y, action)
            #new_state = grid_to_onehot(new_grid)
            memory.push({
                'onehot_state': onehot_state,
                'action': action,
                'reward': reward,
                'done': done,
                'next_onehot_state': new_state,
                'dist': 0
            })
            state = new_state
            onehot_state = new_state
            iters += 1

            if iters % hparams['update_int'] == 0 and iters > 0:
                batch = memory.sample(hparams['batch_size'])
                #loss = pol_net.update(targ_net, env, batch, opt, hparams['discount'], e)
                loss = pol_net.update_simple(targ_net, env, batch, opt, hparams['discount'], e)
                losses.append(loss)

            if iters % hparams['update_int'] == 0 and e > 0:
                targ_net.load_state_dict(pol_net.state_dict())

        #tot_dists.append(dist)

        if e % hparams['log_int'] == 0 and e > 0:
            _k = 100
            log.info('Ep: {:4} | Last {} avg loss: {:.3f} | Exp rate: {:.4}'.format(
                e, hparams['log_int'], np.mean(losses[-hparams['log_int']:]),
                exp_rate(hparams['max_exp_epochs'], e, hparams['min_exp_rate'])
            ))

    try:
        if not (hparams['savename'] is None):
            log.info('Saving model to: {}'.format(hparams['savename']))
            torch.save(pol_net, './models/{}.pt'.format(hparams['savename']))
    except:
        pdb.set_trace()
    eval_model(pol_net, env, 100, 100)
    check_memory()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--partitions', type=str, default='[(8,1)]')
    parser.add_argument('--partitions', type=str, default='[(4,), (3,1), (2, 1, 1), (2, 2), (1, 1, 1, 1)]')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--tile_size', type=int, default=2)
    parser.add_argument('--capacity', type=int, default=10000)
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
    parser.add_argument('--update_int', type=int, default=20)
    parser.add_argument('--target_int', type=int, default=20)
    parser.add_argument('--update_type', type=int, default=1)
    parser.add_argument('--savename', type=str, default='model')

    args = parser.parse_args()
    hparams = vars(args)
    print(args)
    try:
        main(hparams)
    except KeyboardInterrupt:
        print('Keyboard escape!')
        check_memory()
