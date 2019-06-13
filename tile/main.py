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
from tile_memory import SimpleMemory

import numpy as np
import torch
import torch.nn as nn
from tile_irrep_env import TileIrrepEnv
from tile_env import TileEnv, grid_to_onehot
from tensorboardX import SummaryWriter

from tile_models import IrrepDVN, IrrepDQN, IrrepDQNMLP, IrrepOnehotDVN

log = get_logger(None, stream=False)

def t2_grid():
    grids = {
        1234: np.array([[1, 2], [3, 4]]),
        1432: np.array([[1, 4], [3, 2]]),
        1243: np.array([[1, 2], [4, 3]]), 
        4132: np.array([[4, 1], [3, 2]]),
        4213: np.array([[4, 2], [1, 3]]),
        3142: np.array([[3, 1], [4, 2]]),
        2413: np.array([[2, 4], [1, 3]]),
        3124: np.array([[3, 1], [2, 4]]),
        2314: np.array([[2, 3], [1, 4]]),
        3421: np.array([[3, 4], [2, 1]]),
        2341: np.array([[2, 3], [4, 1]]),
        4321: np.array([[4, 3], [2, 1]]),
    }
    return grids

def get_action(model, env, grid_state, e, all_nbrs=None, x=None, y=None):
    if all_nbrs is None:
        all_nbrs = env.all_nbrs(grid_state, x, y) # these are irreps

    invalid_moves = [m for m in env.MOVES if m not in env.valid_moves(x, y)]
    vals = model.forward(torch.from_numpy(all_nbrs).float())
    # TODO: this is pretty hacky
    for m in invalid_moves:
        vals[m] = -float('inf')
    return torch.argmax(vals).item()

def show_vals(pol_net, env):
    for k, v in t2_grid().items():
        print('{} | {}'.format(k, pol_net.forward_grid(v, env).max().item()))

def eval_model(model, env, trials, max_iters):
    successes = 0
    move_cnt = []
    for e in range(trials):
        state = env.reset()
        env.shuffle(100)
        for i in range(max_iters):
            if isinstance(model, IrrepDVN):
                action = model.get_action(env, env.grid, e, x=env.x, y=env.y)
            elif isinstance(model, IrrepDQN):
                action = model.get_action_grid(env, env.grid, x=env.x, y=env.y)
            elif isinstance(model, IrrepOnehotDVN):
                action = model.get_action(env, env.grid, x=env.x, y=env.y)

            new_state, reward, done, _ = env.step(action)
            state = new_state
            if done:
                successes += 1
                move_cnt.append(i + 1)
                break
    if len(move_cnt) > 0:
        log.info('Validation | {} Trials | Solves: {:.2f} | LQ {:.2f} | MQ {:.2f} | UQ: {:.2f} | Max solve: {}'.format(
            trials, successes, np.percentile(move_cnt, 25), np.percentile(move_cnt, 50),
            np.percentile(move_cnt, 75), np.max(move_cnt)
        ))
    else:
        log.info('Validation | {} Trials | Solves: {}'.format(trials, len(move_cnt)))

    if env.n == 2 and successes == trials:
        show_vals(model, env)

def exp_rate(explore_epochs, epoch_num, eps_min):
    return max(eps_min, 1 - (epoch_num / (1 + explore_epochs)))

def main(hparams):
    partitions = eval(hparams['partitions'])
    env = TileIrrepEnv(hparams['tile_size'], partitions, hparams['reward'])

    if hparams['model_type'] == 'IrrepDVN':
        log.info('Making IrrepDVN')
        pol_net = IrrepDVN(partitions)
        targ_net = IrrepDVN(partitions)
    elif hparams['model_type'] == 'IrrepDQN':
        log.info('Making IrrepDQN')
        pol_net = IrrepDQN(partitions, nactions=4)
        targ_net = IrrepDQN(partitions, nactions=4)
    elif hparams['model_type'] == 'IrrepOnehotDVN':
        log.info('Making IrrepOnehotDVN')
        pol_net = IrrepOnehotDVN(env.onehot_shape, env.irrep_shape, hparams['n_hid'], partitions)
        targ_net = IrrepOnehotDVN(env.onehot_shape, env.irrep_shape, hparams['n_hid'], partitions)

    opt = torch.optim.Adam(pol_net.parameters(), hparams['lr'])
    memory = SimpleMemory(hparams['capacity'], pol_net.mem_dict(env), pol_net.dtype_dict())

    torch.manual_seed(hparams['seed'])
    np.random.seed(hparams['seed'])
    random.seed(hparams['seed'])

    n_updates = 0
    iters = 0
    losses = []
    dones = []
    rews = set()
    for e in range(hparams['epochs'] + 1):
        shuffle_len = random.randint(hparams['shuffle_min'], hparams['shuffle_max'])
        states = env.shuffle(shuffle_len)
        #grid_state = env.reset(output='grid') # is this a grid state?
        #for i in range(hparams['max_iters']):
        for dist, (grid_state, _x, _y) in enumerate(states):
            #_x, _y = env.x, env.y # C
            nbrs, onehot_nbrs = env.all_nbrs(grid_state, _x, _y)
            if random.random() < exp_rate(hparams['max_exp_epochs'], e, hparams['min_exp_rate']):
                action = random.choice(env.valid_moves(_x, _y))
            else:
                if hparams['model_type'] == 'IrrepDVN':
                    action  = pol_net.get_action(env, grid_state, e, all_nbrs=nbrs, x=_x, y=_y)
                elif hparams['model_type'] == 'IrrepDQN':
                    action  = pol_net.get_action_grid(env, grid_state, x=_x, y=_y)


            new_irrep_state, reward, done, info = env.peek(grid_state, _x, _y, action)
            rews.add(reward)
            #new_irrep_state, reward, done, info = env.step(action) # c
            if hparams['model_type'] == 'IrrepDVN':
                memory.push({
                    'grid_state': grid_state,
                    'irrep_state': env.cat_irreps(grid_state),
                    'irrep_nbrs': nbrs,
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'next_irrep_state': new_irrep_state,
                    'dist': iters
                })
            elif hparams['model_type'] == 'IrrepDQN':
                memory.push({
                    'grid_state': grid_state,
                    'irrep_state': env.cat_irreps(grid_state),
                    'irrep_nbrs': nbrs,
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'next_irrep_state': new_irrep_state,
                    'dist': iters
                })
            elif hparams['model_type'] == 'IrrepOnehotDVN':
                memory.push({
                    #'grid_state': grid_state,
                    'onehot_state': grid_to_onehot(grid_state),
                    'irrep_state': env.cat_irreps(grid_state),
                    'irrep_nbrs': nbrs,
                    'onehot_nbrs': onehot_nbrs,
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'next_irrep_state': new_irrep_state,
                    'dist': iters
                })

            #grid_state = info['grid'] # c
            iters += 1
            if iters % hparams['update_int'] == 0 and iters > 0:
                if hparams['model_type'] == 'IrrepDVN':
                    batch = memory.sample(hparams['batch_size'])
                    loss = pol_net.update(targ_net, env, batch, opt, hparams['discount'], e)
                    n_updates += 1
                    losses.append(loss)
                elif hparams['model_type'] == 'IrrepDQN':
                    batch = memory.sample(hparams['batch_size'])
                    loss = pol_net.update(targ_net, env, batch, opt, hparams['discount'], e)
                    n_updates += 1
                    losses.append(loss)
                elif hparams['model_type'] == 'IrrepOnehotDVN':
                    batch = memory.sample(hparams['batch_size'])
                    loss = pol_net.update(targ_net, env, batch, opt, hparams['discount'], e)
                    n_updates += 1
                    losses.append(loss)
            if done:
                break

            if iters % hparams['update_int'] == 0 and e > 0:
                targ_net.load_state_dict(pol_net.state_dict())

        dones.append(done)
        if e % hparams['log_int'] == 0 and e > 0:
            log.info('Ep: {:4} | Last {} avg loss: {:.3f} | Exp rate: {:.4} | Updates: {}'.format(
                e, hparams['log_int'], np.mean(losses[-hparams['log_int']:]),
                exp_rate(hparams['max_exp_epochs'], e, hparams['min_exp_rate']),
                n_updates
            ))

        if e % hparams['val_int'] == 0 and e > 0:
            if hparams['tile_size'] == 2:
                eval_model(pol_net, env, 200, 8)
            else:
                eval_model(pol_net, env, 200, 40)
            
    print('-------------------------')
    try:
        if hparams['savename']:
            torch.save(pol_net, './irrep_models/{}.pt'.format(hparams['savename']))
    except:
        log.info('Cant save')

    if hparams['tile_size'] == 2:
        show_vals(pol_net, env)
    check_memory()
    log.info('Rewards seed: {}'.format(rews))
    eval_model(pol_net, env, 200, 8)

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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_hid', type=int, default=16)
    parser.add_argument('--shuffle_len', type=int, default=50)
    parser.add_argument('--shuffle_min', type=int, default=20)
    parser.add_argument('--shuffle_max', type=int, default=80)

    parser.add_argument('--log_int', type=int, default=100)
    parser.add_argument('--val_int', type=int, default=10000)
    parser.add_argument('--update_int', type=int, default=20)
    parser.add_argument('--target_int', type=int, default=20)
    parser.add_argument('--update_type', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='IrrepDVN')
    parser.add_argument('--savename', type=str, default=None)

    args = parser.parse_args()
    hparams = vars(args)
    print(args)
    try:
        main(hparams)
    except KeyboardInterrupt:
        print('Keyboard escape!')
        check_memory()
