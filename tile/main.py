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
from tile_env import TileEnv
from tensorboardX import SummaryWriter

from tile_models import IrrepDQN, MLP, IrrepDQNMLP

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

def show_vals(pol_net, env):
    for k, v in t2_grid().items():
        print('{} | {}'.format(k, pol_net.forward_grid(v, env).item()))

def get_action(model, env, grid_state, e, all_nbrs=None, x=None, y=None):
    if all_nbrs is None:
        all_nbrs = env.all_nbrs(grid_state, x, y) # these are irreps

    invalid_moves = [m for m in env.MOVES if m not in env.valid_moves(x, y)]
    vals = model.forward(torch.from_numpy(all_nbrs).float())
    # TODO: this is pretty hacky
    for m in invalid_moves:
        vals[m] = -float('inf')
    return torch.argmax(vals).item()

def eval_model(model, env, trials, max_iters):
    successes = 0
    move_cnt = []
    for e in range(trials):
        state = env.reset()
        env.shuffle(100)
        for i in range(max_iters):
            action = model.get_action(env, env.grid, e, x=env.x, y=env.y)
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

    if successes == trials:
        show_vals(model, env)

def exp_rate(explore_epochs, epoch_num, eps_min):
    return max(eps_min, 1 - (epoch_num / (1 + explore_epochs)))

def update2(pol_net, targ_net, env, batch, opt, discount, ep):
    rewards = torch.from_numpy(batch['reward'])
    dones = torch.from_numpy(batch['done'])
    states = torch.from_numpy(batch['irrep_state'])
    nbrs = torch.from_numpy(batch['irrep_nbrs'])
    grids = torch.from_numpy(batch['grid_state'])

    # Recall Q(s_t, a_t) = V(s_{t+1})
    pred_vals = pol_net.forward(states)

    irrep_dim = states.size(-1)
    batch_all_nbrs = nbrs.view(-1, irrep_dim)
    all_next_vals = targ_net.forward(batch_all_nbrs)
    all_next_vals = all_next_vals.view(len(states), -1)
    best_vals = all_next_vals.max(dim=1)[0]
    targ_vals = rewards + discount * (1 - dones) * best_vals

    opt.zero_grad()
    loss = F.mse_loss(pred_vals, targ_vals.detach())
    loss.backward()
    opt.step()
    if loss.item() < 0.0001:
        print('Super small loss!')
    return loss.item()

def update(pol_net, targ_net, env, batch, opt, discount, ep):
    '''
    s = state
    Q(s, a) = V(next state)
    SO we need the grid of the next state and the irrep of next state
    We dont actually need the current state
    Q(s, a), r + discount * best neighbor of next state
    The grid actually gives you EVERYTHING!
    So we should only store the grid?
    '''
    rewards = torch.from_numpy(batch['reward'])
    dones = torch.from_numpy(batch['done'])
    states = torch.from_numpy(batch['irrep_state'])
    next_states = torch.from_numpy(batch['next_irrep_state'])
    dists = torch.from_numpy(batch['scramble_dist']).float()
    pred_vals = pol_net.forward(next_states)
    targ_vals = (rewards + discount * (1 - dones) * targ_net.forward(next_states))

    opt.zero_grad()
    loss = F.mse_loss(pred_vals, targ_vals.detach())
    loss.backward()
    opt.step()
    return loss.item()

def main(hparams):
    partitions = eval(hparams['partitions'])

    if hparams['update_type'] == 1:
        pol_net = IrrepDQN(partitions)
        targ_net = IrrepDQN(partitions)
    else:
        pol_net = IrrepDQNMLP(partitions[0], hparams['nhid'], 1)
        targ_net = IrrepDQNMLP(partitions[0], hparams['nhid'], 1)

    env = TileIrrepEnv(hparams['tile_size'], partitions, hparams['reward'])
    opt = torch.optim.Adam(pol_net.parameters(), hparams['lr'])

    if hparams['update_type'] < 3:
        mem_dict = {
            # this should be?
            'grid_state': env.grid.shape,
            'next_grid_state': env.grid.shape,
            'irrep_state': (env.observation_space.shape[0],),
            'irrep_nbrs': (4, env.observation_space.shape[0]),
            'next_irrep_state': (env.observation_space.shape[0],),
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
        print('Made simple memory')

    torch.manual_seed(hparams['seed'])
    np.random.seed(hparams['seed'])
    random.seed(hparams['seed'])
    if hparams['prefill'] and hparams['update_type'] == 1:
        init_grid=  np.array(range(1, env.n * env.n + 1)).reshape(env.n, env.n)
        init_irrep = env.cat_irreps(init_grid)
        memory.prefill({
            'grid_state': init_grid,
            'irrep_state': init_irrep,
            'irrep_nbrs': env.all_nbrs(init_grid, env.n - 1, env.n - 1),
            'reward': 0,
            'done': 1,
            'dist': 0,
        })

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
            nbrs = env.all_nbrs(grid_state, _x, _y)
            if random.random() < exp_rate(hparams['max_exp_epochs'], e, hparams['min_exp_rate']):
                action = random.choice(env.valid_moves(_x, _y))
            else:
                action  = pol_net.get_action(env, grid_state, e, all_nbrs=nbrs, x=_x, y=_y)

            new_irrep_state, reward, done, info = env.peek(grid_state, _x, _y, action)
            rews.add(reward)
            #new_irrep_state, reward, done, info = env.step(action) # c
            if hparams['update_type'] < 3:
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

            #grid_state = info['grid'] # c
            iters += 1
            if iters % hparams['update_int'] == 0 and iters > 0:
                batch = memory.sample(hparams['batch_size'])
                loss = update2(pol_net, targ_net, env, batch, opt, hparams['discount'], e)
                n_updates += 1
                #batch2 = memory2.sample(hparams['batch_size'])
                #loss = update2(pol_net, targ_net, env, batch2, opt, hparams['discount'], e)
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
        #pdb.set_trace()

    if hparams['tile_size'] == 2:
        show_vals(pol_net, env)
    check_memory()
    log.info('Rewards seed: {}'.format(rews))

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
    parser.add_argument('--shuffle_min', type=int, default=20)
    parser.add_argument('--shuffle_max', type=int, default=80)
    parser.add_argument('--prefill', action='store_true')

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
