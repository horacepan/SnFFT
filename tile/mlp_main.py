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

def exp_rate(explore_epochs, epoch_num, eps_min):
    return max(eps_min, 1 - (epoch_num / (1 + explore_epochs)))

def mlp_get_action(pol_net, env, state, e):
    t_state = torch.from_numpy(state).float().unsqueeze(0)
    vals = pol_net.forward(t_state)
    return torch.argmax(vals).item()

def update(pol_net, targ_net, env, batch, opt, discount, ep):
    rewards = torch.from_numpy(batch.reward)
    actions = torch.from_numpy(batch.action).long()
    dones = torch.from_numpy(batch.done)
    states = torch.from_numpy(batch.state)
    next_states = torch.from_numpy(batch.next_state)
    dists = torch.from_numpy(batch.scramble_dist).float()

    pred_vals = pol_net.forward(states)
    vals = torch.gather(pred_vals, 1, actions)

    targ_max = targ_net.forward(next_states).max(dim=1)[0]
    targ_vals = rewards + discount * (1 - dones) * targ_max.unsqueeze(-1)

    opt.zero_grad()
    #scale = 10. / (1. + dists)
    scale = 1
    loss = (scale * (vals - targ_vals.detach())).pow(2).mean()
    #loss = F.mse_loss(vals, targ_vals.detach())
    loss.backward()
    opt.step()
    return loss.item()

def eval_model(model, env, trials, max_iters):
    successes = []
    move_cnt = []
    for e in range(trials):
        state = env.reset()
        for i in range(max_iters):
            action = mlp_get_action(model, env, state, e)
            new_state, reward, done, _ = env.step(action)
            state = new_state
            if done:
                successes.append(True)
                move_cnt.append(i + 1)
                break
    pdb.set_trace()
    print('Validation | {} Trials | Solves: {:.2f} | Avg Solve: {:.2f}'.format(trials, np.mean(successes), np.mean(move_cnt)))

def main(hparams):
    partitions = eval(hparams['partitions'])
    #env = TileIrrepEnv(hparams['tile_size'], partitions, hparams['reward'])
    env = TileEnv(hparams['tile_size'], one_hot=True, reward=hparams['reward'])
    pol_net = MLP(env.observation_space.shape[0], hparams['nhid'], env.actions)
    targ_net = MLP(env.observation_space.shape[0], hparams['nhid'], env.actions)

    opt = torch.optim.Adam(pol_net.parameters(), hparams['lr'])

    memory = ReplayMemory(hparams['capacity'],
                          env.observation_space.shape[0])
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
        #state = env.reset()
        states = env.shuffle(50)
        # are the shuffles grids or one hots?

        #for i in range(hparams['max_iters']):
        # states are onehot vectors
        for dist, (grid_state, onehot_state, _x, _y) in enumerate(states):
            if random.random() < exp_rate(hparams['max_exp_epochs'], e, hparams['min_exp_rate']):
                action = random.choice(env.valid_moves())
            else:
                action = mlp_get_action(pol_net, env, onehot_state, e)

            # need option to do peek instead of step if we want to use a shuffle trajectory!
            #new_state, reward, done, _ = env.step(action)
            new_grid, reward, done, info = env.peek(grid_state, _x, _y, action)
            new_state = info['onehot']
            memory.push(onehot_state, action, new_state, reward, done, 0)
            state = new_state
            iters += 1

            if iters % hparams['update_int'] == 0 and iters > 0:
                batch = memory.sample(hparams['batch_size'])
                loss = update(pol_net, targ_net, env, batch, opt, hparams['discount'], e)
                losses.append(loss)

            if iters % hparams['update_int'] == 0 and e > 0:
                targ_net.load_state_dict(pol_net.state_dict())

        tot_dists.append(dist)

        if e % hparams['log_int'] == 0 and e > 0:
            log.info('Ep: {:4} | Last {} ep solves: {:.3f} | All avg Loss: {:.3f} | Exp rate: {:.4} | Avg done: {:.2f}'.format(
                e, hparams['log_int'], np.mean(dones[-hparams['log_int']:]), np.mean(losses[-100:]),
                exp_rate(hparams['max_exp_epochs'], e, hparams['min_exp_rate']), np.mean(tot_dists[-hparams['log_int']:])
            ))

    eval_model(pol_net, env, 100, 100)
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

    parser.add_argument('--log_int', type=int, default=100)
    parser.add_argument('--update_int', type=int, default=20)
    parser.add_argument('--target_int', type=int, default=20)
    parser.add_argument('--update_type', type=int, default=1)

    args = parser.parse_args()
    hparams = vars(args)
    print(args)
    try:
        main(hparams)
    except KeyboardInterrupt:
        print('Keyboard escape!')
        check_memory()
