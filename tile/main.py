import os
import time
import random
import argparse
from collections import namedtuple
import json
import pdb
import sys
sys.path.append('../')
from utils import check_memory
from tile_memory import ReplayMemory

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tile_irrep_env import TileIrrepEnv
from tile_env import TileEnv
from tensorboardX import SummaryWriter

from tile_dqn import IrrepDQN, MLP
# NOT DOING ANYTHING WITH THE TRUE FOURIER MATRICES!

def exp_rate(explore_epochs, epoch_num, eps_min):
    return max(eps_min, 1 - (epoch_num / (1 + explore_epochs)))

def get_action(pol_net, env):
    vals = np.zeros(env.action_space.n)
    for a, nbr in env.neighbors().items():
        # TODO: This is pretty janky
        try:
            vals[a] = pol_net.forward(torch.from_numpy(nbr).unsqueeze(0).float())
        except:
            pdb.set_trace()

    return np.argmax(vals)

def update(pol_net, targ_net, env, batch, opt, discount, ep):
    #if ep % 2 == 0:
    #    pred_net, targ_net = pol_net, targ_net
    #else:
    #    pred_net, targ_net = targ_net, pol_net

    reward = torch.from_numpy(batch.reward)
    dones = torch.from_numpy(batch.done)
    state = torch.from_numpy(batch.state)
    next_state = torch.from_numpy(batch.next_state)

    # TODO: do the proper argmax action
    pred_vals = pol_net.forward(state)
    targ_vals = reward + discount * (1 - dones) * targ_net.forward(next_state)

    opt.zero_grad()
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
    memory = ReplayMemory(hparams['capacity'], {'state': env.observation_space.shape[0]})

    iters = 0
    losses = []
    dones = []
    for e in range(hparams['epochs'] + 1):
        state = env.reset()

        for i in range(hparams['max_iters']):
            if random.random() < exp_rate(hparams['max_exp_epochs'], e, hparams['min_exp_rate']):
                action = random.choice(list(env.neighbors()))
            else:
                action = get_action(pol_net, env)

            new_state, reward, done, _ = env.step(action)
            memory.push(state, action, new_state, reward, done)
            state = new_state
            iters += 1

            if iters % hparams['update_int'] == 0 and iters > 0:
                batch = memory.sample(hparams['batch_size'])
                loss = update(pol_net, targ_net, env, batch, opt, hparams['discount'], e)
                losses.append(loss)

            if done:
                break

        if e % hparams['update_int'] == 0 and e > 0:
            targ_net.load_state_dict(pol_net.state_dict())

        dones.append(done)
        if e % hparams['log_int'] == 0 and e > 0:
            print('Ep: {:4} | Last {} ep solves: {:.3f} | All avg Loss: {:.3f} | Exp rate: {:.4}'.format(
                e, hparams['log_int'], np.mean(dones[-hparams['log_int']:]), np.mean(losses),
                exp_rate(hparams['max_exp_epochs'], e, hparams['min_exp_rate'])
            ))

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

    parser.add_argument('--log_int', type=int, default=100)
    parser.add_argument('--update_int', type=int, default=20)
    parser.add_argument('--target_int', type=int, default=20)

    args = parser.parse_args()
    hparams = vars(args)
    print(args)
    try:
        main(hparams)
    except KeyboardInterrupt:
        print('Keyboard escape!')
        check_memory()
