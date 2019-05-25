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
from memory import ReplayMemory

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tile_irrep_env import TileIrrepEnv
from tensorboardX import SummaryWriter

from tile_dqn import IrrepDQN
# NOT DOING ANYTHING WITH THE TRUE FOURIER MATRICES!

def exp_rate(max_exp_eps, ep, min_exp_rate):
    return 0.2

def get_action(pol_net, env):
    vals = np.zeros(env.action_space.n)
    for a, nbr in env.neighbors().items():
        # TODO: This is pretty janky
        vals[a] = pol_net.forward(torch.from_numpy(nbr).unsqueeze(0).float())

    return np.argmax(vals)

def main(hparams):
    partitions = eval(hparams['partitions'])
    pol_net = IrrepDQN(partitions)
    targ_net = IrrepDQN(partitions)
    opt = torch.optim.SGD(pol_net.parameters(), hparams['lr'])
    env = TileIrrepEnv(hparams['tile_size'], partitions)
    iters = 0
    memory = ReplayMemory(hparams['capacity'])

    for e in range(hparams['epochs']):
        state = env.reset()

        for i in range(hparams['max_iters']):
            if random.random() < exp_rate(hparams['max_exp_epochs'], e, hparams['min_exp_rate']):
                action = random.choice(list(env.neighbors()))
            else:
                action = get_action(pol_net, env)

            env.step(action)

            if iters % hparams['update_int'] == 0 and iters > 0:
                batch = memory.sample(hparams['batch_size'])
                #update(pol_net, targ_net, env, batch, opt)

            iters += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partitions', type=str, default='[(8,1)]')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--tile_size', type=int, default=3)
    parser.add_argument('--capacity', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--max_exp_epochs', type=int, default=10000)
    parser.add_argument('--min_exp_rate', type=float, default=0.05)
    parser.add_argument('--update_int', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()
    hparams = vars(args)

    try:
        main(hparams)
    except KeyboardInterrupt:
        print('Keyboard escape!')
        check_memory()
