import os
import logging
import time
import random
import argparse
from collections import namedtuple
import json
import pdb
import sys
sys.path.append('./cube')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from complex_utils import cmm, cmse_real, cmm_sparse
from irrep_env import Cube2IrrepEnv
from utils import check_memory
import str_cube

Batch = namedtuple('Batch', ('state', 'action', 'next_state', 'reward', 'done'))
CUBE2_SIZE = 40320 * (3**7)
NP_TOP_IRREP_LOC = '/local/hopan/cube/fourier_unmod/(2, 3, 3)/((2,), (1, 1, 1), (1, 1, 1)).npy'

def get_logger(fname):
    logging.basicConfig(
        filename=fname,
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s %(funcName)s: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    return logger

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.filled = 0

        self.states = np.empty([capacity], np.dtype('<U24'))
        self.actions = np.empty([capacity, 1], np.dtype('i1'))
        self.new_states = np.empty([capacity], np.dtype('<U24'))
        self.rewards = np.empty([capacity, 1], dtype=np.float32)
        self.dones = np.empty([capacity, 1], np.dtype(np.float32))

    def push(self, state, action, new_state, reward, done):
        self.states[self.position]      = state
        self.actions[self.position]     = action
        self.new_states[self.position]  = new_state
        self.rewards[self.position]     = reward
        self.dones[self.position]       = done

        self.position = (self.position + 1) % self.capacity
        self.filled = min(self.filled + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.filled, batch_size)
        state = self.states[idx]
        new_state = self.new_states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        done = self.dones[idx]
        return Batch(state, action, new_state, reward, done)

    def __len__(self):
        return self.filled

class IrrepLinreg(nn.Module):
    '''
    This is a simple linear regression. Input xs will be in
    the fourier basis.
    '''
    def __init__(self, n_in):
        super(IrrepLinreg, self).__init__()
        self.wr = nn.Parameter(torch.rand(n_in, 1))
        self.wi = nn.Parameter(torch.rand(n_in, 1))
        self.zero_weights()

    def np_forward(self, xr, xi):
        '''
        Given real and imaginary tensors for the state, compute
        '''
        txr = torch.from_numpy(xr)
        txi = torch.from_numpy(xi)
        return self.forward(txr, txi)

    def forward(self, xr, xi):
        xr, xi = cmm(xr, xi, self.wr, self.wi)
        return xr, xi

    def forward_sparse(self, xr, xi):
        xr, xi = cmm_sparse(xr, xi, self.wr, self.wi)
        return xr, xi

    def zero_weights(self):
        self.wr.data.zero_()
        self.wi.data.zero_()

    def init_weights(self):
        nn.init.xavier_normal_(self.wr)
        nn.init.xavier_normal_(self.wi)

    def loadnp(self, fname):
        mat = np.load(fname)
        mat_re = mat.real.astype(np.float32)
        mat_im = mat.imag.astype(np.float32)
        size = mat_re.shape[0]
        self.wr.data = torch.from_numpy(mat_re.reshape(size*size, 1)) * (size / CUBE2_SIZE) * -1
        self.wi.data = torch.from_numpy(mat_im.reshape(size*size, 1)) * (size / CUBE2_SIZE) * -1

def get_action(env, model, state):
    if env.sparse:
        return get_action_th(env, model, state)
    else:
        return get_action_np(env, model, state)

def get_action_np(env, model, state):
    '''
    model: nn.Module
    state: string of cube state
    '''
    neighbors = str_cube.neighbors_fixed_core(state)[:6] # do the symmetry modded out version for now
    nbr_irreps = np.stack([env.convert_irrep_np(n) for n in neighbors])
    xr = nbr_irreps.real
    xi = nbr_irreps.imag
    yr, yi = model.np_forward(xr, xi)
    return yr.argmax().item()

def get_action_th(env, model, state):
    neighbors = str_cube.neighbors_fixed_core(state)[:6] # do the symmetry modded out version for now
    xr, xi = zip(*[env.irrep(n) for n in neighbors])
    xr = torch.cat(xr, dim=0)
    xi = torch.cat(xi, dim=0)
    if env.sparse:
        yr, yi = model.forward_sparse(xr, xi)
    else:
        yr, yi = model.forward(xr, xi)

    return yr.argmax().item()

def update(env, model, batch, opt, discount=0.9):
    '''
    batch: named tuple of stuff
    discount: float
    lossfunc: loss function
    opt: torch optim
    '''
    lossfunc = cmse_real
    sr = []
    si = []
    nsr = []
    nsi = []

    for s in batch.state:
        xr, xi = env.irrep(s)
        sr.append(xr)
        si.append(xi)

    for ns in batch.next_state:
        xr, xi = env.real_imag_irrep_sp(ns)
        nsr.append(xr)
        nsi.append(xi)

    reward = torch.from_numpy(batch.reward)#.astype(np.float32))
    dones = torch.from_numpy(batch.done)
    sr = torch.cat(sr, dim=0)
    si = torch.cat(si, dim=0)
    nsr = torch.cat(nsr, dim=0)
    nsi = torch.cat(nsi, dim=0)

    yr_pred, yi_pred = model.forward_sparse(sr, si)
    yr_onestep, yi_onestep = model.forward_sparse(nsr, nsi)
    loss = lossfunc(reward + discount * yr_onestep * (1 - dones),
                    discount * yi_onestep * (1 - dones), yr_pred, yi_pred)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()

def explore_rate(epoch_num, explore_epochs, eps_min):
    return max(eps_min, 1 - (epoch_num / explore_epochs))

def get_logdir(logdir, saveprefix):
    '''
    Produce a unique directory name
    '''
    cnt = 0
    while os.path.exists(os.path.join(logdir, saveprefix + str(cnt))):
        cnt += 1

    return os.path.join(logdir, saveprefix + '_' + str(cnt))

def main(hparams):
    logfname = get_logdir(hparams['logdir'], hparams['savename'])
    if not os.path.exists(hparams['logdir']):
        os.makedirs(hparams['logdir'])
    savedir = get_logdir(hparams['logdir'], hparams['savename'])
    logfile = os.path.join(savedir, 'log.txt')

    os.makedirs(savedir)
    with open(os.path.join(savedir, 'args.json'), 'w') as f:
        json.dump(hparams, f, indent=4)

    log = get_logger(logfile)
    log.debug('Starting main')
    log.debug('hparams: {}'.format(hparams))

    torch.manual_seed(hparams['seed'])
    random.seed(hparams['seed'])
    log.debug('first log!')
    if hparams['test']:
        alpha = (0, 7, 1)
        parts = ((), (6, 1), (1,))
        model = IrrepLinreg(48 * 48)
    else:
        alpha = (2, 3, 3)
        parts = ((2,), (1, 1, 1), (1, 1, 1))
        model = IrrepLinreg(560 * 560)

        if hparams['loadnp']:
            model.loadnp(hparams['loadnp'])

    env = Cube2IrrepEnv(alpha, parts)
    optimizer = torch.optim.SGD(model.parameters(), lr=hparams['lr'], momentum=hparams['momentum'])
    memory = ReplayMemory(hparams['capacity'])
    niter = 0
    nupdates = 0
    tot_loss = 0
    nsolved = 0
    rewards = np.zeros(hparams['logint'])
    seen_states = set()

    for e in range(hparams['epochs']):
        state = env.reset_fixed(max_dist=hparams['max_dist'])
        epoch_rews = 0

        for i in range(hparams['max_dist'] + 5):
            if random.random() < explore_rate(e, hparams['epochs'] * hparams['explore_proportion'], hparams['eps_min']):
                action = random.randint(0, env.action_space.n - 1)
            else:
                action = get_action(env, model, state)

            seen_states.add(state)
            ns, rew, done, _ = env.step(action, irrep=False)
            memory.push(state, action, ns, rew, done)
            epoch_rews += rew
            state = ns
            niter += 1

            if niter > 0 and niter % hparams['update_int'] == 0:
                sample = memory.sample(hparams['batch_size'])
                _loss = update(env, model, sample, optimizer)
                tot_loss += _loss
                nupdates = (nupdates + 1)

            if done:
                nsolved += 1
                break

        rewards[e%len(rewards)] = epoch_rews
        if e % hparams['logint'] == 0 and e > 0:
            log.info('Epoch {:7} | avg rew: {:4.2f} | reset dist: {:3} | solved: {:.3f} | explore: {:.2f} | rez: {} | imz: {}'.format(
                e, np.mean(rewards), hparams['max_dist'], nsolved / hparams['logint'],
                explore_rate(e, hparams['epochs'] * hparams['explore_proportion'], hparams['eps_min']),
                torch.nonzero(model.wr.data).numel(),
                torch.nonzero(model.wi.data).numel()
            ))
            nsolved = 0

    log.info('Total updates: {}'.format(nupdates))
    torch.save(model, os.path.join(savedir, 'model.pt'))
    check_memory()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_dist', type=int, default=15)
    parser.add_argument('--explore_proportion', type=float, default=0.2)
    parser.add_argument('--eps_min', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--capacity', type=int, default=10000)
    parser.add_argument('--update_int', type=int, default=50)
    parser.add_argument('--logint', type=int, default=1000)
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--savename', type=str, default='randomlog')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--loadnp', type=str, default=NP_TOP_IRREP_LOC)
    parser.add_argument('--logdir', type=str, default='./logs/')
    args = parser.parse_args()
    hparams = vars(args)

    try:
        main(hparams)
    except KeyboardInterrupt:
        print('Keyboard escape!')
        check_memory()
