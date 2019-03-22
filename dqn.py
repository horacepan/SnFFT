import sys
sys.path.append('./cube')
from collections import namedtuple
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from complex_utils import cmm, cmse, cmse_real, cmm_sparse
from irrep_env import Cube2IrrepEnv
from utils import check_memory
import numpy as np
import str_cube
import pdb

Batch = namedtuple('Batch', ('state', 'action', 'next_state', 'reward', 'done'))

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
        self.dones = np.empty([capacity, 1], np.dtype('bool'))

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
        self.br = nn.Parameter(torch.rand(1, 1))
        self.bi = nn.Parameter(torch.rand(1, 1))
        self.init_weights()

    def np_forward(self, xr, xi):
        '''
        Given real and imaginary tensors for the state, compute
        '''
        txr = torch.from_numpy(xr)
        txi = torch.from_numpy(xi)
        return self.forward(txr, txi)

    def forward(self, xr, xi):
        xr, xi = cmm(xr, xi, self.wr, self.wi)
        xr     = xr + self.br
        xi     = xi + self.bi
        return xr, xi

    def forward_sparse(self, xr, xi):
        xr, xi = cmm_sparse(xr, xi, self.wr, self.wi)
        xr     = xr + self.br
        xi     = xi + self.bi
        return xr, xi

    def init_weights(self):
        nn.init.xavier_normal_(self.wr)
        nn.init.xavier_normal_(self.wi)
        self.br.data.zero_()
        self.bi.data.zero_()

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

def test_update():
    alpha = (0, 7, 1)
    parts = ((), (6, 1), (1,))
    print('Loading irrep: {} | {}'.format(alpha, parts))
    env = Cube2IrrepEnv(alpha, parts)
    print('Done loading irrep: {} | {}'.format(alpha, parts))
    model = IrrepLinreg(48 * 48)
    c = str_cube.init_2cube()
    batch = [
        [c, 0, c, 1.0],
        [c, 0, c, 1.0],
    ]
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    update(env, model, batch, opt)

def update(env, model, batch, opt, discount=0.9, summary_writer=None):
    '''
    batch: named tuple of stuff
    discount: float
    lossfunc: loss function
    opt: torch optim
    summary_writer: optional
    '''
    lossfunc = cmse_real
    sr = []
    si = []
    nsr = []
    nsi = []

    # this should be fixed
    sir = time.time()
    for s in batch.state:
        xr, xi = env.irrep(s)
        sr.append(xr)
        si.append(xi)
    print('get irrep time: {:3.2f}'.format(time.time() - sir))

    sir = time.time()
    for ns in batch.next_state:
        xr, xi = env.real_imag_irrep_sp(ns)
        nsr.append(xr)
        nsi.append(xi)
    print('get irrep time: {:3.2f}'.format(time.time() - sir))

    sir = time.time()
    reward = torch.from_numpy(batch.reward)#.astype(np.float32))
    sr = torch.cat(sr, dim=0)
    si = torch.cat(si, dim=0)
    nsr = torch.cat(nsr, dim=0)
    nsi = torch.cat(nsi, dim=0)
    print('stacking time: {:3.2f}'.format(time.time() - sir))

    if env.sparse:
        yr_pred, yi_pred = model.forward_sparse(sr, si)
    else:
        yr_pred, yi_pred = model.forward(sr, si)
    print('forward + stacking time: {:3.2f}'.format(time.time() - sir))

    if env.sparse:
        yr_onestep, yi_onestep = model.forward_sparse(nsr, nsi)
    else:
        yr_onestep, yi_onestep = model.forward(nsr, nsi)
    loss = lossfunc(reward + discount * yr_onestep, discount * yi_onestep, yr_pred, yi_pred)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print('forward + stacking time + loss: {:3.2f}'.format(time.time() - sir))
    return loss.item()

def test(hparams):
    start = time.time()
    if hparams['test']:
        alpha = (0, 7, 1)
        parts = ((), (6, 1), (1,))
        model = IrrepLinreg(48 * 48)
    else:
        alpha = (2, 3, 3)
        parts = ((2,), (1, 1, 1), (1, 1, 1))
        model = IrrepLinreg(560 * 560)

    env = Cube2IrrepEnv(alpha, parts, sparse=hparams['sparse'])
    state = env.reset()

    optimizer = torch.optim.SGD(model.parameters(), **hparams['opt_params'])
    memory = ReplayMemory(hparams['mem_size'])
    print('Done setup: {:.2f}s'.format(time.time() - start))
    niter = 0
    nupdates = 1
    losses = np.zeros(hparams['logint'])
    tot_loss = 0
    print('Before training memory check:', end='')
    check_memory()

    start = time.time()
    for e in range(hparams['epochs']):
        state = env.reset()
        for i in range(hparams['max_eplen']):
            # get the opt action
            # stash the thing
            action = get_action(env, model, state)
            # add option to make the irrep optional
            ns, rew, done, _ = env.step(action, irrep=False)
            memory.push(state, action, ns, rew, done)

            if niter > 0 and niter % hparams['update_int'] == 0:
                #print('Updating | niter {} | loss: {:5.2f}'.format(niter, np.mean(losses)))
                sample = memory.sample(hparams['batch_size'])
                sup = time.time()
                _loss = update(env, model, sample, optimizer)
                send = time.time()
                print('update time: {:3.2f}'.format(send - sup))
                tot_loss += _loss
                nupdates = (nupdates + 1)
            niter += 1

        if e % hparams['logint'] == 0:
            elapsed = (time.time() - start) / 60.
            print('Epoch {:6} | Elapsed: {:5.2f}min | nupdates: {} | avg loss: {:4.2f}'.format(e, elapsed, nupdates, tot_loss / nupdates))
    check_memory()

if __name__ == '__main__':
    hparams = {
        'mem_size': 1000, 
        'epochs': 1000,
        'max_eplen': 20,
        'batch_size': 256,
        'capacity': 100000,
        'update_int': 100,
        'logint': 1000,
        'discount': 0.99,
        'opt_params': {
            'lr': 0.1
        },
        'test': False,
        #'test': True,
        'sparse': True
        #'sparse': False
    }
    print('Testing with sparse: {}'.format(hparams['sparse']))
    try:
        test(hparams)
    except KeyboardInterrupt:
        print('Keyboard escape!')
        check_memory()
