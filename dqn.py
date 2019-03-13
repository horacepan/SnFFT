import sys
sys.path.append('./cube')
from collections import namedtuple
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from complex_utils import cmm, cmse, cmse_real
from irrep_env import Cube2IrrepEnv
import numpy as np
import str_cube
import pdb

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        print('sampling | memory size is: {} | batch is: {}'.format(len(self), batch_size))
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

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

    def init_weights(self):
        nn.init.xavier_normal_(self.wr)
        nn.init.xavier_normal_(self.wi)
        self.br.data.zero_()
        self.bi.data.zero_()

def get_action(env, model, state):
    '''
    model: nn.Module
    state: string of cube state
    '''
    start = time.time()
    neighbors = str_cube.neighbors_fixed_core(state)[:6] # do the symmetry modded out version for now
    nbr_irreps = np.stack([env.convert_irrep(n) for n in neighbors])
    xr = nbr_irreps.real.astype(np.float32)
    xi = nbr_irreps.imag.astype(np.float32)
    yr, yi = model.np_forward(xr, xi)
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

    for s in [x[0] for x in batch]:
        xr, xi = env.real_imag_irrep(s)
        sr.append(xr)
        si.append(xi) 

    for ns in [x[2] for x in batch]:
        xr, xi = env.real_imag_irrep(ns)
        nsr.append(xr)
        nsi.append(xi) 

    rewards = [x[3] for x in batch]
    reward = torch.FloatTensor(rewards)
    sr = torch.stack(sr, dim=0)
    si = torch.stack(si, dim=0)
    nsr = torch.stack(nsr, dim=0)
    nsi = torch.stack(nsi, dim=0)
    print('stacked shape:', sr.shape)
    yr_pred, yi_pred = model.forward(sr, si)
    yr_onestep, yi_onestep = model.forward(nsr, nsi)
    loss = lossfunc(reward + discount * yr_onestep, discount * yi_onestep, yr_pred, yi_pred)

    opt.zero_grad()
    loss.backward()
    opt.step()


def test(hparams):
    start = time.time()
    #alpha = (2, 3, 3)
    #parts = ((2,), (1, 1, 1), (1, 1, 1))
    alpha = (0, 7, 1)
    parts = ((), (6, 1), (1,))

    env = Cube2IrrepEnv(alpha, parts)
    state, _ = env.reset()

    #model = IrrepLinreg(560 * 560)
    model = IrrepLinreg(48 * 48)
    optimizer = torch.optim.SGD(model.parameters(), **hparams['opt_params'])
    memory = ReplayMemory(hparams['mem_size'])
    print('Done setup: {:.2f}s'.format(time.time() - start))
    niter = 0

    for e in range(hparams['epochs']):
        state, irrep = env.reset()
        for i in range(hparams['max_eplen']):
            # get the opt action
            # stash the thing
            action = get_action(env, model, state)
            # add option to make the irrep optional
            ns, rew, done, _ = env.step(action, irrep=False)
            memory.push(state, action, ns, rew)

            if niter > 0 and niter % hparams['update_int'] == 0:
                print('Updating | niter {}'.format(niter))
                sample = memory.sample(hparams['batch_size'])
                update(env, model, sample, optimizer)
            niter += 1

if __name__ == '__main__':
    hparams = {
        'mem_size': 1000, 
        'epochs': 10,
        'max_eplen': 20,
        'batch_size': 10,
        'capacity': 10000,
        'update_int': 100,
        'discount': 0.99,
        'opt_params': {
            'lr': 0.001
        }
    }
    test(hparams)
    #test_env(hparams)
