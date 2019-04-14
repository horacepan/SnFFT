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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from complex_utils import cmm, cmse, cmse_min_imag, cmse_real, cmm_sparse
from irrep_env import Cube2IrrepEnv
from utils import check_memory
import str_cube
from tensorboardX import SummaryWriter

Batch = namedtuple('Batch', ('state', 'action', 'next_state', 'reward', 'done'))
CUBE2_SIZE = 40320 * (3**7)

def get_logger(fname):
    str_fmt = '[%(asctime)s.%(msecs)03d] %(levelname)s %(module)s: %(message)s'
    date_fmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        filename=fname,
        level=logging.DEBUG,
        format=str_fmt,
        datefmt=date_fmt)

    logger = logging.getLogger(__name__)
    sh = logging.StreamHandler()
    formatter = logging.Formatter(str_fmt, datefmt=date_fmt)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

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
        self.br = nn.Parameter(torch.zeros(1))
        self.bi = nn.Parameter(torch.zeros(1))
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
        xr = self.br + xr
        xi = self.bi + xi
        return xr, xi

    def zero_weights(self):
        self.wr.data.zero_()
        self.wi.data.zero_()
        self.bi.data.zero_()
        self.br.data.zero_()

    def init(self, mode):
        if mode == 'normal':
            self.normal_init()
        elif mode == 'uniform':
            self.uniform_init()
        elif mode == 'binary':
            scale = 0.01
            self.wr.data = ((torch.rand(self.wr.size()) > 0.5).float() * scale) - (scale / 2)
            self.wi.data = ((torch.rand(self.wr.size()) > 0.5).float() * scale) - (scale / 2)
        else:
            self.zero_weights()

    def normal_init(self):
        nn.init.normal_(self.wr, 0, 0.01)
        nn.init.normal_(self.wi, 0, 0.01)

    def uniform_init(self):
        nn.init.uniform_(self.wr, -0.01, 0.01)
        nn.init.uniform_(self.wi, -0.01, 0.01)

    def binary_init(self):
        pass

    def loadnp(self, fname):
        mat = np.load(fname)
        mat_re = mat.real.astype(np.float32)
        mat_im = mat.imag.astype(np.float32)
        size = mat_re.shape[0]
        self.wr.data = torch.from_numpy(mat_re.reshape(size*size, 1)) * (size / CUBE2_SIZE) * -1
        self.wi.data = torch.from_numpy(mat_im.reshape(size*size, 1)) * (size / CUBE2_SIZE) * -1

def value(model, env, state):
    xr, xi = env.irrep(state)
    yr, yi = model.forward(xr, xi)
    return yr.item(), yi.item()

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

def update(env, pol_net, targ_net, batch, opt, hparams, logger, nupdate):
    '''
    batch: named tuple of stuff
    discount: float
    lossfunc: loss function
    opt: torch optim
    '''
    if hparams['lossfunc'] == 'cmse_real':
        lossfunc = cmse_real
    elif hparams['lossfunc'] == 'cmse_min_imag':
        lossfunc = cmse_min_imag
    else:
        lossfunc = cmse
    discount = hparams['discount']
    sr = []
    si = []
    nsr = []
    nsi = []

    for s in batch.state:
        _xr, _xi = env.irrep(s)
        sr.append(_xr)
        si.append(_xi)

    for ns in batch.next_state:
        xr, xi = env.irrep(ns)
        nsr.append(xr)
        nsi.append(xi)

    reward = torch.from_numpy(batch.reward)#.astype(np.float32))
    dones = torch.from_numpy(batch.done)
    sr = torch.cat(sr, dim=0)
    si = torch.cat(si, dim=0)
    nsr = torch.cat(nsr, dim=0)
    nsi = torch.cat(nsi, dim=0)

    yr_pred, yi_pred = pol_net.forward_sparse(sr, si)
    yr_onestep, yi_onestep = targ_net.forward_sparse(nsr, nsi)

    opt.zero_grad()
    loss = lossfunc(reward + discount * yr_onestep.detach() * (1 - dones),
                             discount * yi_onestep.detach() * (1 - dones), yr_pred, yi_pred)

    loss.backward()
    old_wr = pol_net.wr.detach().clone()
    old_wi = pol_net.wi.detach().clone()
    opt.step()
    # log the size of the update?
    logger.add_scalar('wr_update_norm', (pol_net.wr - old_wr).norm().item(), nupdate)
    logger.add_scalar('wi_update_norm', (pol_net.wi - old_wi).norm().item(), nupdate)
    return loss.item()

def explore_rate(epoch_num, explore_epochs, eps_min):
    return max(eps_min, 1 - (epoch_num / (1 + explore_epochs)))

def get_logdir(logdir, saveprefix):
    '''
    Produce a unique directory name
    '''
    cnt = 0
    while os.path.exists(os.path.join(logdir, saveprefix + '_' + str(cnt))):
        cnt += 1

    return os.path.join(logdir, saveprefix + '_' + str(cnt))
