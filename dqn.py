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
from io_utils import get_prefix
import str_cube
from tensorboardX import SummaryWriter

CUBE2_SIZE = 40320 * (3**7)

def get_logger(fname=None, stdout=True, tofile=True):
    '''
    fname: file location to store the log file
    '''
    handlers = []
    if stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers.append(stdout_handler)
    if tofile:
        file_handler = logging.FileHandler(filename=fname)
        handlers.append(file_handler)

    str_fmt = '[%(asctime)s.%(msecs)03d] %(message)s'
    date_fmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.DEBUG,
        format=str_fmt,
        datefmt=date_fmt,
        handlers=handlers
    )

    logger = logging.getLogger(__name__)
    return logger

class IrrepLinregNP:
    def __init__(self, fourier_loc):
        self.weight = np.load(fourier_loc)
        self.weight = -(self.weight.shape[0] / CUBE2_SIZE) * self.weight

    def forward(self, irrep_mat):
        res = np.sum(irrep_mat * self.weight)
        return res.real, res.imag

    def forward_t(self, irrep_mat):
        res = np.sum(irrep_mat.T * self.weight)
        return res.real, res.imag

class IrrepLinregSp(nn.Module):
    def __init__(self, n_in):
        super(IrrepLinreg, self).__init__()
        self.wr = nn.Parameter(torch.rand(n_in, 1))
        self.wi = nn.Parameter(torch.rand(n_in, 1))
        self.br = nn.Parameter(torch.zeros(1))
        self.bi = nn.Parameter(torch.zeros(1))
        self.zero_weights()

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

    @staticmethod
    def from_np(alpha, parts):
        np_mat = os.path.join(get_prefix(), 'cube', 'fourier_unmod', str(alpha), str(parts) + '.npy')
        mat = np.load(np_mat)
        size = mat.shape[0]
        model = IrrepLinreg(size * size)
        model.setnp(mat)
        return model

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

    def normal_init(self, std):
        nn.init.normal_(self.wr, 0, std)
        nn.init.normal_(self.wi, 0, std)

    def uniform_init(self):
        nn.init.uniform_(self.wr, -0.01, 0.01)
        nn.init.uniform_(self.wi, -0.01, 0.01)

    def binary_init(self):
        pass

    def loadnp(self, fname, transpose=True):
        mat = np.load(fname)
        if transpose:
            mat_re = mat.real.astype(np.float32).T
            mat_im = mat.imag.astype(np.float32).T
        else:
            mat_re = mat.real.astype(np.float32)
            mat_im = mat.imag.astype(np.float32)

        size = mat_re.shape[0]
        self.wr.data = torch.from_numpy(mat_re.reshape(size*size, 1)) * (size / CUBE2_SIZE) * -1
        self.wi.data = torch.from_numpy(mat_im.reshape(size*size, 1)) * (size / CUBE2_SIZE) * -1

    def setnp(self, mat):
        #print('Doing transpose in setnp')
        mat_re = mat.real.astype(np.float32).T
        mat_im = mat.imag.astype(np.float32).T
        size = mat_re.shape[0]
        self.wr.data = torch.from_numpy(mat_re.reshape(size*size, 1)) * (size / CUBE2_SIZE) * -1
        self.wi.data = torch.from_numpy(mat_im.reshape(size*size, 1)) * (size / CUBE2_SIZE) * -1

    def add_gaussian_noise(self, mu, std):
        _mu = torch.zeros(self.wr.size()) + mu
        _std = torch.zeros(self.wr.size()) + std
        r_noise = torch.normal(_mu, _std)
        i_noise = torch.normal(_mu, _std)
        self.wr.data.add_(r_noise)
        self.wi.data.add_(i_noise)

def value(model, env, state):
    xr, xi = env.irrep_inv(state)
    yr, yi = model.forward(xr, xi)
    return yr.item(), yi.item()

def value_tup(model, env, otup, ptup):
    xr, xi = env.tup_irrep_inv(otup, ptup)
    yr, yi = model.forward(xr, xi)
    return yr.item(), yi.item()

def value_inv(model, env, state):
    xr, xi = env.irrep(state)
    yr, yi = model.forward(xr, xi)
    return yr.item(), yi.item()

def value_tup_np(np_model, env, otup, ptup):
    '''
    np_model: IrrepLinregNP object
    env: Cube2Irrep
    otup: tuple of orientation
    ptup: tuple of permutation
    '''
    mat = env.tup_irrep_inv_np(otup, ptup)
    yr, yi = np_model.forward(mat)
    return yr, yi

def value_tup_np_t(np_model, env, otup, ptup):
    '''
    np_model: IrrepLinregNP object
    env: Cube2Irrep
    otup: tuple of orientation
    ptup: tuple of permutation
    '''
    mat = env.tup_irrep_inv_np(otup, ptup)
    yr, yi = np_model.forward_t(mat)
    return yr, yi

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
    neighbors = str_cube.neighbors_fixed_core_small(state) # do the symmetry modded out version for now
    nbr_irreps = np.stack([env.irrep_np(n) for n in neighbors])
    xr = nbr_irreps.real
    xi = nbr_irreps.imag
    yr, yi = model.np_forward(xr, xi)
    return yr.argmax().item()

def get_action_th(env, model, state):
    neighbors = str_cube.neighbors_fixed_core_small(state) # do the symmetry modded out version for now
    xr, xi = env.encode_state(neighbors)

    if env.sparse:
        yr, yi = model.forward_sparse(xr, xi)
    else:
        yr, yi = model.forward(xr, xi)

    return yr.argmax().item()

def correct_move(env, model, state):
    action = get_action(env, model, state)
    neighbors = str_cube.neighbors_fixed_core_small(state) # do the symmetry modded out version for now
    nbr_dists = [env.distance(n) for n in neighbors]
    min_dist = min(nbr_dists)
    return  nbr_dists[action] == min_dist

def train_batch(env, pol_net, targ_net, batch, opt, hparams, logger, nupdate):
    '''
    batch: named tuple of stuff
    discount: float
    lossfunc: loss function
    opt: torch optim
    '''
    opt.zero_grad()
    if hparams['lossfunc'] == 'cmse_real':
        lossfunc = cmse_real
    elif hparams['lossfunc'] == 'cmse_min_imag':
        lossfunc = cmse_min_imag
    else:
        lossfunc = cmse
    discount = hparams['discount']

    reward = torch.from_numpy(batch.reward)#.astype(np.float32))
    dones = torch.from_numpy(batch.done)
    sr, si = env.encode_state(batch.state)
    yr_pred, yi_pred = pol_net.forward_sparse(sr, si)

    nbrs = [n for s in batch.state for n in str_cube.neighbors_fixed_core_small(s)]
    nsr, nsi = env.encode_state(nbrs)

    if hparams['usetarget']:
        yr_onestep, yi_onestep = targ_net.forward_sparse(nsr, nsi)
    else:
        yr_onestep, yi_onestep = pol_net.forward_sparse(nsr, nsi)
    yr_onestep = yr_onestep.reshape(-1, env.actions)
    yi_onestep = yi_onestep.reshape(-1, env.actions)
    yr_next, opt_idx = yr_onestep.max(dim=1, keepdim=True)
    yi_next = yi_onestep.gather(1, opt_idx)

    loss = lossfunc((reward * (-dones)) + discount * yr_next.detach(),
                    discount * yi_next.detach(), yr_pred, yi_pred)
    loss.backward()
    opt.step()
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
