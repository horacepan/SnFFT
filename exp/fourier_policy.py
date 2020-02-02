import pdb
import time
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
from utility import load_yor, load_np, S8_GENERATORS, nbrs
from cg_utility import cg_loss
from logger import get_logger

#log = get_logger(name=__name__)
class FourierPolicy:
    '''
    Wrapper class for fourier linear regression
    '''
    def __init__(self, irreps, prefix):
        '''
        Setup
        '''
        self.size = sum(irreps[0])
        self.irreps = irreps
        self.yors = {irr: load_yor(irr, prefix) for irr in irreps}

        total_size = 0
        for parts, pdict in self.yors.items():
            # sort of hacky but w/e
            # want to grab the size of this thing
            total_size += pdict[(1, 2, 3, 4, 5, 6, 7, 8)].shape[0] ** 2

        self.w = np.random.normal(scale=0.2, size=(total_size + 1, 1))

    def train_batch(self, perms, y, lr, **kwargs):
        X = np.vstack([self.to_irrep(p) for p in perms])
        y_pred = X@self.w
        grad = X.T@y_pred - X.T@y
        self.w -= lr * grad
        loss = np.mean(np.square((y - y_pred)))
        return loss

    def eval_batch(self, perms, y):
        y_pred = self.forward(perms)
        return np.mean(np.square(y - y_pred))

    def __call__(self, gtup):
        '''
        gtup: perm tuple
        '''
        vec = self.to_irrep(gtup)
        return vec.dot(self.w)

    def to_irrep(self, gtup):
        '''
        Loop over irreps -> cat the irrep (via the yor dicts) -> reshape
        gtup: perm tuple
        '''
        irrep_vecs = []
        for irr in self.irreps:
            rep = self.yors[irr][gtup]
            dim = rep.shape[0]
            vec = rep.reshape(1, -1) * np.sqrt(dim)
            irrep_vecs.append(vec)
        irrep_vecs.append(np.array([[1]]))
        return np.hstack(irrep_vecs)

    def forward(self, perms):
        X = np.vstack([self.to_irrep(p) for p in perms])
        y_pred = X@self.w
        return y_pred

    def eval(self, gtups):
        vecs = np.hstack([self.to_irrep(g) for g in gtups])
        return vecs.dot(self.w)

    def set_w(self, fhat_prefix):
        self.fhats = {irr: load_np(irr, fhat_prefix) for irr in self.irreps}
        group_size = 40320
        mean = 5.328571428571428
        vecs = []
        for irr in self.irreps:
            mat = self.fhats[irr]
            coef =  (mat.shape[1] ** 0.5)/ group_size
            vecs.append(coef * mat.reshape(-1, 1))
        self.w = np.vstack(vecs + [[1]])
        self.w[-1] = mean

    def nbr_deltas(self, gtups):
        gnbrs = []
        len_nbrs = len(nbrs(gtups[0]))

        for g in gtups:
            for n in nbrs(g):
                gnbrs.append(n)

        y_nbrs = self.forward(gnbrs).reshape(-1, len_nbrs)
        y_pred = self.forward(gtups)
        return y_pred, y_nbrs

class FourierPolicyTorch(FourierPolicy):
    def __init__(self, irreps, prefix, lr):
        super(FourierPolicyTorch, self).__init__(irreps, prefix)
        self.w_torch = nn.Parameter(torch.from_numpy(self.w))
        self.optim = torch.optim.Adam([self.w_torch], lr=lr)

    def train_batch(self, perms, y, **kwargs):
        st = time.time()
        self.optim.zero_grad()
        y_pred = self.forward(perms)
        loss = nn.functional.mse_loss(y_pred, torch.from_numpy(y).double())
        loss.backward()
        self.optim.step()
        return loss.item()

    def forward(self, perms):
        X = np.vstack([self.to_irrep(p) for p in perms])
        X_th = torch.from_numpy(X)
        y_pred = X_th.matmul(self.w_torch)
        return y_pred

    def compute_loss(self, perms, y):
        y_pred = self.forward(perms)
        return nn.functional.mse_loss(y_pred, torch.from_numpy(y).double()).item()

    def __call__(self, gtup):
        vec = torch.from_numpy(self.to_irrep(gtup))
        return vec.matmul(self.w_torch)

    def _reshaped_mats(self):
        ident_tup = tuple(i for i in range(1, self.size+1))
        idx = 0
        fhats = {}

        for irr in self.irreps:
            size = self.yors[irr][ident_tup].shape[0]
            mat = self.w_torch[idx: idx + (size * size)]
            fhats[irr] = mat.reshape(size, size)
            idx += size * size
        return fhats

class FourierPolicyCG(FourierPolicyTorch):
    def __init__(self, irreps, prefix, lr):
        super(FourierPolicyCG, self).__init__(irreps, prefix, lr=lr)
        self.minibatch_cnt = 0

    def train_cg_loss(self, generators):
        st = time.time()
        self.optim.zero_grad()
        fhats = self._reshaped_mats()
        loss = 0
        for base_p in self.irreps:
            for g in generators:
                loss += cg_loss(base_p, self.irreps, g, fhats)

        loss.backward()
        self.optim.step()
        return loss.item()
