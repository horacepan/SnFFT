import sys
import pdb
import time
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
from utility import load_yor
from cg_utility import cg_loss
from logger import get_logger

sys.path.append('../')
from young_tableau import FerrersDiagram
from utils import check_memory

#log = get_logger(name=__name__)
class FourierPolicyTorch(nn.Module):
    '''
    Wrapper class for fourier linear regression
    '''
    def __init__(self, irreps, yor_prefix, lr, perms):
        '''
        irreps: list of tuples
        yor_prefix: directory containing yor pickles
        lr: learning rate
        perms: list of permutation tuples to cache
        '''
        super(FourierPolicyTorch, self).__init__()
        self.size = sum(irreps[0])
        self.irreps = irreps
        self.yors = {irr: load_yor(irr, yor_prefix) for irr in irreps}
        self.ferrers = {irr: FerrersDiagram(irr) for irr in irreps}

        total_size = sum([f.n_tabs() ** 2 for f in self.ferrers.values()])
        self.w_torch = nn.Parameter(torch.rand(total_size + 1, 1))
        self.w_torch.data.normal_(std=0.2)
        self.w_torch.data[-1] = 5.328571428571428 # TODO: this is a hack
        self.optim = torch.optim.Adam([self.w_torch], lr=lr)
        self.pdict = self.cache_perms(perms)

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

    def nbr_deltas(self, gtups, nbrs_func):
        '''
        gtups: list of tuples to evaluate values of neighbors of
        Returns: torch tensor of shape n x {num_nbrs}
        '''
        gnbrs = []
        len_nbrs = len(nbrs_func(gtups[0]))

        for g in gtups:
            for n in nbrs_func(g):
                gnbrs.append(n)

        y_nbrs = self.forward_dict(gnbrs).reshape(-1, len_nbrs)
        y_pred = self.forward_dict(gtups)
        return y_pred, y_nbrs

    def train_batch(self, perms, y, **kwargs):
        '''
        perms: list of tuples
        y: numpy array
        Returns: training loss on this batch. Also takes a gradient step
        '''
        self.optim.zero_grad()
        y_pred = self.forward_dict(perms)
        loss = nn.functional.mse_loss(y_pred, torch.from_numpy(y).double())
        loss.backward()
        self.optim.step()
        return loss.item()

    def _forward(self, perms):
        '''
        perms; list of tuples
        Returns: the model evaluation on the list of perms (handles the tuple to
        vector representation conversion).
        '''
        X = np.vstack([self.to_irrep(p) for p in perms])
        X_th = torch.from_numpy(X)
        y_pred = X_th.matmul(self.w_torch)
        return y_pred

    def compute_loss(self, perms, y):
        '''
        perms: list of tuples
        y: numpy array
        Returns the MSE loss between the model evaluated on the given perms vs y
        '''
        y_pred = self.forward_dict(perms)
        return nn.functional.mse_loss(y_pred, torch.from_numpy(y).double()).item()

    def __call__(self, gtup):
        '''
        gtup: tuple
        '''
        vec = self.pdict[gtup]
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

    def to_tensor(self, perms):
        '''
        perms: list of tuples
        Returns: the tensor representation of the given permutations
        '''
        X = torch.cat([self.pdict[p] for p in perms], dim=0)
        return X

    def forward_th(self, tensor):
        y_pred = tensor.matmul(self.w_torch)
        return y_pred

    def loss_th(self, tensor, y):
        y_pred = self.forward_th(tensor)
        return nn.functional.mse_loss(y_pred, torch.from_numpy(y).double()).item()

    def cache_perms(self, perms):
        '''
        perms: list of tuples
        Returns a dictionary mapping tuple to its tensor representation
        '''
        pdict = {}
        for p in perms:
            pdict[p] = torch.from_numpy(self.to_irrep(p))
        self.pdict = pdict
        return pdict

    def forward_dict(self, perms):
        X_th = torch.cat([self.pdict[p] for p in perms], dim=0)
        y_pred = X_th.matmul(self.w_torch)
        return y_pred

class FourierPolicyCG(FourierPolicyTorch):
    def __init__(self, irreps, prefix, lr):
        super(FourierPolicyCG, self).__init__(irreps, prefix, lr=lr)
        # load cg matrices
        # load multiplicies

    def train_cg_loss(self, generators):
        self.optim.zero_grad()
        fhats = self._reshaped_mats()
        loss = 0
        for base_p in self.irreps:
            for g in generators:
                loss += cg_loss(base_p, self.irreps, g, fhats)

        loss.backward()
        self.optim.step()
        return loss.item()
