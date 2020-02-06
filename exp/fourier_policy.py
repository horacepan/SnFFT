import sys
import pdb
import time
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
from utility import load_yor, cg_mat, S8_GENERATORS, th_kron
from cg_utility import proj, compute_rhs_block, compute_reduced_block
from logger import get_logger

sys.path.append('../')
from utils import check_memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FourierPolicyTorch(nn.Module):
    '''
    Wrapper class for fourier linear regression
    '''
    def __init__(self, irreps, yor_prefix, lr, perms, yors=None, pdict=None):
        '''
        irreps: list of tuples
        yor_prefix: directory containing yor pickles
        lr: learning rate
        perms: list of permutation tuples to cache
        '''
        super(FourierPolicyTorch, self).__init__()
        self.size = sum(irreps[0])
        self.irreps = irreps
        if yors is None:
            self.yors = {irr: load_yor(irr, yor_prefix) for irr in irreps}
        else:
            self.yors = yors
        self.irrep_sizes = {irr: self.yors[irr][tuple(range(1, self.size+1))].shape[0] for irr in irreps}

        total_size = sum([d ** 2 for d in self.irrep_sizes.values()])
        self.w_torch = nn.Parameter(torch.rand(total_size + 1, 1))
        self.w_torch.data.normal_(std=0.2)
        self.w_torch.data[-1] = 5.328571428571428 # TODO: this is a hack

        if yors is None:
            self.optim = torch.optim.Adam([self.w_torch], lr=lr)
        # this is pretty hacky
        if pdict is None:
            self.pdict = self.cache_perms(perms)
        else:
            self.pdict = pdict
        self.lr = lr

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

        y_nbrs = self.forward_tup(gnbrs).reshape(-1, len_nbrs)
        y_pred = self.forward_tup(gtups)
        return y_pred, y_nbrs

    def train_batch(self, perms, y, **kwargs):
        '''
        perms: list of tuples
        y: numpy array
        Returns: training loss on this batch. Also takes a gradient step
        '''
        self.optim.zero_grad()
        y_pred = self.forward_tup(perms)
        y = torch.from_numpy(y).float().reshape(y_pred.shape).to(device)
        loss = nn.functional.mse_loss(y_pred, y)
        loss.backward()
        self.optim.step()
        return loss.item()

    def compute_loss(self, perms, y):
        '''
        perms: list of tuples
        y: numpy array
        Returns the MSE loss between the model evaluated on the given perms vs y
        '''
        with torch.no_grad():
            y_pred = self.forward_tup(perms)
            y = torch.from_numpy(y).float().reshape(y_pred.shape).to(device)
            return nn.functional.mse_loss(y_pred, y).item()

    def __call__(self, gtup):
        '''
        gtup: tuple
        '''
        vec = self.pdict[gtup].to(device)
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

    def forward(self, tensor):
        y_pred = tensor.matmul(self.w_torch)
        return y_pred

    def eval_opt_nbr(self, tups, nnbrs):
        # eval all nbrs, returns tensor of the value of the opt neighbor
        nbr_eval = self.forward_tup(tups).reshape(-1, nnbrs)
        max_nbr_vals = nbr_eval.max(dim=1, keepdim=True)[0]
        return max_nbr_vals

    def cache_perms(self, perms):
        '''
        perms: list of tuples
        Returns a dictionary mapping tuple to its tensor representation
        '''
        pdict = {}
        for p in perms:
            pdict[p] = torch.from_numpy(self.to_irrep(p)).float()
        self.pdict = pdict
        return pdict

    def forward_tup(self, perms):
        X_th = self.to_tensor(perms).to(device)
        y_pred = X_th.matmul(self.w_torch)
        return y_pred

    def load(self, fname):
        weight = torch.load(fname)
        self.w_torch = nn.Parameter(weight)
        self.optim = torch.optim.Adam(self.parameters(), lr=self.lr)

class FourierPolicyCG(FourierPolicyTorch):
    def __init__(self, irreps, prefix, lr, perms, yors=None, pdict=None):
        super(FourierPolicyCG, self).__init__(irreps, prefix, lr, perms, yors, pdict)
        self.s8_chars = pickle.load(open(f'{prefix}/char_dict.pkl', 'rb'))
        self.cg_mats = {(p1, p2, base_p): torch.from_numpy(cg_mat(p1, p2, base_p)).float().to(device)
                         for p1 in irreps for p2 in irreps for base_p in irreps}
        self.multiplicities = self.compute_multiplicities()

        self.generators = S8_GENERATORS

    def compute_multiplicities(self):
        mults_dict = {}
        for p1 in self.irreps:
            for p2 in self.irreps:
                tens_char = self.s8_chars[p1] * self.s8_chars[p2]
                mults_dict[(p1, p2)] = proj(tens_char, self.s8_chars)
        return mults_dict

    def train_cg_loss_cached(self):
        self.optim.zero_grad()
        fhats = self._reshaped_mats()
        loss = 0
        kron_mats = {(p1, p2): th_kron(fhats[p1], fhats[p2]) for p1 in self.irreps for p2 in self.irreps}

        for base_p in self.irreps:
            for g in self.generators:
                loss += self.cg_loss(base_p, g, fhats, kron_mats)

        loss.backward()
        self.optim.step()
        return loss.item()

    def eval_cg_loss(self):
        with torch.no_grad():
            fhats = self._reshaped_mats()
            loss = 0
            kron_mats = {(p1, p2): th_kron(fhats[p1], fhats[p2]) for p1 in self.irreps for p2 in self.irreps}

            for base_p in self.irreps:
                for g in self.generators:
                    loss += self.cg_loss(base_p, g, fhats, kron_mats)

            return loss.item()

    def cg_loss(self, base_p, gelement, fhats, kron_mats):
        '''
        base_p: partition
        gelement: tuple of a generator of the group
        fhats: torch tensors
        '''

        g_size = 1 # should really be size of the group but doesnt matter much
        pdim = self.irrep_sizes[base_p]
        rho1 = torch.from_numpy(self.yors[base_p][gelement]).float().to(device)
        lmat = rho1 + torch.eye(pdim).to(device)

        lhs = 0
        rhs = 0

        for p1 in self.irreps:
            p1_dim = self.irrep_sizes[p1]
            fhat1 = fhats[p1]
            rhog = torch.from_numpy(self.yors[p1][gelement]).float().to(device)

            for p2 in self.irreps:
                p2_dim = self.irrep_sizes[p2]
                mult = self.multiplicities[(p1, p2)][base_p]

                fhat2 = fhats[p2]
                cgmat = self.cg_mats[(p1, p2, base_p)] # d x drho zrho
                reduced_block = compute_reduced_block(p1, p2, cgmat, mult, kron_mats)

                coeff = (p1_dim * p2_dim) / pdim
                lhs += coeff * lmat @ reduced_block
                rhs += 2 * coeff * compute_rhs_block(fhat1, fhat2, cgmat, mult, rhog)

        return ((lhs - rhs).pow(2)).mean()

    def opt_move(self, x):
        output = self.forward(x)
        return output.argmax()

    def opt_move_tup(self, tup_nbrs):
        tens_nbrs = self.to_tensor(tup_nbrs).to(device)
        return self.opt_move(tens_nbrs)
