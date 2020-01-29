import pdb
from tqdm import tqdm
import time
import os
import pickle
import numpy as np

import torch
import torch.nn as nn

def load_yor(irrep, prefix):
    '''
    Loads the given irreps
    '''
    fname = os.path.join(prefix, '_'.join(str(i) for i in irrep) + '.pkl')
    pkl = pickle.load(open(fname, 'rb'))
    return pkl

def load_np(irrep, prefix):
    fname = os.path.join(prefix, str(irrep) + '.npy')
    return np.load(fname)

class FourierPolicy:
    '''
    Wrapper class for fourier linear regression
    '''
    def __init__(self, irreps, prefix):
        '''
        Setup
        '''
        self.irreps = irreps
        self.yors = {}
        for irr in tqdm(irreps):
            self.yors[irr] = load_yor(irr, prefix)
        #self.yors = {irr: load_yor(irr, prefix) for irr in irreps}

        total_size = 0
        for parts, pdict in self.yors.items():
            # sort of hacky but w/e
            # want to grab the size of this thing
            total_size += pdict[(1,2,3,4,5,6,7,8)].shape[0] ** 2

        self.w = np.random.normal(scale=0.2, size=(total_size + 1, 1))
        self.w[-1] = 4.5

    def train_batch(self, perms, y, lr):
        X = np.vstack([self.to_irrep(p) for p in perms])
        y_pred = X@self.w
        grad = X.T@(X@self.w) - X.T@y
        self.w -= lr * grad
        loss = np.mean(np.square((y - y_pred)))
        return loss

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

class FourierPolicyTorch(FourierPolicy):
    def __init__(self, irreps, prefix, lr):
        super(FourierPolicyTorch, self).__init__(irreps, prefix)
        self.w_torch = nn.Parameter(torch.from_numpy(self.w))
        self.optim = torch.optim.Adam([self.w_torch], lr=lr)

    def train_batch(self, perms, y, **kwargs):
        self.optim.zero_grad()
        X = np.vstack([self.to_irrep(p) for p in perms])
        X_th = torch.from_numpy(X)
        y_pred = X_th.matmul(self.w_torch)
        loss = (y_pred - torch.from_numpy(y).double()).pow(2).mean()
        loss.backward()
        self.optim.step()
        return loss.item()

    def __call__(self, gtup):
        vec = torch.from_numpy(self.to_irrep(gtup))
        return vec.matmul(self.w_torch)
