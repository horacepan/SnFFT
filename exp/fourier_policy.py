import pdb
import time
import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

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
        self.model = LinearRegression(normalize=True)
        self.yors = {irr: load_yor(irr, prefix) for irr in irreps}
        self.fmats = {}

        total_size = 0
        for parts, pdict in self.yors.items():
            # sort of hacky but w/e
            # want to grab the size of this thing
            total_size += pdict[(1,2,3,4,5,6,7,8)].shape[0] ** 2

        self.w = np.random.normal(scale=0.2, size=(total_size + 1, 1))
        self.w[-1] = 4.5

    def fit_perms(self, perms, y):
        '''
        perms: list of permutation tuples
        y: list/numpy array of distances
        '''
        X = np.vstack([self.to_irrep(p) for p in perms])
        self.model.fit(X, y)

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
        val = 0
        for mat in self.fmats:
            val += mat.shape[0] * self.yors[gtup].dot(mat)
        return val

    def to_irrep(self, gtup):
        '''
        Loop over irreps -> cat the irrep (via the yor dicts) -> reshape
        gtup: perm tuple
        '''
        irrep_vecs = []
        for irr in self.irreps:
            rep = self.yors[irr][gtup].T
            dim = rep.shape[0]
            vec = rep.reshape(1, -1) #* np.sqrt(dim)
            irrep_vecs.append(vec)
        irrep_vecs.append(np.array([[1]]))
        return np.hstack(irrep_vecs)

class LoadedFourierPolicy(FourierPolicy):
    def __init__(self, irreps, yor_prefix, fhat_prefix):
        super(LoadedFourierPolicy, self).__init__(irreps, yor_prefix)
        self.fhats = {irr: load_np(irr, fhat_prefix) for irr in irreps}

    def __call__(self, gtup):
        pass
