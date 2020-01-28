import pdb
import time
import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

def load_yor(irrep, prefix):
    '''
    Loads the given irreps
    '''
    fname = os.path.join(prefix, '_'.join(str(i) for i in irrep) + '.pkl')
    pkl = pickle.load(open(fname, 'rb'))
    return pkl

class FourierPolicy:
    '''
    Wrapper class for fourier linear regression
    '''
    def __init__(self, irreps, prefix):
        '''
        Setup
        '''
        self.irreps = irreps
        self.model = LinearRegression()
        self.yors = {irr: load_yor(irr, prefix) for irr in irreps}
        self.fmats = {}

    def fit_perms(self, perms, y):
        '''
        perms: list of permutation tuples
        y: list/numpy array of distances
        '''
        print('Fitting with irreps:', self.irreps)
        st = time.time() 
        X = np.vstack([self.to_irrep(p) for p in perms])
        elapsed = time.time() - st
        print('Irrep conversion elapsed: {:.2f}s | Avg: {:.2f}s over {} perms'.format(elapsed, elapsed / len(perms), len(perms)))
        self.model.fit(X, y)

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
        return np.hstack(irrep_vecs)
