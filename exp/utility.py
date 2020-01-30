import os
import pickle

import numpy as np

def load_yor(irrep, prefix):
    '''
    irrep: tuple
    prefix: directory to load from
    Assumption: the pkl files seprate the parts of the tuple with an underscore
    Ex: (2, 2) -> 2_2.pkl
    '''
    fname = os.path.join(prefix, '_'.join(str(i) for i in irrep) + '.pkl')
    pkl = pickle.load(open(fname, 'rb'))
    return pkl

def load_np(irrep, prefix):
    fname = os.path.join(prefix, str(irrep) + '.npy')
    return np.load(fname)



def cg_mat(p1, p2, p3, prefix='/local/hopan/irreps/s_8/cg'):
    def sflat(tup):
        return ''.join(str(x) for x in tup)
    fname = os.path.join(prefix, '{}_{}_{}.npy'.format(
        sflat(p1), sflat(p2), sflat(p3)
    ))
    return np.load(fname)
