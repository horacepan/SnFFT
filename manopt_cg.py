import os
import time
import pickle
import numpy as np
import tensorflow as tf
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions

from yor import yor
from perm2 import Perm2
from young_tableau import FerrersDiagram
from utils import check_memory
from cg_utils import block_rep
from char_utils import get_char_decomp
PREFIX = 'local' if os.path.exists('/local/hopan/irreps') else 'scratch'
print(PREFIX)

def pymanopt_intertwiner(parts1, parts2, parts3, mult, n):
    '''
    parts1: partition of n
    parts2: partition of n
    parts3: partition of n
    Return CG matrix for the rho1, rho2 -> rho
    rho1 \otimes \rho2
    '''
    st = time.time()
    gtup = Perm2.swap(n, 1, 2)
    gcyc = Perm2.cont_cycle(n, 1, n)

    # FerrersDiagrams are cached so no overhead
    rep1_tup = yor(FerrersDiagram(parts1), gtup)
    rep2_tup = yor(FerrersDiagram(parts2), gtup)
    rep3_tup = yor(FerrersDiagram(parts3), gtup)

    rep1_cyc = yor(FerrersDiagram(parts1), gcyc)
    rep2_cyc = yor(FerrersDiagram(parts2), gcyc)
    rep3_cyc = yor(FerrersDiagram(parts3), gcyc)

    krep1 = np.kron(rep1_tup, rep2_tup)
    krep2 = np.kron(rep1_cyc, rep2_cyc)
    brep1 = block_rep(rep3_tup, mult)
    brep2 = block_rep(rep3_cyc, mult)

    d = krep1.shape[1]
    dz = brep1.shape[0]
    print('Kron shape: {} | block rep shape {}'.format(krep1.shape, brep1.shape), d, dz)
    manifold = Stiefel(d, dz)
    X = tf.Variable(tf.placeholder(tf.float64, [d, dz], name='X'))
    cost = tf.reduce_mean(tf.square(tf.matmul(krep1, X) - tf.matmul(X, brep1))) + \
           tf.reduce_mean(tf.square(tf.matmul(krep2, X) - tf.matmul(X, brep2)))

    problem = Problem(manifold=manifold, cost=cost, arg=X)
    solver = TrustRegions()
    Xopt = solver.solve(problem)
    elapsed = (time.time() - st) / 60.
    print(f'Finished optimization: {elapsed:.2f}min | Xopt shape:', Xopt.shape)
    return Xopt

def tostr(p):
    return ''.join(str(i) for i in p)

def compute_cg(topk):
    st = time.time()
    s8chars = pickle.load(open('/{}/hopan/irreps/s_8/char_dict.pkl'.format(PREFIX), 'rb'))

    for p1 in topk:
        for p2 in topk:
            mult_dict = get_char_decomp(s8chars, s8chars[p1] * s8chars[p2])
            for p3 in topk:
                fname = '/{}/hopan/irreps/s_8/cg/{}_{}_{}.npy'.format(PREFIX, tostr(p1), tostr(p2), tostr(p3))
                if os.path.exists(fname):
                    print('Skipping {}'.format(fname))
                    continue
                mult = mult_dict[p3]
                _st = time.time()
                print('Starting for {} | {} | {}'.format(p1, p2, p3))
                intw = pymanopt_intertwiner(p1, p2, p3, mult, n=8)
                print('Elapsed {:.2f}min | Shape {} | {}'.format((time.time() - _st) / 60., intw.shape, check_memory(verbose=False)))
                np.save(fname, intw)
                del intw
    print('Done')

if __name__ == '__main__':
    for n in range(8, 9):
        topk = [
            (n-1, 1),
            (n-2, 1, 1),
            (n-2, 2),
            # (3, 2, 2, 1),
            # (4, 2, 2),
            # (4, 2, 1, 1),
            # (3, 3, 1, 1),
        ]
        compute_cg(topk)
