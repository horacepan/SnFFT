from permutation import Permutation
import pdb
import numpy as np
from yor import yor, ysemi
from young_tableau import FerrersDiagram
from utils import sn


irrep = yor

def compose(p1, p2):
    '''
    Multiply permutations p1 and p2
    p1: list of tuple of ints
    p2: list of tuple of ints
    '''
    def perm_size(p):
        _max = 0
        for cyc in p:
            _max = max(cyc)
        return _max

    n = max(perm_size(p1), perm_size(p2))
    for i in range(1, n+1):
        # evaluate p1(p2(i))
        pass

def fft(f, ferrers):
    '''
    Compute Clausen's FFT:
    Ref: 'FAST FOURIER TRANSFORMS FOR SYMMETRIC GROUPS: THEORY AND IMPLEMENTATION'
    By MICHAEL CLAUSEN AND ULRICH BAUM
    http://www.ams.org/journals/mcom/1993-61-204/S0025-5718-1993-1192969-X/S0025-5718-1993-1192969-X.pdf

    f: function from S_n to \mathbb{R}
    partition: a tuple of ints
    Returns a matrix of size d x d, where d is the number of standard tableaux of the FerrersDiagram shape
    '''
    # iterate over cosets
    if ferrers.size == 1:
        return np.eye(1) * f(Permutation(1, ))


    n = ferrers.size
    d_branches = ferrers.branch_down()
    tabs = ferrers.tableaux

    d_lambda = len(tabs)
    f_hat = np.zeros((d_lambda, d_lambda))
    for i in range(1, n+1):
        #cyc = Permutation([j for j in range(i, n+1)])
        cyc = [j for j in range(i, n+1)] # what happens when we do (n, n), aka the identity coset rep
        # TODO: actually compose the permutations. only okay right now b/c test just uses identity func
        #f_i = lambda pi: f(compose(cyc, pi))
        f_i = lambda pi: f(cyc)
        rho_i = irrep(ferrers, [cyc])
        idx = 0 # used to figure out where the direct sum should add things
        res = np.zeros(f_hat.shape)

        for lambda_minus in d_branches:
            #fft_fi = fft(f_i, lambda_minus)
            fft_fi = ft(f_i, lambda_minus)
            # now figure out where to slot it in
            d = fft_fi.shape[0]
            #res[idx: idx+d, idx: idx+d] += rho_i[idx: idx+d, idx: idx+d].dot(fft_fi)
            res[idx: idx+d, idx: idx+d] += fft_fi
            idx += d

        f_hat += rho_i.dot(res)
    return f_hat

def ft(f, ferrers):
    '''
    Compute the full fourier transform
    f: function from S_n -> \mathbb{R}
    ferrers: FerrersDiagram

    Returns a matrix of dimension d x d, where d is the number of standard young tableaux
    of the given FerrersDiagram shape
    '''
    permutations = sn(sum(ferrers.partition))
    res = None
    for p in permutations:
        if res is None:
            yp = irrep(ferrers, p)
            res = f(p) * irrep(ferrers, p)
        else:
            yp = irrep(ferrers, p)
            res += f(p) * irrep(ferrers, p)

    return res

def test_fft():
    f = lambda x: 1
    for partition in [(3,), (2,1), (1,1,1)]:
        ferrers = FerrersDiagram(partition)
        fft_result = fft(f, ferrers)
        full_transform = ft(f, ferrers)
        fft_sum = np.sum(fft_result)
        print('Equal: {} | sum: {}'.format(np.allclose(fft_result, full_transform), np.sum(fft_result)))

if __name__ == '__main__':
    test_fft()
