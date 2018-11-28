import time
import pdb
import numpy as np
from yor import yor, ysemi
from young_tableau import FerrersDiagram
from utils import sn, partitions
from perm import Perm


# can use ysemi or yor
irrep = yor

def fft_full(f, n):
    '''
    f: a function from S_n -> \mathbb{R}
    n: integer
    Returns: a dictionary mapping FerrersDiagrams(which index your irreps) to irrep matrices
    '''
    fourier_parts = {}
    parts = partitions(n)

    for p in parts:
        ferrers = FerrersDiagram(p)
        fourier_parts[ferrers] = fft(f, ferrers)

    return fourier_parts

def fft(f, ferrers):
    '''
    Compute Clausen's FFT:
    Ref: 'FAST FOURIER TRANSFORMS FOR SYMMETRIC GROUPS: THEORY AND IMPLEMENTATION'
    By MICHAEL CLAUSEN AND ULRICH BAUM
    http://www.ams.org/journals/mcom/1993-61-204/S0025-5718-1993-1192969-X/S0025-5718-1993-1192969-X.pdf

    f: function from S_n to \mathbb{R}
    ferrers: a FerrersDiagram object (indicates which irrep to compute the transform over)
    Returns a matrix of size d x d, where d is the number of standard tableaux of the FerrersDiagram shape

    Note that the specific permutation group S_n is given by the size of the ferrers diagram
    '''
    # iterate over cosets
    if ferrers.size == 1:
        return np.eye(1) * f(Perm([(1, )]))


    n = ferrers.size
    d_branches = ferrers.branch_down()
    tabs = ferrers.tableaux

    d_lambda = len(tabs)
    f_hat = np.zeros((d_lambda, d_lambda))
    for i in range(1, n+1):
        cyc = Perm([tuple(j for j in range(i, n+1))])
        f_i = lambda pi: f(cyc * pi) # assume that the input function is a function on Perm objects

        # irrep function (aka yor) requires the 2nd argument to be a list of tuples
        rho_i = irrep(ferrers, cyc)
        idx = 0 # used to figure out where the direct sum should add things
        res = np.zeros(f_hat.shape)

        for lambda_minus in d_branches:
            fft_fi = fft(f_i, lambda_minus)
            #fft_fi = ft(f_i, lambda_minus)
            d = fft_fi.shape[0]
            res[idx: idx+d, idx: idx+d] += fft_fi
            idx += d

        f_hat += rho_i.dot(res)
    return f_hat

def fourier_transform(f, ferrers):
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
        perm = Perm(p)
        if res is None:
            res = f(perm) * irrep(ferrers, perm)
        else:
            res += f(perm) * irrep(ferrers, perm)

    return res

if __name__ == '__main__':
    pass
