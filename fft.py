import time
import pdb
import numpy as np
from yor import yor, ysemi
from young_tableau import FerrersDiagram
from utils import partitions
from perm import Perm, sn
from perm2 import Perm2
from perm2 import sn as sn2
from wreath import wreath_rep, WreathCycSn


# can use ysemi or yor
irrep = yor

def ft_full(f, n, algo='fft'):
    '''
    Computes the fourier transform using either the fft or the slow fourier transform.
    f: a function from S_n -> \mathbb{R}
    n: integer
    algo: string

    Returns: a dictionary mapping FerrersDiagrams(which index your irreps) to irrep matrices
    '''
    fourier_parts = {}
    parts = partitions(n)

    for p in parts:
        ferrers = FerrersDiagram(p)
        if algo == 'fft':
            fourier_parts[ferrers] = fft2(f, ferrers)
        elif algo == 'ft1':
            fourier_parts[ferrers] = fourier_transform(f, ferrers)
        elif algo == 'ft2':
            fourier_parts[ferrers] = fourier_transform2(f, ferrers)

    return fourier_parts

def fft2(f, ferrers):
    if ferrers.size == 1:
        #return np.eye(1) * f(Perm([(1, )]))
        return np.eye(1) * f(Perm2.eye(ferrers.size))

    n = ferrers.size
    d_branches = ferrers.branch_down()
    tabs = ferrers.tableaux

    d_lambda = len(tabs)
    f_hat = np.zeros((d_lambda, d_lambda))
    for i in range(1, n+1):
        #cyc = Perm([tuple(j for j in range(i, n+1))])
        cyc_dict = {j:j+1 for j in range(i, n)}
        cyc_dict[n] = i
        cyc = Perm2(cyc_dict, n)
        f_i = lambda pi: f(cyc * pi) # assume that the input function is a function on Perm objects

        # irrep function (aka yor) requires the 2nd argument to be a list of tuples
        rho_i = irrep(ferrers, cyc)
        idx = 0 # used to figure out where the direct sum should add things
        res = np.zeros(f_hat.shape)

        for lambda_minus in d_branches:
            fft_fi = fft2(f_i, lambda_minus)

            d = fft_fi.shape[0]
            res[idx: idx+d, idx: idx+d] += fft_fi
            idx += d

        f_hat += rho_i.dot(res)
    return f_hat

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
    for perm in permutations:
        if res is None:
            res = f(perm) * irrep(ferrers, perm)
        else:
            res += f(perm) * irrep(ferrers, perm)

    return res

def fourier_transform2(f, ferrers):
    '''
    Compute the full fourier transform
    f: function from S_n -> \mathbb{R}
    ferrers: FerrersDiagram

    Returns a matrix of dimension d x d, where d is the number of standard young tableaux
    of the given FerrersDiagram shape
    '''
    permutations = sn2(sum(ferrers.partition))
    res = None
    for perm in permutations:
        if res is None:
            res = f(perm) * irrep(ferrers, perm)
        else:
            res += f(perm) * irrep(ferrers, perm)

    return res

def cube2_inv_fft_func(irrep_dict, fourier_mat, coset_reps, cyclic_irrep_func):
    def f(cyc_tup, perm):
        return cube2_inv_fft_part(irrep_dict, fourier_mat, coset_reps, cyclic_irrep_func, cyc_tup, perm)
    return f

def cube2_inv_fft_part(irrep_dict, fourier_mat, coset_reps, cyclic_irrep_func, cyc_tup, perm_tup):
    '''
    irrep_dict: map from dictionary of permutation(in tuple rep form) -> numpy matrices
    fourier_mat: numpy matrix
    alpha: tuple of ints. Weak partition of 8 into 3 parts
    parts: tuple of tuples, where each tuple is a partition of the corresponding index of alpha
    g_tup: tuple of ints, an s8 permutation in tuple format

    Returns: a float = dimension * Trace(\rho(g inverse) * fourier_mat)
    '''
    # now actually i want the inverse of this group element
    cinv, pinv = WreathCycSn.from_tup(cyc_tup, perm_tup, order=3).inv_tup_rep()
    dim = fourier_mat.shape[0]
    inv_irrep = wreath_rep(cinv, pinv, irrep_dict, coset_reps, cyclic_irrep_func)
    return dim * np.ravel(inv_irrep.T).dot(np.ravel(fourier_mat))

def benchmark(n):
    id_f = lambda x: 1
    sn(n)
    sn2(n)

    start_1 = time.time()
    full_transform_2 = ft_full(id_f, n, 'ft1')
    end_1 = time.time()

    start_2 = time.time()
    full_transform2 = ft_full(id_f, n, 'ft2')
    end_2 = time.time()

    #for p in parts:
    #    p_start = time.time()
    #    ferrers = FerrersDiagram(p)
    #    fft_res = fft2(f, ferrers)
    #    p_end = time.time()
    #    print('Partition: {:20} | Dim: {:10} | Time: {:.2f}'.format(str(p), str(fft_res.shape), p_end-p_start))
    print('Full transform 1 | Elapsed: {:.2f}'.format(end_1 - start_1))
    print('Full transform 2 | Elapsed: {:.2f}'.format(end_2 - start_2))
    #pdb.set_trace()

if __name__ == '__main__':
    benchmark(7)
