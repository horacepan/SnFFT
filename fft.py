from permutation import Permutation
import pdb
import numpy as np
from yor import yor, yor_t, FerrersDiagram


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
    f: function from group to \mathbb{R}
    partition: a tuple of ints
    Returns a matrix
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
        rho_i = yor(ferrers, [cyc])
        idx = 0 # used to figure out where the direct sum should add things
        res = np.zeros(f_hat.shape)
        for lambda_minus in d_branches:
            #fft_fi = fft(f_i, lambda_minus)
            fft_fi = ft(f_i, lambda_minus)
            # now figure out where to slot it in
            d = fft_fi.shape[0]
            #res[idx: idx+d, idx: idx+d] += rho_i[idx: idx+d, idx: idx+d].dot(fft_fi)
            res[idx: idx+d, idx: idx+d] = fft_fi
            idx += d

        if n == 3:
            print('Adding for cycle: {}'.format(cyc))
            print(rho_i.dot(res))
            print('=====')
        f_hat += rho_i.dot(res)
    return f_hat

def s1():
    permutations = [ [(1,)] ]
    return permutations

def s2():
    permutations = [
        [(1,)],
        [(1,2)],
    ]
    return permutations

def s3():
    permutations = [
        [(1,)],
        [(1, 2)],
        [(1, 3)],
        [(2, 3)],
        [(1, 2, 3)],
        [(1, 3, 2)],
    ]
    return permutations
def s4():
    permutations = [
        [(1,)],
        [(1, 2)],
        [(1, 3)],
        [(1, 4)],
        [(2, 3)],
        [(2, 4)],
        [(3, 4)],
        [(1, 2), (3, 4)],
        [(1, 3), (2, 4)],
        [(1, 4), (2, 3)],
        [(1, 2, 3)],
        [(1, 2, 4)],
        [(1, 3, 2)],
        [(1, 3, 4)],
        [(1, 4, 2)],
        [(1, 4, 3)],
        [(2, 3, 4)],
        [(2, 4, 3)],
        [(1, 2, 3, 4)],
        [(1, 2, 4, 3)],
        [(1, 3, 2, 4)],
        [(1, 3, 4, 2)],
        [(1, 4, 2, 3)],
        [(1, 4, 3, 2)],
    ]
    return permutations

def sn(n):
    fname = 's{}'.format(n)
    func = eval(fname)
    return func()

def ft(f, ferrers):
    permutations = sn(sum(ferrers.partition))
    res = None
    for p in permutations:
        if res is None:
            yp = yor(ferrers, p)
            res = f(p) * yor(ferrers, p)
        else:
            yp = yor(ferrers, p)
            res += f(p) * yor(ferrers, p)

    return res

def trivial_func(x):
    result = 0
    for cyc in x:
        result += sum(x)

    return result

def sn_func(perm):
    return 1

def test_fft():
    f = sn_func
    partition = (1,1)
    ferrers = FerrersDiagram(partition)
    fft_res = fft(f, ferrers)
    full_transform = ft(f, ferrers)
    print('FFT:')
    print(fft_res)
    print('Full Transform:')
    print(full_transform)
    print('Equal ?: {}'.format(np.allclose(fft_res, full_transform)))
    pdb.set_trace()

def test_yor():
    partition = (2,1)
    ferrers = FerrersDiagram(partition)
    p = [(1, 2)]
    yt = yor(ferrers, p)

if __name__ == '__main__':
    test_fft()
