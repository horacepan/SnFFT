import pickle
import psutil
import time
import pdb
import os
from functools import reduce
import resource

def chunk(lst, n):
    '''
    Split the given lit into n approximately equal chunks
    '''
    if len(lst) % n == 0:
        size = len(lst) // n
    else:
        size = (len(lst) // n) + 1
    return [lst[i:i + size] for i in range(0, len(lst), size)]

def partitions(n, start=1):
    '''
    Generate all the partitions of n
    n: integer
    start: integer
    Returns a list of int tuples
    '''
    if n == 0:
        return [()]

    parts = [(n, )]
    for i in range(start, n // 2 + 1):
        for p in partitions(n-i, i):
            parts.append(p + (i, ))

    return parts

def weak_partitions(n, k):
    '''
    Returns a list of the weak partitions of n into k parts.
    n: integer
    k: integer

    Returns a list of k-tuples of ints. Each index of the k-tuple is
    a nonnegative integer.
    Ex:
        weak_partitions(3, 2): [(3,0), (0,3), (2,1), (1,2)]
    '''
    if n < 0 or k == 0:
        return []
    if k == 1:
        return [(n,)]

    lst = []
    for l in range(n+1):
        # we can use 0, 1, ..., n pieces of n in the first index
        others = weak_partitions(n-l, k-1)
        for part in others:
            lst.append((l,) + part)

    return lst

def check_memory2():
    res_ = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Consumed {:.2f}mb memory".format(res_/(1024**2)))

def check_memory():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    print("Consumed {:.2f}mb memory".format(mem))
    return mem

def gcd(a, b):
    while b:
        a, b = b, a%b
    return a

# Source: https://stackoverflow.com/a/147539
def _lcm(a, b):
    return a*b // gcd(a, b)

def lcm(*args):
    return reduce(_lcm, args)

def canonicalize(cycle_decomp):
    '''
    cycle_decomp: list of lists/tuples of ints
    Just rotates the cycles so that the smallest element is first
    '''
    new_decomp = []
    for cyc in cycle_decomp:
        if len(cyc) == 0:
            pdb.set_trace()
        min_idx = cyc.index(min(cyc))
        new_cyc = cyc[min_idx:] + cyc[:min_idx]

        new_decomp.append(new_cyc)

    return new_decomp

def commutator(x, y):
    return (x * y * x.inverse() * y.inverse())

def test_canonicalize():
    print('utils.test_canonicalize')
    x = [(1,2,3), (1,2)]
    print(canonicalize(x), x)

    x = [(2, 1, 3)]
    print(canonicalize(x), x)

def load_pkl(fname, options='rb'):
    #print('loading from pkl: {}'.format(fname))
    with open(fname, options) as f:
        res = pickle.load(f)
        return res

def tf(f, args=None):
    if args is None:
        args = []

    start = time.time()
    feval = f(*args)
    end = time.time()
    print('Running {} | time {:.2f}'.format(f.__name__, end - start))
    return feval

if __name__ == '__main__':
    print(list(partitions(5)))
