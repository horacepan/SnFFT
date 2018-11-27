import os
from functools import reduce
import psutil
import resource

# TODO: algorithm to generate these instead of hardcoding
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
    '''
    Return the permutations (in cycle notation) for n = 1, 2, 3, 4
    Returns a list of list of tuples
    '''
    fname = 's{}'.format(n)
    func = eval(fname) # TODO: hack city lol
    return func()

def check_memory():
    res_ = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    process = psutil.Process(os.getpid())
    resp = process.memory_info().rss
    print("Consumed {:.2f}mb memory".format(res_/(1024**2)))

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

if __name__ == '__main__':
    for p in s3():
        print(Perm(p))
