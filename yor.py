import pdb
import math
import itertools
import time
from utils import check_memory
import numpy as np
from young_tableau import YoungTableau, FerrersDiagram

def cycle_to_adj_transpositions(cyc, n):
    '''
    cyc: tuple of ints, the permutation cycle
    n: size of the S_n group that this permutation lives in
    Given a cycle: a -> b -> c
    Generate perm = [cyc(i) for i in range(1, n+1)]
    TODO: can we do this without creating the mapping list
    '''
    # do bubble sort to turn this into a product of adjacent transpositions
    cyc_map = lambda x: x if x not in cyc else cyc[(cyc.index(x) + 1) % len(cyc)]
    perm = [ cyc_map(i) for i in range(1, n+1)]
    factors = []

    for i in range(n):
        for j in range(n-1, i, -1):
            if perm[j] < perm[j-1]:
                perm[j], perm[j-1] = perm[j-1], perm[j]
                factors.append((j, j+1))

    return list(reversed(factors))

def perm_to_adj_transpositions(perm, n):
    '''
    perm: a permutation of S_n list of tuple of ints. perm is given in its canonical cycle decomposition
    format
    n: integer, size of the permutation group that perm is a member of
    '''
    all_trans = []
    for cyc in perm:
        all_trans.extend(cycle_to_adj_transpositions(cyc, n))

    return all_trans

def yor(ferrers, permutation):
    '''
    Compute the irreps of a given shape using Young's Orthogonal Representation (YOR)

    ferrers: FerrersDiagram
    permutation: perm.Perm object
    permutation: list of tuples for the permutation in disjoint
                 cycle notation
    Returns: an irrep matrix of size d x d, where d is the number of standard tableaux of the
    given FerrersDiagram shape
    '''
    if all(map(lambda x: len(x) <= 1, permutation.cycle_decomposition)):
        # TODO: make a static/class function for this
        n = len(FerrersDiagram.TABLEAUX_CACHE[ferrers.partition])
        return np.eye(n)

    res = None
    for cycle in permutation.cycle_decomposition:
        ts = []
        for t in cycle_to_adj_transpositions(cycle, ferrers.size):
            y = yor_trans(ferrers, t)
            ts.append(t)
            if res is None:
                res = y
            else:
                res = res.dot(y)

    return res

def yor_trans(ferrers, transposition):
    '''
    Young's seminormal form for adjacent transposition permutations.
    EG: permutations of the form (k, k+1)
    ferrers: a FerrersDiagram object
    transposition: a 2-tuple of ints

    Returns: an irrep matrix of size d x d, where d is the number of standard tableaux of the
    given FerrersDiagram shape
    '''
    assert transposition[0] < transposition[1]
    tabs = ferrers.tableaux
    rep = np.zeros((len(tabs), len(tabs)))
    for tab in tabs:
        other = tab.transpose(transposition)
        dist = tab.dist(*transposition)
        i = tab.idx
        rep[i, i] = 1. / dist

        if other is not None:
            j = other.idx
            rep[i, j] = np.sqrt(1 - (1. / dist) ** 2)
            rep[j, i] = rep[i, j]
            rep[j, j] = 1. / other.dist(*transposition)

    return rep

def ysemi(ferrers, permutation):
    '''
    Compute the irreps of the given shape using Young's Seminormal Form

    ferrers: FerrersDiagram
    partition: tuple of ints
    permutation: list of tuples in standard cycle notation. Order doesnt matter
                 since the cycles are disjoint in standard notation
    Returns: an irrep matrix of size d x d, where d is the number of standard tableaux of the
    given FerrersDiagram shape
    '''
    # TODO: This is a hacky way of passing in the identity permutation
    if all(map(lambda x: len(x) <= 1, permutation.cycle_decomposition)):
        return np.eye(len(FerrersDiagram.TABLEAUX_CACHE[ferrers.partition]) )

    res = None
    for cycle in permutation.cycle_decomposition:
        # rewrite the cycle in terms of the adjacent transpositions group generators
        for t in reversed(cycle_to_adj_transpositions(cycle, ferrers.size)):
            if t[0] > t[1]:
                # keep transpositions in standard notation
                t = (t[1], t[0])
            y = ysemi_t(ferrers, t)
            if res is None:
                res = y
            else:
                res = y.dot(res)

    return res

def ysemi_t(f, transposition):
    '''
    Young's seminormal form for adjacent transposition permutations.
    EG: permutations of the form (k, k+1)

    f: ferrers diagram
    Returns a matrix of size = n_p x n_p, where
    n_p = the number of young tableau of the given partition
    '''
    tabs = f.tableaux
    rep = np.zeros((len(tabs), len(tabs)))

    for t in tabs:
        res = t.transpose(transposition)
        if res is None:
            # see if same row/col
            # same row --> 1
            if t.get_row(transposition[0]) == t.get_row(transposition[1]):
                rep[t.idx, t.idx] = 1.
            # same col --> -1
            elif t.get_col(transposition[0]) == t.get_col(transposition[1]):
                rep[t.idx, t.idx] = -1.
            continue

        i = t.idx
        j = res.idx
        if i < j:
            dist = t.ax_dist(*transposition)
            rep[i, i] = 1. / dist
            rep[i, j] = 1. - (1./(dist**2))
            rep[j, i] = 1.
            rep[j, j] = -1. / dist

    return rep

# TODO: Benchmarking function should go elsewhere
def benchmark():
    '''
    Benchmark time/memory usage for generating all YoungTableau for S_8
    '''
    partitions = [
        (8,),
        (7, 1),
        (6, 2),
        (6, 1, 1),
        (5, 3),
        (5, 2, 1),
        (5, 1, 1, 1),
        (4, 4),
        (4, 3, 1),
        (4, 2, 2),
        (4, 2, 1, 1),
        (4, 1, 1, 1, 1),
        (3, 3, 2),
        (3, 3, 1, 1),
        (3, 2, 2, 1),
        (3, 2, 1, 1, 1),
        (3, 1, 1, 1, 1, 1),
        (2, 2, 2, 2),
        (2, 2, 2, 1, 1),
        (2, 2, 1, 1, 1, 1),
        (2, 1, 1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 1, 1, 1),
    ]
    total_tabs = 0
    tstart = time.time()
    perms = list(((1, ) + p for p in itertools.permutations(range(2, 8+1))))
    transpositions = [(i, i+1) for i in range(1, 8)]
    dims = []
    for p in partitions:
        start = time.time()
        f = FerrersDiagram(p)
        tabs = f.tableaux
        total_tabs += len(tabs)

        #for idx, trans in enumerate(transpositions):
        #    yt = yor_t(p, trans)
        #    if idx == 0:
        #        dim = yt.shape[0]
        #        dims.append(dim)
        done = time.time() - start
        #print('Time to create {:5} tableaux for partition {:25} : {:.3f}'.format(len(tabs), str(p), done))
        #print('-' * 80)
    print('Dimensions: {}'.format(dims))
    print('Sq dims: {} | 8!: {}'.format(sum(x * x for x in dims), math.factorial(8)))
    print('Total tabs for partitions of 8: {}'.format(total_tabs))
    tend = time.time() - tstart
    print('Total time to find all tableaux: {:3f}'.format(tend))

    cache_size = 0
    mats = {}
    for k, v in YoungTableau.CACHE.items():
        cache_size += len(v)
        mats[k] = np.random.random((len(v), len(v)))
    print('Cache size: {}'.format(cache_size))
    check_memory()

if __name__ == '__main__':
    pass
