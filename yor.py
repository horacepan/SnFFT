import pdb
import math
import itertools
import time
from utils import check_memory
import numpy as np
from young_tableau import YoungTableau, FerrersDiagram

def cycle_to_transpositions(cyc):
    '''
    cyc: iterable of ints
    Return: list of 2-tuples
    '''
    try:
        if len(cyc) == 2:
            return [cyc[:]]
    except:
        pdb.set_trace()

    return [(cyc[i], cyc[(i+1)]) for i in range(len(cyc) - 1)]

def yor(ferrers, permutation):
    '''
    ferrers: FerrersDiagram
    partition: tuple of ints
    permutation: list of tuples in standard cycle notation. Order doesnt matter
                 since the cycles are disjoint in standard notation
    Identity permutation is given as?
    '''
    if len(permutation[0]) <= 1:
        return np.eye(len(FerrersDiagram.TABLEAUX_CACHE[ferrers.partition]) )

    res = None
    for cycle in permutation:
        for t in reversed(cycle_to_transpositions(cycle)):
            if res is None:
                res = yor_t(ferrers, t)
            else:
                res = yor_t(ferrers, t).dot(res)

    return res

def yor_t(f, transposition):
    '''
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

def test():
    partitions = [(4, ), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)]
    for p in partitions:
        f = FerrersDiagram(p)
        tabs = f.tableaux

        for trans in [(1,2), (2, 3), (3,4)]:
            print('Transposition: {}'.format(trans))
            print(yor_t(p, trans))
            print('-'*10)
        print('-'*80)

def benchmark():
    '''
    Benchmark time/memory usage for generating all YoungTableau for S_8
    '''
    # TODO: function to do the partitions
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

def test_yor():
    partition = (3, 1)
    f = FerrersDiagram(partition)
    py = yor(f, [(1,2)])
    pt = yor_t(f, (1,2))
    print(py)
    print('=?')
    print(pt)
    print(np.allclose(py, pt))

if __name__ == '__main__':
    test_yor()
