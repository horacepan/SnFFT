from multiprocessing import Pool, Manager, Process
import os
import sys
import random
import time
from functools import reduce
from collections import Counter
import itertools
import pdb
from young_tableau import FerrersDiagram
from yor import yor, load_yor
import numpy as np
from utils import partitions, weak_partitions, check_memory, chunk
import perm2
from coset_utils import coset_reps, young_subgroup_canonical, young_subgroup_perm, young_subgroup, perm_from_young_tuple, tup_set
sys.path.append('./cube')
from str_cube import init_2cube, rotate, get_wreath

def dot(perm, cyc):
    p_inv = perm.inv()
    new_cyc = tuple(cyc[p_inv[i] - 1] for i in range(1, len(cyc) + 1))
    p_cyc = CyclicGroup(new_cyc, cyc.order)
    return p_cyc

class CyclicGroup:
    def __init__(self, cyc, order): 
        self.cyc = cyc
        self.size = len(cyc)
        self.order = order

    def inv(self):
        cyc = tuple((self.order - a) % self.order for a in self.cyc)
        return CyclicGroup(cyc, self.order)

    def __mul__(self, other):
        cyc = tuple((a+b) % self.order for a, b in zip(self.cyc, other.cyc))
        return CyclicGroup(cyc, self.order)

    def __add__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return '{}/{}'.format(str(self.cyc), self.order)

    def __len__(self):
        return len(self.cyc)

    def __getitem__(self, i):
        try:
            return self.cyc[i]
        except:
            pdb.set_trace()

class WreathCycSn:
    def __init__(self, cyc, perm):
        assert perm.size == len(cyc)
        self.cyc = cyc
        self.perm = perm

    @staticmethod
    def from_tup(cyc_tup, perm_tup, order):
        return WreathCycSn(CyclicGroup(cyc_tup, order), perm2.Perm2.from_tup(perm_tup))

    def __mul__(self, w):
        perm = self.perm * w.perm
        cyc = self.cyc + dot(self.perm, w.cyc)
        return WreathCycSn(cyc, perm)

    def inv(self):
        perm = self.perm.inv()
        cyc = dot(self.perm.inv(), self.cyc.inv())
        return WreathCycSn(cyc, perm)

    def __repr__(self):
        return '{} | {}'.format(self.cyc, self.perm)

def cyclic_irreps(weak_partition):
    '''
    Returns a function that computes the cyclic irrep portion
    of the wreath product irreps of the 2x2 cube.
    weak_partition: a tuple of length 3

    Returns a function that takes in an 8 tuple and returns the
        product of the cyclic irreps (float)
    '''
    _idx0 = weak_partition[0]
    _idx1 = weak_partition[0] + weak_partition[1]
    g_x = ((0,) * weak_partition[0]) + ((1,) * weak_partition[1]) + ((2,) * weak_partition[2])
    def func(tup, perm):
        # first indices 0: weak_partition[0]
        # 2nd indices weak_partition[0]: weak_partition[0] + weak_partition[1]
        # 3rd indices weak_partition[0] + weak_partition[1]:  sum(weak_partition)
        ginv_perm = perm2.Perm2.from_tup(perm).inv()
        tot = 0
        for i in range(1, 9):
            gidx = ginv_perm[i]-1 # perm is 1-indexed
            gval = g_x[gidx]
            tot += tup[i-1] * gval
        return np.exp(2j * np.pi * tot / 3.)
        p1 = sum(tup[:_idx0])
        p2 = sum(tup[_idx0: _idx1])
        p3 = sum(tup[_idx1:])

        print('Evaluating on {} | {}, {}, {}'.format(tup, p1, p2, p3))
        return np.exp(2j * np.pi * 0 * p1 / 3.) * \
               np.exp(2j * np.pi * 1 * p2 / 3.) * \
               np.exp(2j * np.pi * 2 * p3 / 3.)
    return func

def load_partition(partition, prefix='/local/hopan/irreps/'):
    '''
    partition: tuple of ints
    Returns a dict of yor matrices mapping permutation to yor rep matrices
    If the partition is (0,): 
    '''
    n = sum(partition)
    if n == 0:
        return None

    try:
        prefix = '/local/hopan/irreps'
        fname = os.path.join(prefix,  's_{}/{}.pkl'.format(n, '_'.join(map(str, partition))))
        yd = load_yor(fname, partition)
        return yd
    except:
        prefix = '/scratch/hopan/irreps'
        fname = os.path.join(prefix,  's_{}/{}.pkl'.format(n, '_'.join(map(str, partition))))
        yd = load_yor(fname, partition)
        return yd

def canonical_order(tup_rep):
    new_tup_rep = []
    idx = 0
    for lvl, tup in enumerate(tup_rep):
        if lvl == 0:
            new_tup_rep.append(tup)
        else:
            tnew = tuple(idx + i for i in tup)
            new_tup_rep.append(tnew)
        idx += len(tup)

    return tuple(new_tup_rep)

# This young subgroups that are direct products
# But here, the young subgroup wont be a subgroup of S_n, where n = sum(alpha)
# S_alpha, alpha=(4,4) will just be S_{1,2,3,4} x S_{1,2,3,4}
def young_subgroup_yor(alpha, _parts, prefix='/local/hopan/irreps/'):
    '''
    Compute the wreath product group for the irreps indexed by the weak partition alpha
    with partition components given by _parts
    alpha: tuple of ints, where each part is nonnegative
    _parts: list of partitions of the ints in alpha

    Ex usage:
        alpha = (4, 4)
        parts = [(2,2), (3,1)]

        alpha = (2, 3, 0)
        parts = [(2,0), (2,1), ()]
    '''
    assert len(alpha) == 3, 'alpha must be length 3'
    #assert sum(alpha) == 8, 'alpha must be a partition of 8'
    #assert (alpha[1] * 1 + alpha[2] * 2) % 3 == 0, 'alpha must be a valid configuration'

    wreath_dict = {}
    # load the irreps for each partition
    nonzero_parts = [load_partition(p, prefix) for p in _parts if sum(p) > 0]
    # group elements are in iterproduct(
    # iterate over s_alpha subgroup and compute tensor stuff
    # currently the S_alpha looks like S_{1, ..., alpha_1} x S_{1, ..., alpha_2} x ... x S_{1, ..., alpha_n}
    # but we'd like it to look like (1...alpha_1), (alpha_1+1 .... alpha_2), .... 

    for g in young_subgroup(alpha):
        # length of g should be length of nonzero_parts
        ms = [yd[perm] for yd, perm in zip(nonzero_parts, g)]
        # convert the permutation to the corresponding perm in S_{1, ..., alpha_1} x S_{alpha_1+1, ..., alpha_1+alpha_2} x ...
        gprime = canonical_order(g) # (1,2,3,4)(1,2) -> (1,2,3,4)(5,6)
        tup = tuple(i for t in gprime for i in t)
        wreath_dict[tup] = reduce(np.kron, ms)

    return wreath_dict

def _proc_yor(perms, young_yor, young_sub_set, reps, rep_dict):
    '''
    perms: list of perm objects or tuples?
    young_yor: 
    young_sub_set:
    reps:
    rep_dict: save dict
    '''
    for g in perms:
        g_rep = {}
        for i, t_i in enumerate(reps):
            for j, t_j in enumerate(reps):
                ti_g_tj = t_i.inv() * g * t_j
                if ti_g_tj.tup_rep in young_sub_set:
                    g_rep[(i, j)] = young_yor[ti_g_tj.tup_rep]

        rep_dict[g.tup_rep] = g_rep 

def wreath_yor_par(alpha, _parts, prefix='/local/hopan/', par=8):
    '''
    alpha: weak partition of 8 into 3 parts?
    _parts: list of partitions of each part of alpha
    Return a dict mapping group elmeent in S_8 -> rep
    The rep actually needs to be a dictionary of tuples (i, j) -> matrix
    where the i, j denote the i, j block in the matrix.
    Ex:
        alpha = (0, 0, 0, 0, 1, 1, 1, 1)
        _parts = [(2,2), (3,1)]
    '''
    #print('Wreath yor with {} processes'.format(par))
    n = sum(alpha)
    _sn = perm2.sn(n, prefix)
    young_sub = young_subgroup_perm(alpha)
    young_sub_set = tup_set(young_sub)
    young_yor = young_subgroup_yor(alpha, _parts, os.path.join(prefix, 'irreps'))
    reps = coset_reps(_sn, young_sub)
    #print('Len coset reps: {}'.format(len(reps)))
    #print('Total loop iters: {}'.format(len(_sn) * len(reps) * len(reps)))
    cnts = np.zeros((len(reps), len(reps)))
    sn_chunks = chunk(_sn, par)
    manager = Manager()
    rep_dict = manager.dict()
    nprocs =  []

    for i in range(par):
        perms = sn_chunks[i]
        proc = Process(target=_proc_yor, args=[perms, young_yor, young_sub_set, reps, rep_dict])
        nprocs.append(proc)

    for p in nprocs:
        p.start()
    for p in nprocs:
        p.join()

    return rep_dict


def wreath_yor(alpha, _parts, prefix='/local/hopan/'):
    '''
    alpha: weak partition of 8 into 3 parts?
    _parts: list of partitions of each part of alpha
    Return a dict mapping group elmeent in S_8 -> rep
    The rep actually needs to be a dictionary of tuples (i, j) -> matrix
    where the i, j denote the i, j block in the matrix.
    Ex:
        alpha = (0, 0, 0, 0, 1, 1, 1, 1)
        _parts = [(2,2), (3,1)]
    '''
    n = sum(alpha)
    _sn = perm2.sn(n, prefix)
    young_sub = young_subgroup_perm(alpha)
    young_sub_set = tup_set(young_sub)
    young_yor = young_subgroup_yor(alpha, _parts, os.path.join(prefix, 'irreps'))
    reps = coset_reps(_sn, young_sub)
    rep_dict = {}
    #print('Len coset reps: {}'.format(len(reps)))
    #print('Total loop iters: {}'.format(len(_sn) * len(reps) * len(reps)))
    cnts = np.zeros((len(reps), len(reps)))

    # this part can be parallelized
    # loop over the group
    # things we need are: group element inv, group element multiplication
    # then grabbing the yor for the appropriate yor thing
    for g in _sn:
        g_rep = {}
        for i, t_i in enumerate(reps):
            for j, t_j in enumerate(reps):
                ti_g_tj = t_i.inv() * g * t_j
                if ti_g_tj.tup_rep in young_sub_set:
                    g_rep[(i, j)] = young_yor[ti_g_tj.tup_rep]
                    cnts[i, j] = cnts[i, j] + 1

        rep_dict[g.tup_rep] = g_rep 

    #return rep_dict, cnts
    return rep_dict

def get_mat(g, yor_dict):
    '''
    g: perm2.Perm2 object
    yor_dict: dict mapping perm2.Perm2 object -> (dict of (i, j) -> numpy matrix)

    Returns matrix for this ydict
    '''
    if type(g) == tuple:
        g = perm2.Perm2.from_tup(g)

    yg = yor_dict[g.tup_rep]
    vs = list(yg.values())
    block_size = vs[0].shape[0]
    size = len(yg) * block_size
    mat = np.zeros((size, size))

    for (i, j), v in yg.items():
        x1, x2 = (block_size*i, block_size*i+block_size)
        y1, y2 = (block_size*j, block_size*j+block_size)
        mat[x1:x2, y1:y2] = v

    return mat

def wreath_rep(orientation, perm, yor_dict, cyc_irrep_func=None, alpha=None):
    if (cyc_irrep_func is None) and (alpha is None):
        raise ValueError('Need to supply either irrep func or alpha')
    if cyc_irrep_func is None:
        cyc_irrep_func = cyclic_irreps(alpha)

    return cyc_irrep_func(orientation, perm) * get_mat(perm, yor_dict)

def mult(g, h, yd):
    '''
    g: perm2.Perm2
    h: perm2.Perm2
    yd: dictionary mapping Perm2 objects -> (dicts of (i, j) int tuples -> numpy matrices)
            which represent the wreath product matrices
    Returns a numpy matrix
    '''
    if type(g) == tuple:
        g = perm2.Perm2.from_tup(g)
    if type(h) == tuple:
        h = perm2.Perm2.from_tup(h)
    mat_g = get_mat(g, yd)
    mat_h = get_mat(h, yd)
    return mat_g.dot(mat_h)

def test_wreath_class():
    perm = perm2.Perm2.from_tup((1,3,4,2))
    cyc = CyclicGroup((0, 1, 0, 2), 3)
    #w = WreathCycSn(cyc, perm)
    c = init_2cube()
    for f in ['r', 'l', 'f', 'b', 'u', 'd']:
        cube_str = rotate(c, f)
        o1, p1 = get_wreath(cube_str)
        o2, pinv = get_wreath(rotate(c, 'i' + f))
        #c1 = CyclicGroup((2, 1, 0, 0, 0, 0, 2, 1), 3)
        c1 = CyclicGroup(o1, 3)
        #c2 = CyclicGroup((2, 1, 0, 0, 0, 0, 2, 1), 3)
        c2 = CyclicGroup(o2, 3)
        #p1 = perm2.Perm2.from_tup((2, 7, 3, 4, 5, 6, 8, 1))
        #p2 = perm2.Perm2.from_tup((8, 1, 3, 4, 5, 6, 2, 7))
        p1 = perm2.Perm2.from_tup(p1)
        p2 = perm2.Perm2.from_tup(pinv)

        w = WreathCycSn(c1, p1)
        winv = WreathCycSn(c2, p2)
        prod = w * winv
        print('Face: {} | prod should be identity wreath: {}'.format(f, prod))
        print('===============')

def test_wreath(alpha, _parts, pkl_prefix='/local/hopan/'):
    start = time.time()
    print('alpha: {} | parts: {}'.format(alpha, _parts))
    wreath_yor(alpha, _parts, pkl_prefix)
    print('Elapsed: {:.2f}'.format(time.time() - start))

if __name__ == '__main__':
    test_wreath_class()
    exit()
    alpha = (6, 1, 1)
    _parts = ((4,2), (1,), (1,))

    if len(sys.argv) > 1:
        print('looking in {}'.format(sys.argv[1]))
        test_wreath(alpha, _parts, sys.argv[1])
    else:
        print('using default prefix')
        test_wreath(alpha, _parts)
    check_memory()
