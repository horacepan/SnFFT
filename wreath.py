import time
from functools import reduce
from collections import Counter
import itertools
import pdb
from young_tableau import FerrersDiagram
from yor import yor, load_yor
import numpy as np
from utils import partitions, weak_partitions
import perm2
from coset_utils import coset_reps, young_subgroup_canonical, young_subgroup_perm, young_subgroup, perm_from_young_tuple, tup_set

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
    Return a list of the irreps of the cylic group of order k
    given by the weak_partition, where weak_partition is a weak partition of k.

    weak_partition: list/tuple of nonnegative ints
    Ex:
        cyclic_irreps((1, 1)) = [exp{2i pi * 0/2}, exp{2i * pi * 1/2}]
    '''
    cyc_irreps = []

    for i, cnt in enumerate(weak_partition):
        if cnt > 0:
            cyc_irreps.extend([np.exp(2j*np.pi*i) / cyc_order] * cnt)

    return cyc_irreps

def load_partition(partition):
    '''
    partition: tuple of ints
    Returns a dict of yor matrices mapping permutation to yor rep matrices
    If the partition is (0,): 
    '''
    n = sum(partition)
    if n == 0:
        return None
    fname = '/local/hopan/irreps/s_{}/{}.pkl'.format(n, '_'.join(map(str, partition)))
    return load_yor(fname, partition)

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
def young_subgroup_yor(alpha, _parts):
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
    assert sum(alpha) == 8, 'alpha must be a partition of 8'
    assert (alpha[1] * 1 + alpha[2] * 2) % 3 == 0, 'alpha must be a valid configuration'

    wreath_dict = {}
    # load the irreps for each partition
    nonzero_parts = [load_partition(p) for p in _parts if sum(p) > 0]
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

def wreath_yor(alpha, _parts):
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
    _sn = perm2.sn(n)
    young_sub = young_subgroup_perm(alpha)
    young_sub_set = tup_set(young_sub)
    young_yor = young_subgroup_yor(alpha, _parts)
    reps = coset_reps(_sn, young_sub)
    rep_dict = {}
    print('Len coset reps: {}'.format(len(reps)))
    print('Total loop iters: {}'.format(len(_sn) * len(reps) * len(reps)))

    for g in _sn:
        g_rep = {}
        for i, t_i in enumerate(reps):
            for j, t_j in enumerate(reps):
                ti_g_tj = t_i.inv() * g * t_j
                if ti_g_tj.tup_rep in young_sub_set:
                    g_rep[(i, j)] = young_yor[ti_g_tj.tup_rep]
        rep_dict[g] = g_rep 

    return rep_dict

def compute_dims():
    '''
    Compute the dimension of all the irreps of Z/3Z[S_8](aka 2x2 cube group)
    '''
    res = 1
    tot = 0
    _max = 0
    _max_irrep = None
    all_dims = []
    dim_dict = {}
    srted_dict = []
    for pk in weak_partitions(8, 3):
        if ((pk[1] * 1 + pk[2] * 2) % 3) != 0:
            continue

        for p0 in partitions(pk[0]):
            
            pf0 = FerrersDiagram(p0)
            for p1 in partitions(pk[1]):
                pf1 = FerrersDiagram(p1)
                for p2 in partitions(pk[2]):
                    pf2 = FerrersDiagram(p2)
                    dim = max(1, len(pf0.tableaux)) * max(1, len(pf1.tableaux)) * max(1, len(pf2.tableaux))
                    all_dims.append(dim)
                    dim_dict[(p0, p1, p2)] = dim
                    srted_dict.append((dim, (p0, p1, p2)))
                    if dim > _max:
                        max_irrep = (p0, p1, p2)
                        _max = dim

    print('Largest dim irrep: {}'.format(_max))
    print('Max irrep: {}'.format(max_irrep))
    print('Num irreps: {}'.format(len(all_dims)))

    '''
    srted_dict.sort(reverse=True)
    with open('dims.txt', 'w') as f:
        for d, ps in srted_dict:
            f.write('{:2} | {}, {}, {}\n'.format(str(d), str(ps[0]), str(ps[1]), str(ps[2])))
    '''
def test_wreath_class():
    perm = perm2.Perm2.from_tup((1,3,4,2))
    cyc = CyclicGroup((0, 1, 0, 2), 3)
    w = WreathCycSn(cyc, perm)
    w_inv = w.inv()
    print('w: {}'.format(w))
    print('w\': {}'.format(w_inv))
    print('I: {}'.format(w*w_inv))
    print('I: {}'.format(w_inv*w))

def test_ysubgroup():
    alpha = (4, 2, 2)
    _parts = ((3, 1), (1,1), (1,1))
    yd = young_subgroup_yor(alpha, _parts) 
    ks = list(yd.keys())
    print('One key of young subgroup yor dict: {}'.format(ks[0]))
    pdb.set_trace()

def test_wreath():
    start = time.time()
    #alpha = (4, 2, 2)
    alpha = (0, 7, 1)
    _parts = ((), (4,3), (1,))
    yd = wreath_yor(alpha, _parts)
    ks = list(yd.keys())
    print('Elapsed: {:.2f}'.format(time.time() - start))
    pdb.set_trace()

if __name__ == '__main__':
    test_wreath() 
