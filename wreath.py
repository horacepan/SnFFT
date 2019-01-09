from functools import reduce
from collections import Counter
import itertools
import pdb
from young_tableau import FerrersDiagram
from yor import yor, load_yor
import numpy as np
from utils import partitions, weak_partitions
import perm2

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

def perm_from_young_tuple(cyc_tup):
    n = sum(map(len, cyc_tup))
    lst = [0] * n
    idx = 0
    for cyc in cyc_tup:
        for x in cyc:
            lst[idx] = x
            idx += 1

    return perm2.Perm2.from_lst(lst)

def young_subgroup(weak_partition):
    '''
    weak_partition: tuple of ints

    Let alpha be the weak partition
    Returns the generator for the product group S_alpha_0 x S_alpha_1 x ... x S_alpha_k
    '''
    #sym_subgroups = [perm2.sn(p) for p in weak_partition if p > 0]
    sym_subgroups = [itertools.permutations(range(1, p+1)) for p in weak_partition if p > 0]
    return itertools.product(*sym_subgroups)

def young_subgroup_perm(weak_partition):
    return [perm_from_young_tuple(t) for t in young_subgroup_canonical(weak_partition)]

def young_subgroup_canonical(weak_partition):
    '''
    This is not quite right...
    '''
    subgroups = []
    idx = 1
    for p in weak_partition:
        if p == 0:
            continue
        subgroups.append(itertools.permutations(range(idx, idx+p)))
        idx += p

    #sym_subgroups = [itertools.permutations(range(1, p+1)) for p in weak_partition if p > 0]
    return itertools.product(*subgroups)

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
    for g in young_subgroup(alpha):
        # length of g should be length of nonzero_parts
        ms = [yd[perm] for yd, perm in zip(nonzero_parts, g)]
        wreath_dict[g] = reduce(np.kron, ms)

    return wreath_dict

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

if __name__ == '__main__':
    perm = perm2.Perm2.from_tup((1,3,4,2))
    cyc = CyclicGroup((0, 1, 0, 1), 3)
    w = WreathCycSn(cyc, perm)
    w_inv = w.inv()
    print('w: {}'.format(w))
    print('w\': {}'.format(w_inv))
    print('I: {}'.format(w*w_inv))
    print('I: {}'.format(w_inv*w))
