import itertools
import matplotlib.pyplot as plt
import pdb
from young_tableau import FerrersDiagram
from yor import yor, load_yor
import numpy as np
from utils import partitions, weak_partitions
from perm2 import sn as sn

class EmptyGroup:
    def __next__(self):
        return None
 
class WreathIrrep:
    def __init__(self, alpha, alpha_parts):
        '''
        This class is an abstraction for the wreath product group of Z/kZ wreathed with S_n
        alpha: weak partition
        alpha_parts: dict of int to lists/tuples. Length of alpha_parts is of length nnz of alpha
                     The keys of this dictionary are the irreps of Z/kZ (0, 1, ..., k-1)

        Ex:
            alpha = (2, 0, 1)
            alpha_parts = {0: (1,1), 2: (1,)}
        '''
        self.cyclic_order = len(alpha)
        self.n = sum(alpha)
        self.alpha = alpha
        self.alpha_parts = alpha_parts
        self.irreps = self._gen_irreps()

    def _gen_irreps(self):
        irrep_dict = {}
        for cyc_irrep, part in self.alpha_parts.items():
            ferrers = FerrersDiagram(part)
            irrep_dict[cyc_irrep] = yor(ferrers)

def wreath_irrep(n, cyclic_order, weak_partition, partitions):
    '''
    We compute an irrep of the wreath product group: Z/cZ \wreath S_n
    Recall that the irreps of the wreath product group are indexed by
    weak partitions and partitions of the non-zero parts of the weak partition

    The cyclic order gives us the number of irreps
    n: integer. Size of the symmetric group
    cyclic_order: size of the cyclic group
    weak_partition: tuple of nonnegative ints. The contents should sum to n
    partitions: list of partitions of the non-zero parts of the weak partition

    Ex usage:
        wreath_irrep(5, 2, (3, 2), [(2,1), (1,1)])
        wreath_irrep(6, 3, (4, 0, 2), [(2,1,1), (1,1)])
    '''

    cyc_irreps = []
    for i, cnt in enumerate(weak_partition):
        if cnt > 0:
            cyc_irreps.append(np.exp(2j*np.pi*i) / cyc_order)

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

def wreath_yor(weak_partition, _partitions, cyc_element, permutation):
    '''
    weak_partition: tuple of nonnegative ints. The contents should sum to n
    partitions: list of partitions of the non-zero parts of the weak partition
    cyc_element: tuple of ints, where each int is in the range 0, ..., n-1
    permutation: Perm object
    '''
    n = sum(weak_partition)
    cyc_irreps = cyclic_irreps(weak_partition)
    cyc_part = np.diag(cyc_irreps)
    perm_matrix = permutation.matrix()

    yor_parts = []
    for part in _partitions:
        f = FerrersDiagram(part)
        y = yor(ferrers, permutation)

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

def get_mat(yor_dict, perm):
    return yor_dict[perm]

def young_subgroup(weak_partition):
    '''

    '''
    #sym_subgroups = [sn(p) for p in weak_partition if p > 0]
    sym_subgroups = [itertools.permutations(range(1, p+1)) for p in weak_partition if p > 0]
    return itertools.product(*sym_subgroups)

def young_subgroup_yor(alpha, _parts):
    '''
    Compute the wreath product group irreps of S_alpha]
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
        ms = [get_mat(yd, perm) for yd, perm in zip(nonzero_parts, g)]
        wreath_dict[g] = np.kron(*ms)

    return wreath_dict

def compute_dims():
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
    #plt.hist(all_dims, bins=20)
    #plt.show()
    srted_dict.sort(reverse=True)
    with open('dims.txt', 'w') as f:
        for d, ps in srted_dict:
            f.write('{:2} | {}, {}, {}\n'.format(str(d), str(ps[0]), str(ps[1]), str(ps[2])))
    pdb.set_trace()

if __name__ == '__main__':
    d = young_subgroup_yor((0, 4, 4), [(), (2, 2), (3, 1)])
    print(len(d))
