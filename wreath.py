from multiprocessing import Manager, Process
import os
import sys
import time
from functools import reduce
import pdb
from yor import load_yor
import numpy as np
from scipy.sparse import csr_matrix
from utils import check_memory, chunk
import perm2
from coset_utils import coset_reps, young_subgroup_perm, young_subgroup, tup_set
import torch

def dot(perm, cyc):
    '''
    Have the permutation act on the cyclic irrep
    perm: Perm2
    cyc: tuple or CyclicGroup object (this works bc CyclicGroup object implemented __getitem__)
    Returns: CyclicGroup o
    '''
    p_inv = perm.inv()
    new_cyc = tuple(cyc[p_inv[i] - 1] for i in range(1, len(cyc) + 1))
    p_cyc = CyclicGroup(new_cyc, cyc.order)
    return p_cyc

def dot_tup(perm, tup):
    p_inv = perm.inv()
    new_tup = tuple(tup[p_inv[i] - 1] for i in range(1, len(tup) + 1))
    return new_tup

def dot_tup_inv(perm, tup):
    '''
    perm: Perm2 object
    tup: tuple of ints
    Returns: tuple of ints
    '''
    new_tup = tuple(tup[perm[i] - 1] for i in range(1, len(tup) + 1))
    return new_tup

def block_cyclic_irreps(tup, coset_reps, cyclic_irrep_func):
    '''
    tup: tuple
    coset_reps: list of Perm2 objects
    cyclic_irrep_func: function from tuple -> cyclic_irrep
    Return a dictionary mapping coset rep index -> cyclic irrep
    Ex:
        coset_reps = [pi_0, pi_1, ...]
        Returns: {0: cyclic_irrep_func(f \dot pi_0), 1: cyclic_irrep_func(f \dot pi_1), ...}
    '''
    scalars = np.zeros(len(coset_reps), dtype=np.complex64)
    for idx, rep in enumerate(coset_reps):
        tup_g = dot_tup_inv(rep, tup) # original tuple is f(x) |-> f(g(x))
        scalars[idx] = cyclic_irrep_func(tup_g)

    return scalars

class CyclicGroup:
    def __init__(self, cyc, order):
        self.cyc = cyc
        self.size = len(cyc)
        self.order = order

    @property
    def tup_rep(self):
        return self.cyc

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

    @property
    def tup_rep(self):
        return self.cyc.tup_rep, self.perm.tup_rep

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

    def inv_tup_rep(self):
        return self.inv().tup_rep

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
    idx0 = weak_partition[0]
    idx1 = weak_partition[0] + weak_partition[1]
    g_x = ((0,) * weak_partition[0]) + ((1,) * weak_partition[1]) + ((2,) * weak_partition[2])
    def func(tup):
        p1 = sum(tup[:idx0])
        p2 = sum(tup[idx0: idx1])
        p3 = sum(tup[idx1:])
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
    #assert len(alpha) == 3, 'alpha must be length 3'
    assert sum(alpha) == sum(map(sum, _parts)), 'alpha must be length 3'

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
                tiinv_g_tj = t_i.inv() * g * t_j
                if tiinv_g_tj.tup_rep in young_sub_set:
                    g_rep[(i, j)] = young_yor[tiinv_g_tj.tup_rep]
                    break
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

    sn_chunks = chunk(_sn, par)
    manager = Manager()
    rep_dict = manager.dict()
    nprocs = []

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

    # this part can be parallelized
    # loop over the group
    # things we need are: group element inv, group element multiplication
    # then grabbing the yor for the appropriate yor thing
    for g in _sn:
        g_rep = {}
        for i, t_i in enumerate(reps):
            for j, t_j in enumerate(reps):
                tiinv_g_tj = t_i.inv() * g * t_j
                if tiinv_g_tj.tup_rep in young_sub_set:
                    g_rep[(i, j)] = young_yor[tiinv_g_tj.tup_rep]
                    break

        rep_dict[g.tup_rep] = g_rep 

    return rep_dict

def get_mat(g, yor_dict, block_scalars=None):
    '''
    g: perm2.Perm2 object or tuple representation of a permutation
    yor_dict: dict mapping permutation tuple -> (dict of (i, j) -> numpy matrix)

    Returns matrix for this ydict
    '''
    if type(g) == tuple:
        yg = yor_dict[g]
    else:
        yg = yor_dict[g.tup_rep]
    vs = list(yg.values())
    block_size = vs[0].shape[0]
    size = len(yg) * block_size
    mat = np.zeros((size, size), dtype=np.complex64)
    #mat = np.zeros((size, size))

    for (i, j), v in yg.items():
        x1, x2 = (block_size*i, block_size*i+block_size)
        y1, y2 = (block_size*j, block_size*j+block_size)
        if block_scalars is None:
            scalar = 1
        else:
            scalar = block_scalars[i]
        mat[x1:x2, y1:y2] = scalar * v

    return mat

def convert(idx, size):
    return idx[1] + (size * idx[0])

# TODO: This is a hack
def get_sparse_mat(g, yor_dict, block_scalars=None, shape=None):
    '''
    g: group element or perm tuple
    yor_dict: dictionary of perm tuples -> dictionaries of block idx -> matrix
    block_scalars: map of scalars multipliers for each block
    shape: shape of the output sparse tensor
    Returns: torch sparse tensor
    '''
    if type(g) == tuple:
        yg = yor_dict[g]
    else:
        yg = yor_dict[g.tup_rep]

    if shape is None:
        vs = list(yg.values())
        block_size = vs[0].shape[0]
        size = len(yg) * block_size
        shape = torch.Size([1, size * size])
    else:
        size = shape[0]
        shape = torch.Size([1, size * size])

    idxs = torch.LongTensor([(0, convert(idx, size)) for idx in yg.keys()]).t()
    re_vals = []
    im_vals = []
    for idx, mat in yg.items():
        scalar = 1 if block_scalars is None else block_scalars[idx[0]]
        # this is a hack that only works for the 1 dim blocks
        # but we don't use get_sparse_mat so this is to be deprecated anyhow
        re_vals.append(scalar.real * mat[0, 0])
        im_vals.append(scalar.imag * mat[0, 0])

    ret = torch.FloatTensor(re_vals)
    imt = torch.FloatTensor(im_vals)
    sparse_re = torch.sparse.FloatTensor(idxs, ret, shape)
    sparse_im = torch.sparse.FloatTensor(idxs, imt, shape)
    return sparse_re, sparse_im

def wreath_rep(cyc_tup, perm, yor_dict, cos_reps, cyc_irrep_func=None, alpha=None):
    if (cyc_irrep_func is None) and (alpha is None):
        raise ValueError('Need to supply either irrep func or alpha')
    if cyc_irrep_func is None:
        cyc_irrep_func = cyclic_irreps(alpha)

    block_scalars = block_cyclic_irreps(cyc_tup, cos_reps, cyc_irrep_func)
    return get_mat(perm, yor_dict, block_scalars)

def wreath_rep_sp(cyc_tup, perm_tup, sp_irrep_dict, cos_reps, cyc_irrep_func, cyc_irr_dict=None):
    # i actually want these block scalars ordered ...
    if cyc_irr_dict is None:
        block_scalars = block_cyclic_irreps(cyc_tup, cos_reps, cyc_irrep_func)
    else:
        block_scalars = cyc_irr_dict[cyc_tup]

    # block multiply
    nblocks = len(cos_reps)
    # TODO: this is a hack
    block_size = sp_irrep_dict[(1,2,3,4,5,6,7,8)].shape[0] // nblocks
    sp_mat = sp_irrep_dict[perm_tup]

    new_data = np.zeros(sp_mat.data.shape, dtype=np.complex128)
    for idx, c in enumerate(block_scalars):
        st = sp_mat.indptr[idx * block_size]
        end = sp_mat.indptr[idx * block_size + block_size]
        new_data[st: end] = c * sp_mat.data[st: end]

    return csr_matrix((new_data, sp_mat.indices, sp_mat.indptr), shape=sp_mat.shape)

if __name__ == '__main__':
    pass
