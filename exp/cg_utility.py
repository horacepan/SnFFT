import pdb
import os
import sys
import pickle

import numpy as np
import torch
sys.path.append('../')
from young_tableau import FerrersDiagram
from utility import load_yor, cg_mat, th_kron
torch.set_default_tensor_type(torch.DoubleTensor)

def proj(c1, all_chars):
    mults = {}
    for part, c in all_chars.items():
        inner_prod = np.dot(c1, c) / len(c1)
        prod = int(np.round(inner_prod, decimals=4))
        if prod > 0:
            mults[part] = prod
    return mults

def kron_mults(irreps, basep, char_dict):
    '''
    irreps: list of tuples of irrep partitions
    basep: irrep tuple to compute the multiplicity of p1xp2 -> basep
    char_dict: dict mapping irrep tuple -> np vec of the character evaluations of each group element
    Returns: dictionary mapping (tuple, tuple) -> int (multiplicity of the basep irrep
    '''
    mults = {}
    for p1 in irreps:
        for p2 in irreps:
            p1xp2 = char_dict[p1] * char_dict[p2]
            mult_dict = proj(p1xp2, char_dict)
            mults[(p1, p2)] = mult_dict[basep]
    return mults

def compute_reduced_block(p1, p2, cg, mult, kron_mats):
    '''
    fhat1: nn.Parameter(matrix)
    fhat2: nn.Parameter(matrix)
    cg: cg matrix
    mult: int
    kron_mats: dict mapping tuples of tuples -> tensor
    '''
    kron = kron_mats[(p1, p2)]
    mat = cg.t() @ kron @ cg
    block = cg.shape[1] // mult
    output = torch.zeros((block, block)).double()

    for i in range(mult):
        output += mat[i*block: i*block + block, i*block: i*block + block]

    return output

def compute_rhs_block(fhat1, fhat2, cg, mult, rho):
    kron = th_kron(rho.matmul(fhat1), fhat2)
    mat = cg.t() @ kron @ cg
    block = cg.shape[1] // mult
    output = torch.zeros((block, block)).double()

    for i in range(mult):
        output += mat[i*block: i*block + block, i*block: i*block + block]

    return output

def cg_loss(base_p, partitions, gelement, fhats):
    '''
    base_p: irrep (tuple)
    partitions: list of irreps (tuples)
    gelement: group element (Perm2 object) in the generator
    fhats: dictionary mapping irrep (tuple) -> fourier matrix
    Compute the lhs and rhs of the fourier equality after fourier transforming:
    (f(\tau \sigma) - f(\sigma))^2 = 1
    '''
    s8_chars = pickle.load(open('/local/hopan/irreps/s_8/char_dict.pkl', 'rb'))
    cg_mats = {(p1, p2): torch.from_numpy(cg_mat(p1, p2, base_p)).double() for p1 in partitions for p2 in partitions}
    yors = {p: load_yor(p, '/local/hopan/irreps/s_8') for p in partitions}

    g_size = 40320
    g_size = 1
    dbase = FerrersDiagram(base_p).n_tabs()
    rho1 = torch.from_numpy(yors[base_p][gelement])
    lmat = rho1 + torch.eye(dbase).double()

    lhs = torch.zeros((dbase, dbase)).double()
    rhs = torch.zeros((dbase, dbase)).double()

    for p1 in partitions:
        f1 = FerrersDiagram(p1)
        for p2 in partitions:
            f2 = FerrersDiagram(p2)
            tens_char = s8_chars[p1] * s8_chars[p2]
            mult_dict = proj(tens_char, s8_chars)
            mult = mult_dict[base_p]

            fhat1 = fhats[p1]
            fhat2 = fhats[p2]
            cgmat = cg_mats[(p1, p2)] # d x drho zrho
            reduced_block = compute_reduced_block(fhat1, fhat2, cgmat, mult)

            coeff = (f1.n_tabs() * f2.n_tabs()) / (g_size * dbase)
            lhs += coeff * lmat @ reduced_block
            rhs += 2 * coeff * compute_rhs_block(fhat1, fhat2, cgmat, mult, torch.from_numpy(yors[p1][gelement]).double())

    return ((lhs - rhs).pow(2)).mean()
