import time
import numpy as np
from yor import yor
from young_tableau import FerrersDiagram
from utils import check_memory, partitions
from perm2 import Perm2, sn

def tensor_rep(ferrers):
    '''
    ferrers: ferrers diagram object. The resulting rep is the representation
    of the tensor product ferrers x ferrers.
    Returns a function f: Perm2 object -> numpy matrix of the representation
    of the given perm 
    '''
    def f(perm):
        return np.kron(yor(ferrers, perm), yor(ferrers, perm))
    return f

def get_irrep_dict(part):
    '''
    part: tuple of ints denoting a conjugacy class
    Returns: dictionary mapping permutation tuples -> irrep
    '''
    
    f = FerrersDiagram(part)
    group = sn(f.size)
    reps = {}
    for p in group:
        reps[p.tup_rep] = yor(f, p)
        
    return reps    

def get_char(irrep_dict, n):
    '''
    irrep dict: dictionary mapping perm tuples (int tuple) -> irrep matrices
    Returns numpy vector of the group size with character evaluations. Order is in order
    of the permutation given by sn (which should be the order fixed by itertools.permutation)
    '''
    char = np.zeros(len(irrep_dict))
    group = sn(n)
    for i in range(len(group)):
        rep = group[i].tup_rep
        char[i] = np.trace(irrep_dict[rep])
    return char

def get_tensor_char(char1, char2):
    '''
    char1: numpy vec of character evaluations of the group
    char2: same as char1
    Returns elementwise product which corresponds to the character evlauation
    of the tensor rep of char1 x char2
    '''
    return char1 * char2

def inner_prod(v1, v2):
    '''
    v1: numpy vector of length {group size}
    Group inner product
    '''
    return np.sum(v1 * v2) / len(v1)

def get_char_decomp(char_dict, new_char, tol=1e-8):
    '''
    char_dict: dictionary mapping conjugacy class to its irreps character evaluation
    new_char: numpy vector of th 
    Returns: dictionary of conj -> multiplicity of that conj class irrep in the new_char
        representation. If a conj class isnt in the dict, the multiplicity is assumed
        to be 0.
    '''
    res = {}
    for p, cvec in char_dict.items():
        # get inner product, see how big/small it is
        iprod = inner_prod(new_char, cvec)
        if iprod > tol:
            res[p] = int(np.round(iprod))
    return res

def get_all_irrep_chars(n=8):
    '''
    Returns a dict mapping conjugacy class -> character vector for the given conj irrep
    for the given sn (n being the function parameter)
    '''
    start = time.time()
    all_chars_8 = {}
    for part in partitions(n):
        irrep_dict = get_irrep_dict(part)
        all_chars_8[part] = get_char(irrep_dict, n)
        print('Elapsed: {:.2f} | {}'.format(time.time() - start, len(all_chars_8)))
    end = time.time()
    print(end - start)
    return all_chars_8


def rep_func(p1, p2):
    f1 = FerrersDiagram(p1)
    f2 = FerrersDiagram(p2)
    def f(perm_tup):
        per = Perm2.from_tup(perm_tup)
        return np.kron(yor(f1, per), yor(f2, per))
    return f

def irrep_func(p):
    fe = FerrersDiagram(p)
    def f(ptup):
        return yor(fe, Perm2.from_tup(ptup))
    
    return f
