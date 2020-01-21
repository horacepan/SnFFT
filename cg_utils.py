import pdb
import numpy as np
from scipy import linalg as splinalg

def in_null(mat, vec):
    return np.allclose(0, np.matmul(mat, vec.reshape((-1, 1))))

def block_rep(mat, mult):
    '''
    mat: square matrix
    mult: num times to direct sum the matrix
    Returns \otimes^mult mat
    '''
    n = mat.shape[0]
    blocked = np.zeros((n*mult, n*mult))
    for i in range(mult):
        blocked[i*n: i*n + n, i*n: i*n+n] = mat
    return blocked

def random_null_vec(mat, dim):
    '''
    mat: n x k matrix
    Return: random vector in the nullspace of mat
    '''
    nvecs = splinalg.null_space(mat)
    nbasis = nvecs[:, -dim:]
    weights = (np.random.random(dim) * 2) - 1 # [-1, 1]
    vec = (nbasis * weights).sum(axis=1)
    return vec

def rand_null_vec(mat):
    nbasis = splinalg.null_space(mat)
    dims = nbasis.shape[1]
    weights = np.random.random(dims)
    vec = (nbasis * weights).sum(axis=1)
    return vec

def kron_factor_id(mat, dim):
    size = mat.shape[0] // dim
    m = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            _i = i * dim
            _j = j * dim
            m[i, j] = mat[_i, _j]
    return m

def normalize_rows(mat):
    row_norms = np.sqrt(np.square(mat).sum(axis=1))
    return mat / row_norms[:, np.newaxis]

def test_block_rep(bmat, mat, mult):
    '''
    Check that bmat is in fact \oplus^mult mat
    '''
    n = mat.shape[0]
    for i in range(mult):
        for j in range(mult):
            block = bmat[i*n: i*n + n, j*n: j*n + n]
            if i == j:
                if not np.allclose(block, mat):
                    return False
            else:
                if not np.allclose(block, 0):
                    return False
    return True

def intertwine(trep1, trep2, irrep1, irrep2, mult, verbose=False):
    '''
    trep1: rep of a generator (often a tensor of irreps)
    trep2: rep of a different generator (often a tensor of irreps)
    irrep1: irrep of a generator
    irrep2: irrep of the second generator
    mult: multiplicity of the given irrep in trep1/trep2
    Returns: intertwiner between the given reps
    '''
    d = trep1.shape[0]
    dnu = irrep1.shape[0]
    bxrep1 = block_rep(irrep1, mult)
    bxrep2 = block_rep(irrep2, mult)
    
    k1 = np.kron(np.eye(d), bxrep1) - np.kron(trep1.T, np.eye(mult * dnu))
    k2 = np.kron(np.eye(d), bxrep2) - np.kron(trep2.T, np.eye(mult * dnu))
    k = np.concatenate([k1, k2], axis=0)

    rand_null = random_null_vec(k, mult * mult)
    R = np.reshape(rand_null, (mult * dnu, d), 'F') # this reshapes columnwise
    RRT = np.matmul(R, R.T)

    M = kron_factor_id(RRT, dnu)
    _, evecs = np.linalg.eig(M)
    S = np.kron(evecs, np.eye(dnu))
    output = normalize_rows(np.matmul(S.T, R))
 
    if verbose:
        print('rand null in null', in_null(k, rand_null))
        print('R intertwines g1?', np.allclose(np.matmul(R, trep1), np.matmul(bxrep1, R)))
        print('R intertwines g2?', np.allclose(np.matmul(R, trep2), np.matmul(bxrep2, R)))
        print('rrt is commutant?', np.allclose(np.matmul(RRT, bxrep1), np.matmul(bxrep1, RRT)))
        print('rrt is commutant 1?', np.allclose(np.matmul(RRT, bxrep1), np.matmul(bxrep1, RRT)))
        print('rrt is commutant 2?', np.allclose(np.matmul(RRT, bxrep2), np.matmul(bxrep2, RRT)))
    return output 
