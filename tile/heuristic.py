import pdb
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../')
from perm2 import sn, Perm2
from young_tableau import FerrersDiagram
from yor import yor

S9_SIZE = 362880
CORRECT_IDX = {
    2: {
        1: (0, 0),
        2: (0, 1),
        3: (1, 0),
        4: (1, 1),
    },
    3: {
        1: (0, 0),
        2: (0, 1),
        3: (0, 2),
        4: (1, 0),
        5: (1, 1),
        6: (1, 2),
        7: (2, 0),
        8: (2, 1),
        9: (2, 2),
    },
    4: {
        1: (0, 0),
        2: (0, 1),
        3: (0, 2),
        4: (0, 3),
        5: (1, 0),
        6: (1, 1),
        7: (1, 2),
        8: (1, 3),
        9: (2, 0),
        10: (2, 1),
        11: (2, 2),
        12: (2, 3),
        13: (3, 0),
        14: (3, 1),
        15: (3, 2),
        16: (3, 3),
    }
}

def hamming(tup):
    '''
    tup is given in: (sigma(1), ..., sigma(n)) format
    '''
    return sum(0 if ((idx+1) == i) else 1 for idx, i in enumerate(tup))

def hamming_grid(grid):
    return hamming(grid.ravel())

def manhattan(tup):
    n = int(np.sqrt(len(tup)))
    grid = np.array(tup).reshape(n, n)

    # loop over grid. for each item in grid, compute how far it is from correct spot
    tot_dist = 0
    for i in range(n):
        for j in range(n):
            val = grid[i, j]
            cx, cy = CORRECT_IDX[n][val]
            #print('Val: {} | At index: {}, {} | Correct index: {}, {}'.format(val, i, j, cx, cy))
            tot_dist += (abs(cx-i) + abs(cy-j))

    #print('tup: {} | dist: {}'.format(tup, tot_dist))
    return tot_dist

def manhattan_grid(grid):
    return manhattan(grid.ravel())

def tup_to_str(t):
    return ''.join(str(i) for i in t)

def write_feval(feval, saveloc):
    perms = sn(9)
    with open(saveloc, 'w') as f:
        for p in tqdm(perms):
            ptup = p.tup_rep
            f.write('{},{}\n'.format(tup_to_str(ptup), feval(ptup)))

def main():
    write_feval(manhattan, '/local/hopan/tile/manhattan_s9.txt')
    write_feval(hamming, '/local/hopan/tile/hamming_s9.txt')


def test_hamming():
    assert hamming((1, 2, 3, 4)) == 0
    assert hamming((1, 2, 4, 3)) == 2
    assert hamming((1, 2, 3, 3)) == 1
    print('Hamming okay')

def test_manhattan():
    assert manhattan((1, 2, 3, 4)) == 0
    assert manhattan((1, 2, 4, 3)) == 2
    assert manhattan((4, 2, 3, 1)) == 4

    assert manhattan((1, 2, 3, 4, 5, 6, 7, 8, 9)) == 0
    assert manhattan((1, 2, 3, 4, 5, 6, 9, 8, 7)) == 4
    assert manhattan((1, 2, 3, 4, 5, 6, 7, 9, 8)) == 2
    assert manhattan((2, 3, 1, 4, 5, 6, 7, 9, 8)) == 6

    print('Manhattan okay')

def irrep_gen_func(partitions, _dir='fourier_eval'):
    fmats = {}
    ferrers = {}
    for p in partitions:
        # load the appropriate matrix
        f = FerrersDiagram(p)
        # TODO: this should load from the appropriate place
        #fourier_mat = np.load('./tile/{}/{}.npy'.format(_dir, p))
        try:
            fourier_mat = np.load('/local/hopan/tile/{}/{}.npy'.format(_dir, p))
        except:
            fourier_mat = np.load('/scratch/hopan/tile/{}/{}.npy'.format(_dir, p))
        scale = fourier_mat.shape[0] / S9_SIZE
        fmats[p] = scale * fourier_mat
        ferrers[p] = f

    def func(grid):
        tup = tuple(i for row in grid for i in row)
        val = 0
        pinv = Perm2.from_tup(tup).inv()
        for p, mat in fmats.items():
            f = ferrers[p]
            mat_inv = yor(f, pinv)
            val += np.sum(mat_inv * mat)

        return val

    return func

def test():
    test_hamming()
    test_manhattan()


if __name__ == '__main__':
    main()
