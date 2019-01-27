from tqdm import tqdm
from itertools import permutations
import time
import pdb
import argparse
import ast
import os
from utils import load_pkl, tf
from fft import cube2_inv_fft_part, cube2_inv_fft_func
import numpy as np

FOURIER_SUBDIR = 'fourier'
IRREP_SUBDIR = 'pickles'
    
def main():
    '''
    Loads the fourier and irrep dict and then computes the inverse fourier transform
    portion for this irrep.
    '''
    parser = argparse.ArgumentParser(description='Test drive inverse fft')
    parser.add_argument('--cubedir', type=str, default='/local/hopan/cube/')
    parser.add_argument('--alpha', type=str, default='(0, 4, 4)')
    parser.add_argument('--parts', type=str, default='((),(4,),(3,1))')
    args = parser.parse_args()

    alpha = ast.literal_eval(args.alpha)
    parts = ast.literal_eval(args.parts)
    start = time.time()
    irrep_path = os.path.join(args.cubedir, IRREP_SUBDIR, '{}/{}.pkl'.format(alpha, parts))
    fourier_npy = os.path.join(args.cubedir, FOURIER_SUBDIR, '{}/{}.npy'.format(alpha, parts))

    fourier_mat = np.load(fourier_npy)
    irreps = load_pkl(irrep_path, 'rb')
    end = time.time()
    ift = cube2_inv_fft_func(irreps, fourier_mat, alpha, parts)

    print('Fourier matrix size: {}'.format(fourier_mat.shape))
    pstart = time.time()
    perms = 0
    for idx, p in enumerate(tqdm(permutations(range(1, 9)))):
        ift(p)
        perms += 1
    pend = time.time()

    print('Computed {} inverse ffts'.format(perms))
    print('Load time: {:.2f}'.format(end - start))
    print('Time per perm: {:.2f}'.format((pend - pstart)/perms))
    print('Total time for s8: {:.2f}'.format(pend - pstart))

if __name__ == '__main__':
    tf(main, [])
