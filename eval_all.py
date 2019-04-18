import pdb
from dqn import IrrepLinreg, value, value_tup
import numpy as np
import os
from tqdm import tqdm
from irrep_env import Cube2IrrepEnv
from utils import check_memory, partition_parts
import ast

# only do this on a subset of the cubes?
if os.path.exists('/local/hopan')
    CUBE_FILE = '/local/hopan/cube/split_unmod/xaa.split'
    FOURIER_DIR = '/local/hopan/cube/fourier_unmod/'
    SAVE_DIR = '/local/hopan/cube/fourier_eval'
elif os.path.exists('/scratch/hopan'):
    CUBE_FILE = '/scratch/hopan/cube/split_unmod/xaa.split'
    FOURIER_DIR = '/scratch/hopan/cube/fourier_unmod/'
    SAVE_DIR = '/scratch/hopan/cube/fourier_eval'
else:
    CUBE_FILE = '/project2/risi/cube/split_unmod/xaa.split'
    FOURIER_DIR = '/project2/risi/cube/fourier_unmod/'
    SAVE_DIR = '/project2/risi/cube/fourier_eval'

def compute(alpha, fourier_dir, save_dir):
    fourier_dir = os.path.join(fourier_dir, str(alpha))
    save_dir = os.path.join(save_dir, str(alpha))

    if not os.path.exists(save_dir):
        print('Making {}'.format(save_dir))
        os.makedirs(save_dir)

    with open(CUBE_FILE, 'r') as fcube:
        for parts in partition_parts(alpha):
            parts_dir = os.path.join(save_dir, '{}.txt'.format(parts))
            if os.path.exists(parts_dir):
                print('Skipping {}'.format(parts_dir))
                continue

            print('Loading env')
            env = Cube2IrrepEnv(alpha, parts)
            print('Done loading env')
            np_file = os.path.join(fourier_dir, str(parts) + '.npy')
            mat = np.load(np_file)
            model = IrrepLinreg(mat.size)
            model.setnp(mat)
            check_memory()
            with open(parts_dir, 'w') as f:
                for line in tqdm(fcube):
                    line = line.strip().split(',')
                    otup = tuple(int(x) for x in line[0])
                    ptup = tuple(int(x) for x in line[1])
                    re, im = value_tup(model, env, otup, ptup)
                    f.write('{},{},{}\n'.format(line[0], line[1], re))
 
            fcube.seek(0)
            check_memory()
            del env

def main():
    alpha = (4, 2, 2)
    compute(alpha, FOURIER_DIR, SAVE_DIR)

if __name__ == '__main__':
    main()
