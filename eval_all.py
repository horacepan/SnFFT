import sys
import pdb
from dqn import IrrepLinreg, value, value_tup, IrrepLinregNP, value_tup_np
import numpy as np
import os
from tqdm import tqdm
from irrep_env import Cube2IrrepEnv
from utils import check_memory, partition_parts
import argparse
sys.path.append('./cube')
from str_cube import get_wreath

# only do this on a subset of the cubes?
if os.path.exists('/local/hopan'):
    CUBE_FILE = '/local/hopan/cube/cube_sym_mod.txt'
    #CUBE_FILE = '/local/hopan/cube/split_unmod/xaa.split'
    FOURIER_DIR = '/local/hopan/cube/fourier_unmod/'
    SAVE_DIR = '/local/hopan/cube/fourier_eval_sym'
elif os.path.exists('/scratch/hopan'):
    CUBE_FILE = '/scratch/hopan/cube/cube_sym_mod.txt'
    #CUBE_FILE = '/scratch/hopan/cube/split_unmod/xaa.split'
    FOURIER_DIR = '/scratch/hopan/cube/fourier_unmod/'
    SAVE_DIR = '/scratch/hopan/cube/fourier_eval_sym'
else:
    CUBE_FILE = '/project2/risi/cube/cube_sym_mod.txt'
    #CUBE_FILE = '/project2/risi/cube/split_unmod/xaa.split'
    FOURIER_DIR = '/project2/risi/cube/fourier_unmod/'
    SAVE_DIR = '/project2/risi/cube/fourier_eval_sym'

def main(args, fourier_dir=FOURIER_DIR, save_dir=SAVE_DIR):
    alpha = args.alpha
    max_dist = args.dist
    fourier_dir = os.path.join(fourier_dir, str(alpha))
    save_dir = save_dir + str(args.dist)
    save_dir = os.path.join(save_dir, str(alpha))

    if not os.path.exists(save_dir):
        print('Making {}'.format(save_dir))
        os.makedirs(save_dir)
    peak_memory = 0

    with open(CUBE_FILE, 'r') as fcube:
        for parts in partition_parts(alpha):
            parts_file = os.path.join(save_dir, '{}.txt'.format(parts))
            if os.path.exists(parts_file):
                print('Skipping {}'.format(parts_file))
                continue

            print('Loading env')
            try:
                env = Cube2IrrepEnv(alpha, parts, numpy=args.numpy, sparse=args.sparse)
            except:
                print('Skipping. Cant load env for {} | {}'.format(alpha, parts))
                continue
            print('Done loading env')
            np_file = os.path.join(fourier_dir, str(parts) + '.npy')

            if args.numpy:
                model = IrrepLinregNP(np_file)
            else:
                mat = np.load(np_file)
                model = IrrepLinreg(mat.size)
                model = IrrepLinregNP(np_file)
                model.setnp(mat)

            peak_memory = max(check_memory(), peak_memory)
            with open(parts_file, 'w') as f:
                idx = 0
                for line in (fcube):
                    line = line.strip().split(',')
                    dist = int(line[-1])
                    if dist > max_dist:
                        print('Hit max dist at index: {} | cube: {}'.format(idx, line))
                        break
                    if '.split' in CUBE_FILE:
                        otup = tuple(int(x) for x in line[0])
                        ptup = tuple(int(x) for x in line[1])
                    else:
                        otup, ptup = get_wreath(line[0])
                    if args.numpy:
                        re, im = value_tup_np(model, env, otup, ptup)
                    elif args.sparse:
                        re, im = value_tup(model, env, otup, ptup)

                    f.write('{},{},{},{}\n'.format(line[0], line[1], re, im))
                    idx += 1

            print('Done with {} | {}'.format(alpha, parts))
            fcube.seek(0)
            peak_memory = max(check_memory(), peak_memory)
            del env
    print('Peak mem usg: {}mb'.format(peak_memory))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=str)
    parser.add_argument('--dist', type=int)
    parser.add_argument('--numpy', action='store_true', default=False)
    parser.add_argument('--sparse', action='store_true', default=False)
    args = parser.parse_args()
    args.alpha = eval(args.alpha)
    main(args)
