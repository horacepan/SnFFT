import sys
import pickle
import pdb
sys.path.append('./cube')

import numpy as np
from str_cube import get_wreath, scramble_fixedcore, init_2cube
from utils import check_memory, get_logger
from irrep_env import Cube2IrrepEnv
from dqn import IrrepLinreg, IrrepLinregNP
import pickle
from compute_policy import load_pkls

log = get_logger()

def log_mem(log):
    mem = check_memory(False)
    log.info('Memory usage: {:.2f}mb'.format(mem))

def eval_cube(c, mat, model_np, model_sp, env_s, env_np, pkls):
    idx_to_nbrs, idx_to_cube, idx_to_dist, cube_to_idx = pkls
    otup, ptup = get_wreath(c)
    cidx = cube_to_idx[c]
    np_mat = env_np.tup_irrep_inv_np(otup, ptup)
    sp_r, sp_i = env_s.irrep_inv(c)

    # numpy model actually computes the transpose of the matrix
    npr, npi = model_np.forward(np_mat.T)
    spr, spi = model_sp.forward(sp_r, sp_i)
    true_val = mat[cidx]
    log.info('True: {:.4f} | NP: {:.4f} | SP: {:.4f}'.format(true_val, npr, spr.item()))

def main():
    log.info('Starting...')
    alpha = (2, 3, 3)
    parts = ((2,), (1, 1, 1), (1, 1, 1))
    env_s = Cube2IrrepEnv(alpha, parts, sparse=True)
    log.info('Done loading sparse env...')
    log_mem(log)

    env_np = Cube2IrrepEnv(alpha, parts, numpy=True)
    log.info('Done loading np env...')
    log_mem(log)

    model = IrrepLinreg.from_np(alpha, parts)
    model_np = IrrepLinregNP('/local/hopan/cube/fourier_unmod/{}/{}.npy'.format(alpha, parts))
    log_mem(log)
    log.info('Done loading models. Now loading pkls')

    pkls = load_pkls()
    log.info('Done loading pkls')
    log_mem(log)

    mat = np.load('/local/hopan/cube/fourier_eval_sym_all/{}/{}.npy'.format(alpha, parts)).real
    cubes = [scramble_fixedcore(init_2cube(), 100) for _ in range(20)]

    for c in cubes:
        eval_cube(c, mat, model_np, model, env_s, env_np, pkls)

if __name__ == '__main__':
    main()
