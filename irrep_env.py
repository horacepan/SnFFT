import sys
import time
import pdb
sys.path.append('./cube/')
from str_cube import *
from cube_env import CubeEnv
from wreath import cyclic_irreps, wreath_rep
from coset_utils import coset_reps, young_subgroup_perm
from perm2 import sn
from cube_irrep import Cube2Irrep
from utils import check_memory
import numpy as np
import torch

class Cube2IrrepEnv(CubeEnv):
    '''
    This class represents the 2-cube environment but wraps each cube state
    with an irrep(corresponding to alpha, parts) matrix.
    '''
    def __init__(self, alpha, parts):
        '''
        alpha: tuple of ints, the weak partition of 8 into 3 parts
        parts: tuple of ints, the partitions of each part of alpha
        '''
        super(Cube2IrrepEnv, self).__init__(size=2)
        self.alpha = alpha
        self.parts = parts
        self._cubeirrep = Cube2Irrep(alpha, parts)

    def reset(self):
        state = super(Cube2IrrepEnv, self).reset()
        return state

    def step(self, action, irrep=False):
        state, rew, done, _dict = super(Cube2IrrepEnv, self).step(action)
        if irrep:
            irrep_state = self.convert_irrep_np(self.state)
            _dict['irrep'] = irrep_state
        return state, rew, done, _dict

    def convert_irrep_np(self, cube_state, shape=None):
        '''
        state: string representing 2-cube state
        Returns: numpy matrix
        '''
        if shape is None:
            rep = self._cubeirrep.str_to_irrep_np(cube_state)
            return rep.ravel()
        else:
            return rep.reshape(shape)

    def real_imag_irrep_torch(self, cube_state):
        re, im = self._cubeirrep.str_to_irrep_th(cube_state)
        return re, im

    def real_imag_irrep(self, cube_state):
        irrep = self._cubeirrep.str_to_irrep_np(cube_state).ravel()
        return torch.from_numpy(irrep.real), torch.from_numpy(irrep.imag)

def test(ntrials=100):
    start = time.time()
    alpha = (2,3,3)
    parts = ((2,), (1, 1, 1), (1, 1, 1))

    env = Cube2IrrepEnv(alpha, parts)
    setup_time = time.time() - start
    print('Done loading: {:.2f}s'.format(setup_time))

    res = env.reset()
    stuff = []
    for _ in range(ntrials):
        action = random.choice(range(1, 7))
        res, _, _, _ = env.step(action)
        stuff.append(res)

    check_memory()
    end = time.time()
    sim_time = (end - start) - setup_time
    per_action_time = sim_time / ntrials
    print('Setup time: {:.4f}s'.format(setup_time))
    print('Total time: {:.4f}s'.format(sim_time))
    print('Per action: {:.4f}s'.format(per_action_time))

def test_simple():
    alpha = (2,3,3)
    parts = ((2,), (1,1,1), (1,1,1))
    env = Cube2IrrepEnv(alpha, parts)
    state = env.reset()

if __name__ == '__main__':
    test_simple()
    ntrials = 100 if len(sys.argv) < 2 else int(sys.argv[1])
    test(ntrials)