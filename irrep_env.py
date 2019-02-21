import sys
import pdb
sys.path.append('./cube/')
from str_cube import *
from cube_env import CubeEnv
from wreath import cyclic_irreps, wreath_rep
from coset_utils import coset_reps, young_subgroup_perm
from perm2 import sn
from cube_irrep import Cube2Irrep
from utils import check_memory

class Cube2IrrepEnv(CubeEnv):
    '''
    This class represents the 2-cube environment but wraps each cube state
    with an irrep(corresponding to alpha, parts) matrix.
    '''
    def __init__(self, alpha, parts, pkl_loc):
        '''
        alpha: tuple of ints, the weak partition of 8 into 3 parts
        parts: tuple of ints, the partitions of each part of alpha
        pkl_loc: string, location of the cached pickle file of S_8 mod S_alpha irreps
        '''
        super(Cube2IrrepEnv, self).__init__(size=2)
        self.alpha = alpha
        self.parts = parts
        self._cubeirrep = Cube2Irrep(alpha, parts, pkl_loc)

    def reset(self):
        state = super(Cube2IrrepEnv, self).reset()
        return self.convert_irrep(state)

    def step(self, action):
        state, rew, done, _dict = super(Cube2IrrepEnv, self).step(action)
        irrep_state = self.convert_irrep(self.state)
        return irrep_state, rew, done, _dict

    def convert_irrep(self, cube_state):
        '''
        state: string representing 2-cube state
        Returns: numpy matrix
        '''
        rep = self._cubeirrep.str_to_irrep(cube_state)
        return rep

def test():
    alpha = (2,3,3)
    parts = ((2,), (1, 1, 1), (1, 1, 1))
    loc = '/local/hopan/cube/pickles/{}/{}.pkl'.format(alpha, parts)
    env = Cube2IrrepEnv(alpha, parts, loc)

    res = env.reset()
    env.step(0)
    env.render()
    check_memory()

if __name__ == '__main__':
    test()
