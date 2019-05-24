import pdb
import sys
sys.path.append('../')
from young_tableau import FerrersDiagram
from yor import yor
from perm2 import Perm2, sn
from tile_env import TileEnv, neighbors, env_neighbors
import numpy as np
from gym import spaces

class SnIrrep:
    def __init__(self, partition):
        self.partition = partition
        self.ferrers = FerrersDiagram(partition)

    def grid_irrep(self, grid):
        ptup = tuple(i for row in grid for i in row)
        pinv = Perm2.from_tup(ptup).inv()
        return self.perm_irrep(pinv)

    def irrep(self, tup):
        # TODO: maybe avoid cache?
        return yor(self.ferrers, Perm2.from_tup(tup)).ravel()

    def perm_irrep(self, perm):
        return yor(self.ferrers, perm).ravel()

class TileIrrepEnv(TileEnv):
    def __init__(self, n, partitions):
        '''
        n: int, size of tile puzzle
        partitions: list of partitions of 9(tuples of ints)
        '''
        super(TileIrrepEnv, self).__init__(n, one_hot=False) # want to get the grid states
        self.n = n
        self.partitions = partitions
        self.ferrers = [FerrersDiagram(p) for p in partitions]
        self.sn_irreps = [SnIrrep(p) for p in partitions]

        # will only know this from sn_irreps cat irreps!
        irrep_shape = self.cat_irreps(TileEnv.solved_grid(n)).shape
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=irrep_shape)

    def step(self, action):
        grid_state, reward, done, info = super(TileIrrepEnv, self).step(action)
        cat_irrep = self.cat_irreps(grid_state)
        return cat_irrep, reward, done, info

    # TODO: maybe manually get the pinv here?
    def cat_irreps(self, grid):
        '''
        grid: n x n numpy matrix
        Returns the concattenated numpy vector for the given grid state.
        '''
        ptup = tuple(i for row in grid for i in row)
        pinv = Perm2.from_tup(ptup).inv()
        return np.concatenate([s.perm_irrep(pinv) for s in self.sn_irreps])

    def reset(self):
        grid_state = super(TileIrrepEnv, self).reset()
        return self.cat_irreps(grid_state)

    def neighbors(self):
        '''
        Returns neighboring puzzle states' irrep vector
        '''
        nbrs = super(TileIrrepEnv, self).neighbors()
        irrep_nbrs = {}
        for a, nb_grid in nbrs.items():
            irrep_nbrs[a] = self.cat_irreps(nb_grid)

        return irrep_nbrs

def test():
    n = 3
    partitions = [(9,), (8, 1)]
    env = TileIrrepEnv(n, partitions)
    irrep_state = env.reset()
    nbrs = env.neighbors()

if __name__ == '__main__':
    test()
