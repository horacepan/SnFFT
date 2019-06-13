import pdb
import sys
sys.path.append('../')
from young_tableau import FerrersDiagram
from yor import yor
from perm2 import Perm2, sn
from tile_env import TileEnv, neighbors, env_neighbors, grid_to_onehot
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
    def __init__(self, n, partitions, reward='penalty'):
        '''
        n: int, size of tile puzzle
        partitions: list of partitions of 9(tuples of ints)
        '''
        super(TileIrrepEnv, self).__init__(n, one_hot=False, reward=reward) # get grid state
        self.n = n
        self.partitions = partitions
        self.ferrers = [FerrersDiagram(p) for p in partitions]
        self.sn_irreps = [SnIrrep(p) for p in partitions]

        # will only know this from sn_irreps cat irreps!
        self.irrep_shape = self.cat_irreps(TileEnv.solved_grid(n)).shape
        self.observation_space = spaces.Box(low=-float('inf'),
                                            high=float('inf'),
                                            shape=self.irrep_shape)

    def step(self, action):
        grid_state, reward, done, info = super(TileIrrepEnv, self).step(action)
        cat_irrep = self.cat_irreps(grid_state)
        info['grid'] = grid_state
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

    def reset(self, output='grid'):
        grid_state = super(TileIrrepEnv, self).reset()
        if output == 'grid':
            return grid_state
        else:
            return self.cat_irreps(grid_state)

    def all_nbrs(self, grid, x=None, y=None, verbose=False):
        if x  is None:
            n = grid.shape[0]
            empty_loc = np.where(grid == (n * n))
            x, y = empty_loc[0][0], empty_loc[1][0]
        else:
            n = grid.shape[0]
            empty_loc = np.where(grid == (n * n))
            _x, _y = empty_loc[0][0], empty_loc[1][0]
            try:
                assert (_x == x) and (_y == y)
            except:
                pdb.set_trace()

        nbrs = super(TileIrrepEnv, self).neighbors(grid, x, y)
        # call something else instead?
        irrep_nbrs = np.zeros((len(TileIrrepEnv.MOVES),) + self.irrep_shape)
        onehot_nbrs = np.zeros((len(TileIrrepEnv.MOVES),) + self.onehot_shape) # size of onehot?
        self_irrep = None
        self_onehot = grid_to_onehot(grid)

        for move in TileEnv.MOVES:
            if move not in nbrs and self_irrep is None:
                self_irrep = self.cat_irreps(grid)
                irrep_nbrs[move] = self_irrep
            elif move not in nbrs:
                irrep_nbrs[move] = self_irrep
            else:
                irrep_nbrs[move] = self.cat_irreps(nbrs[move])

            if move not in nbrs:
                onehot_nbrs[move] = self_onehot
            else:
                onehot_nbrs[move] = grid_to_onehot(nbrs[move])

        return irrep_nbrs, onehot_nbrs

    def get_all_nbrs(grid):
        nbrs = neighbors(grid)
        irrep_nbrs = []
        for n in nbrs:
            irrep_nbrs.append(self.cat_irreps(n))
        return irrep_nbrs

    def neighbors_dict(self, grid=None, x=None, y=None):
        '''
        Returns neighboring puzzle states' irrep vector
        '''
        nbrs = super(TileIrrepEnv, self).neighbors(grid, x, y)
        irrep_nbrs = {}
        for a, nb_grid in nbrs.items():
            irrep_nbrs[a] = self.cat_irreps(nb_grid)

        return irrep_nbrs

    def peek(self, grid_state, x, y, action):
        new_grid_state, rew, done, info = super(TileIrrepEnv, self).peek(grid_state, x, y, action)
        irrep_state = self.cat_irreps(new_grid_state)
        info = {'grid': new_grid_state}
        return irrep_state, rew, done, info

def test():
    n = 3
    partitions = [(9,), (8, 1)]
    env = TileIrrepEnv(n, partitions)
    irrep_state = env.reset()
    nbrs = env.neighbors_dict()

if __name__ == '__main__':
    test()
