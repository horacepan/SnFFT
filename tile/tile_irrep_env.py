import pdb
import sys
sys.path.append('../')
from young_tableau import FerrersDiagram
from yor import yor
from perm2 import Perm2, sn
from tile_env import TileEnv, neighbors, env_neighbors

class SnIrrep:
    def __init__(self, partition):
        self.partition = partition
        self.ferrers = FerrersDiagram(partition)

    def grid_irrep(self, grid):
        ptup = tuple(i for row in grid for i in row)
        return self.irrep(ptup)

    def irrep(self, tup):
        return yor(self.ferrers, Perm2.from_tup(tup))

    def perm_irrep(self, perm):
        return yor(self.ferrers, perm)

class TileIrrepEnv(TileEnv):
    def __init__(self, n, partition, one_hot=False):
        super(TileIrrepEnv, self).__init__(n, one_hot)
        self.partition = partition
        self.ferrers = FerrersDiagram(partition)
        self.sn_irrep = SnIrrep(partition)

    @staticmethod
    def get_yor(self, grid):
        ptup = tuple(i for row in grid for i in row)
        return yor(self.ferrers, Perm2.from_tup(ptup).inv())

    def step(self, action):
        grid_state, reward, done, info = super(TileIrrepEnv, self).step(action)
        return self.sn_irrep.grid_irrep(grid_state), reward, done, info

    def reset(self):
        grid_state = super(TileIrrepEnv, self).reset()
        return self.sn_irrep.grid_irrep(grid_state)

    def neighbors(self):
        nbrs = super(TileIrrepEnv, self).neighbors()
        irrep_nbrs = {}
        for a, nb in nbrs.items():
            irrep_nbrs[a] = self.sn_irrep.grid_irrep(nb)

        return irrep_nbrs

def test():
    n = 3
    partition = (8, 1)
    env = TileIrrepEnv(n, partition)
    irrep_state = env.reset()
    nbrs = env.neighbors()

if __name__ == '__main__':
    test()
