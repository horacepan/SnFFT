import pdb
import pickle
import random
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

GENERATORS = [
    (4, 2, 3, 5, 8, 6, 7, 1),
    (8, 2, 3, 1, 4, 6, 7, 5),
    (2, 7, 3, 4, 5, 6, 8, 1),
    (8, 1, 3, 4, 5, 6, 2, 7),
    (2, 3, 4, 1, 5, 6, 7, 8),
    (4, 1, 2, 3, 5, 6, 7, 8),
]

DONE_STATE = (1, 2, 3, 4, 5, 6, 7, 8)

def px_mult(p1, p2):
    return tuple(
        p1[p2[i] - 1] for i in range(len(p1))
    )

class S8CubeEnv(gym.Env):
    def __init__(self,
                 nvecs,
                 evec_loc='/local/hopan/s8cube/lap_eigvecs.npy',
                 idx_pkl_loc='/local/hopan/s8cube/idx_dict.pkl'):
        self.eigvecs = np.load(evec_loc)
        self.idx_dict = pickle.load(open(idx_pkl_loc, 'rb'))
        self.nvecs = nvecs
        self.action_space = spaces.Discrete(6)

        self.tup_state = (1, 2, 3, 4, 5, 6, 7, 8)

    def eigvec(self, ptup):
        idx = self.idx_dict[ptup]
        return self.eigvecs[idx, :self.nvecs]

    @property
    def actions(self):
        return self.action_space.n

    @property
    def state(self):
        idx = self.idx_dict[self.tup_state]
        return self.eigvecs[idx, :self.nvecs]

    def neighbors(self, ptup):
        return [px_mult(g, ptup) for g in GENERATORS]

    # TODO: cache to speed up
    def neighbors_vec(self, ptup):
        nbrs = self.neighbors(ptup)
        nbrs_idx = [self.idx_dict[n] for n in nbrs]
        return self.eigvecs[nbrs_idx, :self.nvecs]

    def step(self, action):
        move = GENERATORS[action]
        self.tup_state = px_mult(move, self.tup_state)
        done = (self.tup_state == DONE_STATE)
        reward = 1 if done else 0
        return self.eigvec(self.tup_state), reward, done, {}

    # TODO: better reset?
    def reset(self):
        actions = [0, 1, 2, 3, 4, 5]
        for _ in range(1000):
            self.step(random.choice(actions))
        return self.state

def test():
    env = S8CubeEnv(10)
    state = env.reset()
    print(state.shape)

    for i in range(env.actions):
        env.step(i)
        print(env.state, env.tup_state)

if __name__ == '__main__':
    test()
