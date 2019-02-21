import pdb
from gym import spaces
from gym.utils import seeding
import gym
import numpy as np
import str_cube
import cube_perms

SOLVED_STATES = set(cube_perms.rot_permutations(str_cube.init_2cube()))
def is_done(cube_state):
    return (cube_state in SOLVED_STATES)

class CubeEnv(gym.Env):
    ACTION_MAP = {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r',
        4: 'f',
        5: 'b',
    }
    def __init__(self, size, reward_mode='penalty'):
        self.size = size
        self.action_space = spaces.Discrete(6)
        if reward_mode == 'binary':
            self.reward_func = self.binary_reward
        elif reward_mode == 'penalty':
            self.reward_func = self.penalty_reward
        self.seed()

        if size == 2:
            self.state = CubeEnv.reset2()

    def step(self, action):
        if action in CubeEnv.ACTION_MAP:
            face = CubeEnv.ACTION_MAP[action]
            self.state = str_cube.rotate(self.state, face)
        else:
            pass
            #raise ValueError('Action {} is invalid'.format(action))

        done = is_done(self.state)
        reward = self.reward_func(done)
        return self.state, reward, done, {}

    @staticmethod
    def is_done(self):
        return (self.state in SOLVED_STATES)

    def reset(self):
        if self.size == 2:
            self.state = CubeEnv.reset2()
            return self.state
        else:
            raise NotImplementedError('Havent implemented other sizes yet')

    def render(self, mode='console'):
        if mode is 'ascii':
            # text output
            print(self.state)
        elif mode is 'console':
            str_cube.render(self.state)
        else:
            raise ValueError('Invalid render mode for a cube!')

    def binary_reward(self, solved):
        return 1 if solved else 0

    def penalty_reward(self, solved):
        return 1 if solved else -1

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    @staticmethod
    def reset2():
        c = str_cube.init_2cube() 
        return str_cube.scramble(c, 1000)

    def soft_reset(self):
        self.state = str_cube.init_2cube()

def test():
    env = CubeEnv(2)
    env.reset()
    env.render()
    print('----')
    for i in range(6):
        print('===========')
        _, rew, d, _ = env.step(i)
        print('Rotate face: {} | Reward: {} | Done: {}'.format(
            CubeEnv.ACTION_MAP[i], rew, d)
        )
        env.render()

if __name__ == '__main__':
    test()
