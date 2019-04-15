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
    # This is really the fixed core function map
    FIXEDCORE_FUNCTION_MAP = {
        'u': str_cube.rot_u,
        'd': str_cube.rot_d2,
        'l': str_cube.rot_l2,
        'r': str_cube.rot_r,
        'f': str_cube.rot_f,
        'b': str_cube.rot_ib2,
    }
    # standard rotations
    FUNCTION_MAP = {
        'u': str_cube.rot_u,
        'd': str_cube.rot_d,
        'l': str_cube.rot_l,
        'r': str_cube.rot_r,
        'f': str_cube.rot_f,
        'b': str_cube.rot_ib,
        'iu': str_cube.rot_iu,
        'id': str_cube.rot_id,
        'il': str_cube.rot_il,
        'ir': str_cube.rot_ir,
        'if': str_cube.rot_if,
        'ib': str_cube.rot_ib,
    }

    @property
    def actions(self):
        return self.action_space.n

    def next_state(self, state, action):
        face = CubeEnv.ACTION_MAP[action]
        if self.fixedcore:
            rot_func = CubeEnv.FIXEDCORE_FUNCTION_MAP[face]
        else:
            rot_func = CubeEnv.FUNCTION_MAP[face]
        return rot_func(state)

    def __init__(self, size, reward_mode='penalty', fixedcore=True, solve_rew=1):
        self.size = size
        self.fixedcore = fixedcore
        self.solve_rew = solve_rew
        if fixedcore:
            self.action_space = spaces.Discrete(6)
        else:
            self.action_space = spaces.Discrete(12)
        if reward_mode == 'binary':
            self.reward_func = self.binary_reward
        elif reward_mode == 'penalty':
            self.reward_func = self.penalty_reward
        self.seed()

        if size == 2:
            self.state = self.reset2()

    def step(self, action):
        if action in CubeEnv.ACTION_MAP:
            face = CubeEnv.ACTION_MAP[action]
            if self.fixedcore:
                rot_func = CubeEnv.FIXEDCORE_FUNCTION_MAP[face]
            else:
                rot_func = CubeENv.FUNCTION_MAP[face]
            self.state = rot_func(self.state)
        else:
            raise ValueError('Action {} is invalid'.format(action))

        done = is_done(self.state)
        reward = self.reward_func(done)
        return self.state, reward, done, {}

    @staticmethod
    def is_done(state):
        return (state in SOLVED_STATES)

    def reset(self, max_dist=100):
        if self.size == 2:
            self.state = CubeEnv.reset2(max_dist)
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
        return self.solve_rew if solved else 0

    def penalty_reward(self, solved):
        return self.solve_rew if solved else -1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def reset2(self, max_dist=100):
        '''
        Resets the state to a random 2-cube configuration
        '''
        c = str_cube.init_2cube()
        self.state = str_cube.scramble(c, max_dist)
        return self.state

    def reset_fixed(self, max_dist=100):
        '''
        Resets the cube state using the fixed core moves for the 2-cube.
        '''
        c = str_cube.init_2cube() 
        self.state = str_cube.scramble_fixedcore(c, max_dist)
        return self.state

    def soft_reset(self):
        self.state = str_cube.init_2cube()

    def neighbors(self, state):
        return str_cube.neighbors_fixed_core(state)[:6]

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
