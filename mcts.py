import os
import sys
sys.path.append('./cube')

import time
import torch
import random
import numpy as np
import collections
from irrep_env import Cube2IrrepEnv
import pdb
from str_cube import *
from utils import get_logger
import argparse

# currently the irrep model returns a tuple of real, imag parts
def get_value(model, state):
    return model.forward(state)[0]

class MCTS(object):
    def __init__(self, root_state, model, env):
        self.root = root_state
        self.model = model
        self.env = env
        self.visits = collections.defaultdict(lambda: np.zeros(self.env.actions))
        self.values = collections.defaultdict(lambda: np.zeros(self.env.actions))

        # store the string rep or the encoded rep?
        self.nbr_cache = {} # tree will not grow to be so large so this is fine

    @property
    def nexplored(self):
        return len(self.nbr_cache)

    def neighbors(self, state):
        if state in self.nbr_cache:
            return self.nbr_cache[state]
        else:
            return self.env.neighbors(state)

    def search(self):
        leaf, path_states, path_actions = self.search_leaf()
        leaf_nbrs = self.neighbors(leaf) # these are strings?
        self.nbr_cache[leaf] = leaf_nbrs

        # in the autodidactic iter paper the network spits out prob of neighbors
        # we do no such thing, so expand leaves is just evaluates 1 state's value
        value = self.expand_leaves([leaf])[0]
        self.backup_leaf(path_states, path_actions, value)

        is_solved = [Cube2IrrepEnv.is_done(s) for s in leaf_nbrs]
        if any(is_solved):
            max_actions = [idx for idx, solved in enumerate(is_solved) if solved]
            path_actions.append(random.choice(max_actions))
            return path_actions

        return None

    # how do we know when something is a leaf?
    def search_leaf(self):
        curr = self.root
        path_states = []
        path_actions = []
        cnt = 0
        curr_traj = set([curr])
        while True:
            # if we have not yet cached this node, then it is a leaf
            next_states = self.nbr_cache.get(curr)
            if next_states is None:
                break

            # num visits to this state
            sqrt_n = np.sqrt(np.sum(self.visits[curr]))
            act_cnts = self.visits[curr]
            if sqrt_n < 1e-5: # no visits
                act = random.randint(0, self.env.actions - 1)
            else:
                u = sqrt_n / (act_cnts + 1)
                q = self.values[curr]
                act = np.argmax(u + q)
                if cnt > 500:
                    print('{} | u+q: {} | {} | {} | exp: {}'.format(curr, u + q, q, self.visits[curr], self.nexplored))
                    pdb.set_trace()

            curr = next_states[act]
            if curr in curr_traj:
                act = random.randint(0, self.env.actions - 1)
                curr = next_states[act]

            curr_traj.add(curr)
            path_actions.append(act)
            path_states.append(curr)
            cnt += 1
        return curr, path_states, path_actions

    def expand_leaves(self, leaves):
        '''
        leaves: string states
        Returns the values of the leaves
        '''
        # now that we're at a leaf node, try to expand it. evaluate the model on each node
        r_th, i_th = self.env.encode_inv(leaves)
        rv, _ = self.model.forward(r_th, i_th)
        rv = rv.detach().cpu().numpy()
        return rv

    def backup_leaf(self, states, actions, value):
        for s, a in zip(states, actions):
            self.values[s][a] = max(value, self.values[s][a])
            self.visits[s][a] += 1

def solve(state, model, env, log, time_limit=None, max_steps=None):
    tree = MCTS(state, model, env)
    # number of steps == max
    nsteps = 0
    start = time.time()
    while True:
        sol_path = tree.search()
        if sol_path:
            # we've found a solution!
            # we can postprocess and bfs
            return sol_path, tree
        nsteps += 1

        if max_steps and nsteps > max_steps:
            #log.info('Max steps exceeded. Stopping now after {} iters'.format(nsteps))
            break

        if time_limit and (time.time() - start) > time_limit:
            #log.info('Max time exceeded. Stopping now after {} iters'.format(nsteps))
            break

    return None, tree

def main():
    _dir = '/local/hopan/cube/logs/dist422long_9/'
    #_dir = '/local/hopan/cube/logs/dist233long_curric_2'
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=_dir)
    parser.add_argument('--dist', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--time_limit', type=int, default=600)
    parser.add_argument('--ntest', type=int, default=100)
    args = parser.parse_args()

    model = os.path.join(args.path, 'model.pt')
    logpath = os.path.join(args.path, 'mcts.txt')
    log = get_logger(logpath)

    log.info('params: {}'.format(args))
    model = torch.load(model)
    log.info('Done loading model')

    if model.wr.shape == torch.Size([560 * 560, 1]):
        alpha = (2, 3, 3)
        parts = ((2,), (1, 1, 1), (1, 1, 1))
    else:
        alpha = (4, 2, 2)
        parts = ((4,), (1, 1), (1, 1))

    log.info('Loaded model using irrep: {} | {}'.format(alpha, parts))
    env = Cube2IrrepEnv(alpha, parts)
    log.info('Loaded env')
    cubes = [scramble_fixedcore(init_2cube(), n=args.dist) for _ in range(args.ntest)]
    time_limit = 600
    log.info('Starting to attempt solves')
    nsolved = 0

    solved = []
    notsolved = []

    for c in cubes:
        res, tree = solve(c, model, env, log, max_steps=args.max_steps, time_limit=args.time_limit)

        if res is None:
            log.info('Unable to solve: {} | total explored: {}'.format(c, tree.nexplored))
            notsolved.append(c)
        else:
            log.info('Solved: {} | len: {} | total explored: {}'.format(c, len(res), tree.nexplored))
            nsolved += 1
            solved.append(c)

    log.info('Solved: {} / {} | {:.3f}'.format(nsolved, len(cubes), nsolved / len(cubes)))
    log.info('unable to solve:')
    for c in notsolved:
        log.info(c)

if __name__ == '__main__':
    np.set_printoptions(precision=2)
    main()
