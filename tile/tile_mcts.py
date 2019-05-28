import os
import sys
sys.path.append('../')

import time
import torch
import random
import numpy as np
import collections
import pdb
from utils import get_logger, load_pkl
import argparse
from io_utils import get_prefix
from tile_env import *
from tile_env import neighbors as tile_neighbors

np.set_printoptions(precision=5)
'''
How does the monte carlo tree search interact with the environment and model?
mcts.model.forward: maybe force the model to have something else?
mcts.env.action_space.n
mcts.env.neighbors()
mcts.env.encode_inv(state)
'''
class MCTS(object):
    def __init__(self, root_grid, model, env, coeff=1):
        self.root_grid = root_grid # this is a grid
        self.root_tup = grid_to_tup(root_grid) # this is a grid
        self.model = model
        self.env = env
        self.coeff = coeff
        self.visits = collections.defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.values = collections.defaultdict(lambda: np.zeros(self.env.action_space.n))

        # store the string rep or the encoded rep?
        self.nbr_cache = {} # tree will not grow to be so large so this is fine

    @property
    def nexplored(self):
        return len(self.nbr_cache)

    def neighbors(self, state):
        grid_leaf = tup_to_grid(state)
        if state in self.nbr_cache:
            return self.nbr_cache[state] # tup state?
        else:
            # state needs to be a grid but we also want it to be
            grid_nbrs_dict =  tile_neighbors(grid_leaf)
            nbrs = [grid_to_tup(grid_nbrs_dict[a]) for a in TileEnv.MOVES]
            return nbrs

    def search(self):
        # leaf is a tuple
        leaf, path_states, path_actions = self.search_leaf()
        leaf_nbrs = self.neighbors(leaf) # leaf is a tuple
        self.nbr_cache[leaf] = leaf_nbrs

        # in the autodidactic iter paper the network spits out prob of neighbors
        # we do no such thing, so expand leaves is just evaluates 1 state's value
        value = self.expand_leaves([leaf])
        self.backup_leaf(path_states, path_actions, value)
        is_solved = [TileEnv.is_solved_perm(s) for s in leaf_nbrs]
        if any(is_solved):
            max_actions = [idx for idx, solved in enumerate(is_solved) if solved]
            path_actions.append(random.choice(max_actions))
            return path_actions

        return None

    # how do we know when something is a leaf?
    def search_leaf(self):
        curr = self.root_tup
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
                act = random.randint(0, self.env.action_space.n - 1)
            else:
                u = self.coeff * sqrt_n / (act_cnts + 1)
                q = self.values[curr]
                act = np.argmax(u + q)
                #print('visits:', u)
                #print('vals', q)
                #print('combined', u + q)
                if cnt > 500:
                    pdb.set_trace()

            curr = next_states[act]
            if curr in curr_traj:
                act = random.randint(0, self.env.action_space.n - 1)
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
        # leaves is a list of perm tuples
        # want to convert this to something else
        onehots = np.array([tup_to_onehot(t) for t in leaves])
        onehots_th = torch.from_numpy(onehots).float()
        val = self.model.forward(onehots_th).max().item()
        #val = val.detach().cpu().numpy()
        return val

    def backup_leaf(self, states, actions, value):
        for s, a in zip(states, actions):
            self.values[s][a] = max(value, self.values[s][a])
            self.visits[s][a] += 1

def solve(state, model, env, log, time_limit=None, max_steps=None, coeff=1):
    '''
    state: cube state (currently this is a string)
    model: something that implements forward to compute values
    env: TileEnv 
    log: logger instance
    time_limit: int seconds allowed to run
    max_steps: int number of nodes allowed to explore
    coeff: float, how much to weight exploration
    '''
    tree = MCTS(state, model, env, coeff)
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
    parser.add_argument('--nptrue', action='store_true')
    parser.add_argument('--alpha', type=str)
    parser.add_argument('--parts', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tile_size', type=int, default=3)
    parser.add_argument('--coeff', type=float, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    logpath = None
    model = torch.load('./models/tile3_onehot.pt')

    log = get_logger(logpath)
    log.info('params: {}'.format(args))

    env = TileEnv(args.tile_size, one_hot=False)
    #cubes = [scramble_fixedcore(init_2cube(), n=args.dist) for _ in range(args.ntest)]
    grid_puzzles = [env.reset() for _ in range(args.ntest)]
    time_limit = 100
    log.info('Starting to attempt solves')
    nsolved = 0

    solved = []
    notsolved = []

    for c in grid_puzzles:
        # c is a grid
        res, tree = solve(c, model, env, log, max_steps=args.max_steps, time_limit=args.time_limit, coeff=args.coeff)
        if res is None:
            log.info('Unable to solve: {} | total explored: {}'.format(c, tree.nexplored))
            notsolved.append(c)
        else:
            log.info('Solved: {} | len: {} | total explored: {}'.format(c, len(res), tree.nexplored))
            nsolved += 1
            solved.append(c)

    log.info('Solved: {} / {} | {:.3f}'.format(nsolved, len(grid_puzzles), nsolved / len(grid_puzzles)))
    log.info('unable to solve:')
    for c in notsolved:
        log.info(c)

if __name__ == '__main__':
    np.set_printoptions(precision=2)
    main()
