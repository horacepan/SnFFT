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
from cube_env import CubeEnv
from utils import get_logger, load_pkl
import argparse
from io_utils import get_prefix

np.set_printoptions(precision=5)
'''
How does the monte carlo tree search interact with the environment and model?
mcts.model.forward: maybe force the model to have something else?
mcts.env.action_space.n
mcts.env.neighbors()
mcts.env.encode_inv(state)
'''
def model_path(alpha, parts):
    return os.path.join(get_prefix(), 'cube', 'fourier_eval_sym_all', str(alpha), str(parts) + '.npy')

class NPModel:
    def __init__(self, alpha, parts):
        self.alpha = alpha
        self.parts = parts
        self.values = -np.load(model_path(alpha, parts))
        self.idx_to_dist = load_pkl(os.path.join(get_prefix(), 'cube', 'cube2_pkls', 'idx_to_dist.pkl'))
        self.correct = np.load(os.path.join(get_prefix(), 'cube', 'fourier_eval_results', str(alpha), str(parts), 'correct.npy'))
        self.cube_to_idx = load_pkl(os.path.join(get_prefix(), 'cube', 'cube2_pkls', 'cube_to_idx.pkl'))

    def forward(self, cube_state):
        # convert cube state  -> idx -> index into self.values
        idx = self.cube_to_idx[cube_state]
        return self.values[idx].real

    def is_correct(self, state):
        cidx = self.cube_to_idx[state]
        return self.correct[cidx]

    def get_dist(self, state):
        return self.idx_to_dist[self.cube_to_idx[state]]

    def opt_nbr(self, state):
        '''
        Returns the neighbor with the max value. Recall that the true values have been loaded as negatives.
        '''
        nbrs = neighbors_fixed_core_small(state)
        nvals = [self.values[self.cube_to_idx[n]].real for n in nbrs]
        opt_idx = np.argmax(nvals)
        return nbrs[opt_idx]

    def dfs(self, state, max_steps=100):
        step = 0
        curr = state
        path = [curr]
        prev = None
        while not CubeEnv.is_done(curr) and step < max_steps:
            #print(step, curr, prev)
            prev = curr
            curr = self.opt_nbr(curr)
            path.append(curr)
            step += 1
            if prev == curr:
                pdb.set_trace()

        return CubeEnv.is_done(curr), step, path

class MCTS(object):
    def __init__(self, root_state, model, env, coeff=1):
        self.root = root_state
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
        if state in self.nbr_cache:
            return self.nbr_cache[state]
        else:
            return CubeEnv.neighbors(state)

    def search(self):
        leaf, path_states, path_actions = self.search_leaf()
        leaf_nbrs = self.neighbors(leaf) # these are strings?
        self.nbr_cache[leaf] = leaf_nbrs

        # in the autodidactic iter paper the network spits out prob of neighbors
        # we do no such thing, so expand leaves is just evaluates 1 state's value
        if isinstance(self.model, NPModel):
            value = self.expand_leaves_np([leaf])[0]
        else:
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
        r_th, i_th = self.env.encode_inv(leaves)
        rv, _ = self.model.forward(r_th, i_th)
        rv = rv.detach().cpu().numpy()
        return rv

    def expand_leaves_np(self, leaves):
        leaf_vals = [self.model.forward(l) for l in leaves]
        return leaf_vals

    def backup_leaf(self, states, actions, value):
        for s, a in zip(states, actions):
            self.values[s][a] = max(value, self.values[s][a])
            self.visits[s][a] += 1

def solve(state, model, env, log, time_limit=None, max_steps=None, coeff=1):
    '''
    state: cube state (currently this is a string)
    model: something that implements forward to compute values
    env: CubeEnv
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
    parser.add_argument('--coeff', type=float, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    if args.nptrue:
        args.alpha = eval(args.alpha)
        args.parts = eval(args.parts)
        logpath = None
        log = get_logger(logpath)
        log.info('Making NP model!')
        model = NPModel(args.alpha, args.parts)
    else:
        logpath = os.path.join(args.path, 'mcts.txt')
        model = os.path.join(args.path, 'model.pt')
        model = torch.load(model)

        log = get_logger(logpath)
        if model.wr.shape == torch.Size([560 * 560, 1]):
            alpha = (2, 3, 3)
            parts = ((2,), (1, 1, 1), (1, 1, 1))
        else:
            alpha = (4, 2, 2)
            parts = ((4,), (1, 1), (1, 1))

    log.info('params: {}'.format(args))

    if args.nptrue:
        env = CubeEnv(2, fixedcore=True)
    else:
        env = Cube2IrrepEnv(alpha, parts, fixedcore=True)
        log.info('Loaded env')

    cubes = [scramble_fixedcore(init_2cube(), n=args.dist) for _ in range(args.ntest)]
    time_limit = 600
    log.info('Starting to attempt solves')
    nsolved = 0

    solved = []
    notsolved = []

    for c in cubes:
        res, tree = solve(c, model, env, log, max_steps=args.max_steps, time_limit=args.time_limit, coeff=args.coeff)
        pdb.set_trace()
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
