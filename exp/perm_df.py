import pdb
import random
import numpy as np
import pandas as pd
import torch
from utility import S8_GENERATORS, px_mult

def str2tup(s):
    return tuple(int(i) for i in s)

class PermDF:
    '''
    Container class for holding mapping from perm tuple -> distance
    Use it to evaluate policies (mapping from perm tuple -> number).
    '''
    #def __init__(self, fname, ident):
    def __init__(self, fname, ngenerators):
        '''
        Assumption: The first {ngenerator} states of dist 1 from solved state are generators of the puzzle
        '''
        self.df = self.load_df(fname)
        self.dist_dict = self.load_dist_dict()
        self.max_dist = self.df['dist'].max()
        self.generators = S8_GENERATORS
        self._num_nbrs = ngenerators

        self._done_states = []
        self._done_states_set = set(self._done_states)
        for idx, row in self.df[self.df['dist'] == 0].iterrows():
            self._done_states.append(str2tup(row['state']))

        self.generators = []
        d1_elements = self.df[self.df['dist'] == 1]['state']
        for s in d1_elements[:ngenerators]:
            self.generators.append(str2tup(s))

    def is_done(self, state):
        return state in self._done_states_set

    @property
    def num_nbrs(self):
        return self._num_nbrs

    def nbrs(self, state):
        return [px_mult(g, state) for g in self.generators]

    def step(self, state, action_idx):
        g = self.generators[action_idx]
        return px_mult(g, state)

    def random_walk(self, length):
        states = []
        state = random.choice(self._done_states)
        states.append(state)
        for _ in range(length):
            action = random.randint(0, len(self.generators) - 1)
            state = self.step(state, action)
            states.append(state)
        return states

    def random_element(self, scramble_len):
        state = random.choice(self._done_states)
        for _ in range(scramble_len):
            action = random.randint(0, len(self.generators) - 1)
            state = self.step(state, action)
        return state

    def load_df(self, fname):
        df = pd.read_csv(fname, header=None, dtype={0: str, 1: int})
        df.columns = ['state', 'dist']
        return df

    def load_dist_dict(self):
        return {str2tup(row['state']): row['dist'] for s, row in self.df.iterrows()}

    def __getitem__(self, state):
        return self.dist_dict[state]

    def __call__(self, state):
        return self.dist_dict[state]

    def distance(self, state):
        return self.dist_dict[state]

    def benchmark(self):
        states = list(self.dist_dict.keys())

        dist_probs = {}
        probs = []
        for p in states:
            dist = self.dist_dict[p]
            true_vals = self.nbr_values(p)
            opt_val = min(true_vals.values())
            opt_nbrs = [n for n, dist in true_vals.items() if dist == opt_val]
            prob_opt_step = len(opt_nbrs) / len(true_vals)
            probs.append(prob_opt_step)

            if dist not in dist_probs:
                dist_probs[dist] = []

            dist_probs[dist].append(prob_opt_step)

        res_prob = {}
        for i in range(1, self.max_dist+1):
            res_prob[i] = np.mean(dist_probs[i])

        return np.mean(probs), res_prob

    def opt_nbr(self, state, policy):
        '''
        See if the optimal move given by the policy coicides with the trust
        dist's optimal move
        '''
        true_vals = self.nbr_values(state)
        opt_val = min(true_vals.values())
        opt_nbrs = [n for n, dist in true_vals.items() if dist == opt_val]

        pol_vals = self.nbr_values(state, policy)

        if hasattr(policy, 'optmin') and policy.optmin:
            opt_pol_nbr = min(pol_vals, key=pol_vals.get)
        else:
            opt_pol_nbr = max(pol_vals, key=pol_vals.get)
        return opt_pol_nbr in opt_nbrs

    def nbr_values(self, state, func=None):
        if func is None:
            func = self.__call__

        vals = {}
        state_nbrs = self.nbrs(state)

        if hasattr(func, 'forward_tup') and hasattr(func, 'nout') and func.nout > 1:
            res = func.forward_tup([state])
            for i, ntup in enumerate(state_nbrs):
                vals[ntup] = res[0, i].item()
            return vals

        for ntup in state_nbrs:
            if hasattr(func, 'forward_tup'):
                vals[ntup] = func.forward_tup([ntup]).item()
            else:
                vals[ntup] = func(ntup)

        return vals

    def benchmark_policy(self, states, policy):
        if len(states) == 0:
            return -1

        with torch.no_grad():
            ncorrect = 0
            for g in states:
                ncorrect += int(self.opt_nbr(g, policy))
            return ncorrect / len(states)

    def prop_corr_by_dist(self, policy):
        dist_corr = {}
        dist_cnts = {}
        ncorrect = 0
        for state, dist in self.dist_dict.items():
            correct = int(self.opt_nbr(state, policy))
            ncorrect += correct
            dist_corr[dist] = dist_corr.get(dist, 0) + correct
            dist_cnts[dist] = dist_cnts.get(dist, 0) + 1

        for i in range(self.max_dist + 1):
            dist_corr[i] = dist_corr[i] / dist_cnts[i]
        prop_corr = ncorrect / len(self.dist_dict)
        return prop_corr, dist_corr

    def opt_move_tup(self, tup):
        dists = [self.dist_dict[t] for t in tup]
        return dists.index(min(dists))

    def random_states(self, dist, cnt):
        subdf = self.df[self.df['dist'] == dist]
        if len(subdf) > cnt:
            subdf = subdf.sample(n=cnt)
        perms = [str2tup(row['state']) for _, row in subdf.iterrows()]
        return perms

    def forward_tup(self, state):
        return self.dist_dict[state]

    def all_states(self):
        return self.dist_dict.keys()

def test():
    fname = '/home/hopan/github/idastar/s8_dists_red.txt'
    eye = (1, 2, 3, 4, 5, 6, 7, 8)
    pdf = PermDF(fname, 6)
    policy = lambda g: g.index(8)
    print(pdf.opt_nbr(eye, policy))

if __name__ == '__main__':
    test()
