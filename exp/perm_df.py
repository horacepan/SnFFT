import pdb
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import torch
from utility import S8_GENERATORS, px_mult
from wreath_puzzle import PYRAMINX_GENERATORS, px_wreath_mul, CUBE2_GENERATORS

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
        for idx, row in self.df[self.df['dist'] == 0].iterrows():
            self._done_states.append(self._get_state(row))
        self._done_states_set = set(self._done_states)

        self._generators = []
        d1_elements = self.df[self.df['dist'] == 1]
        for idx, row in d1_elements.iterrows():
            self._generators.append(self._get_state(row))
            if len(self._generators) == ngenerators:
                break
        self._all_states = list(self.dist_dict.keys())

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

    def random_state(self, scramble_len):
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
        return {self._get_state(row): row['dist'] for s, row in self.df.iterrows()}

    def distance(self, state):
        return self.dist_dict[state]

    def benchmark(self):
        states = list(self.dist_dict.keys())

        dist_probs = {}
        probs = []
        for p in states:
            dist = self.dist_dict[p]
            true_nbr_vals = {n: self.distance(n) for n in self.nbrs(p)}
            opt_val = min(true_nbr_vals.values())
            opt_nbrs = [n for n, dist in true_nbr_vals.items() if dist == opt_val]
            prob_opt_step = len(opt_nbrs) / len(true_nbr_vals)
            probs.append(prob_opt_step)

            if dist not in dist_probs:
                dist_probs[dist] = []

            dist_probs[dist].append(prob_opt_step)

        res_prob = {}
        for i in range(1, self.max_dist+1):
            res_prob[i] = np.mean(dist_probs[i])

        return np.mean(probs), res_prob

    def opt_nbr(self, state, policy, to_tensor):
        '''
        See if the optimal move given by the policy coicides with the trust
        dist's optimal move
        '''
        true_nbr_vals = {n: self.distance(n) for n in self.nbr_values(state, policy, to_tensor)}
        opt_val = min(true_nbr_vals.values())
        opt_nbrs = [n for n, dist in true_nbr_vals.items() if dist == opt_val]

        pol_vals = self.nbr_values(state, policy, to_tensor)
        opt_pol_nbr = max(pol_vals, key=pol_vals.get)
        return opt_pol_nbr in opt_nbrs

    def nbr_values(self, state, policy, to_tensor):
        vals = {}
        state_nbrs = self.nbrs(state)

        if hasattr(policy, 'nout') and policy.nout > 1:
            tens = to_tensor([state])
            res = policy.forward(tens)
            for i, ntup in enumerate(state_nbrs):
                vals[ntup] = res[0, i].item()
            return vals
        elif hasattr(policy, 'nout') and policy.nout == 1:
            for ntup in state_nbrs:
                tens = to_tensor([ntup])
                vals[ntup] = policy.forward(tens).item()
            return vals
        else:
            raise Exception('Cant compute nbr vals')
            return vals

    def benchmark_policy(self, states, policy, to_tensor):
        if len(states) == 0:
            return -1

        with torch.no_grad():
            ncorrect = 0
            for g in states:
                ncorrect += int(self.opt_nbr(g, policy, to_tensor))
            return ncorrect / len(states)

    def prop_corr_sample(self, policy, to_tensor, cnt, scramble_len=1000):
        states = [self.random_state(scramble_len + (1 if random.random() < 0 else 0))
                  for _ in range(cnt)]
        dist_corr = {}
        dist_cnts = {}
        ncorrect = 0

        for i in (range(cnt)):
            state = states[i]
            dist = self.dist_dict[state]
            correct = int(self.opt_nbr(state, policy, to_tensor))
            ncorrect += correct
            dist_corr[dist] = dist_corr.get(dist, 0) + correct
            dist_cnts[dist] = dist_cnts.get(dist, 0) + 1

        for i in range(self.max_dist + 1):
            if i not in dist_cnts or i not in dist_corr:
                continue
            dist_corr[i] = dist_corr[i] / dist_cnts[i]
        prop_corr = ncorrect / len(states)
        return prop_corr, dist_corr


    def prop_corr_by_dist(self, policy, to_tensor, distance_check=None, cnt=0):
        dist_corr = {}
        dist_cnts = {}
        ncorrect = 0

        if len(self.dist_dict) > 50000 and distance_check:
            states = []
            for d in distance_check:
                sample_df = self.df[self.df['dist'] == d]
                states.extend(self.random_states(d, cnt))
        else:
            states = self.dist_dict.keys()

        for state in states:
            dist = self.dist_dict[state]
            correct = int(self.opt_nbr(state, policy, to_tensor))
            ncorrect += correct
            dist_corr[dist] = dist_corr.get(dist, 0) + correct
            dist_cnts[dist] = dist_cnts.get(dist, 0) + 1

        for i in range(self.max_dist + 1):
            if i not in dist_cnts or i not in dist_corr:
                continue
            dist_corr[i] = dist_corr[i] / dist_cnts[i]
        prop_corr = ncorrect / len(states)
        return prop_corr, dist_corr

    def opt_move_tup(self, tup):
        dists = [self.dist_dict[t] for t in tup]
        return dists.index(min(dists))

    def random_states(self, dist, cnt):
        subdf = self.df[self.df['dist'] == dist]
        if len(subdf) > cnt:
            subdf = subdf.sample(n=cnt)
        perms = [self._get_state(row) for _, row in subdf.iterrows()]
        return perms

    def all_states(self):
        return self._all_states

    def _get_state(self, df_row):
        return str2tup(df_row['state'])

class WreathDF(PermDF):
    def __init__(self, fname, ngenerators, cyc_size, dist_dict_pkl=None):
        '''
        Assumption: The first {ngenerator} states of dist 1 from solved state are generators of the puzzle
        '''
        self.df = self.load_df(fname)
        self.dist_dict = self.load_dist_dict(dist_dict_pkl)
        self.max_dist = self.df['dist'].max()
        #self.generators = PYRAMINX_GENERATORS
        self._num_nbrs = ngenerators
        self.cyc_size = cyc_size

        self._done_states = []
        for idx, row in self.df[self.df['dist'] == 0].iterrows():
            self._done_states.append(self._get_state(row))
        self._done_states_set = set(self._done_states)

        self.generators = []
        d1_elements = self.df[self.df['dist'] == 1]
        for idx, row in d1_elements[:ngenerators].iterrows():
            self.generators.append(self._get_state(row))
            if len(self.generators) == ngenerators:
                break
        self._all_states = list(self.dist_dict.keys())

    def load_dist_dict(self, dist_dict_pkl=None):
        if dist_dict_pkl:
            return pickle.load(open(dist_dict_pkl, 'rb'))
        return {self._get_state(row): row['dist'] for s, row in tqdm(self.df.iterrows())}

    def load_df(self, fname):
        df = pd.read_csv(fname, header=None, dtype={0: str, 1: str, 2: int})
        df[0] = df[0].apply(str2tup)
        df[1] = df[1].apply(str2tup)
        df.columns = ['otup', 'ptup', 'dist']
        return df

    def nbrs(self, state):
        ot, pt = state
        return [px_wreath_mul(o, p, ot, pt, self.cyc_size) for (o, p) in self.generators]

    def step(self, state, action_idx):
        g = self.generators[action_idx]
        ot, pt = state
        return px_wreath_mul(g[0], g[1], ot, pt, self.cyc_size)

    def _get_state(self, df_row):
        return (df_row['otup'], df_row['ptup'])

'''
def get_group_df(group_name, df_name):
    if group_name == 's8_sym':
        return PermDF()
    elif group_name == 's8_onestart':
        return PermDF()
    elif group_name == 'pyraminx':
        return WreathDF(fname)
    elif group_name == 'cube2'
        return WreathDF(fname, )
'''

def test():
    #fname = '/home/hopan/github/idastar/s8_dists_red.txt'
    fname = '/local/hopan/pyraminx/dists.txt'
    eye = (1, 2, 3, 4, 5, 6, 7, 8)
    pdf = WreathDF(fname, 8, 2)
    to_tensor = None
    pdb.set_trace()

if __name__ == '__main__':
    test()
