import pdb
import time
import sys
sys.path.append('../')
sys.path.append('../cube')
import numpy as np
import pandas as pd
from cube_irrep import Cube2Irrep
from utils import check_memory
from wreath_puzzle import PYRAMINX_GENERATORS, px_wreath_mul
MOVES = [

]

def px_mul(p1, p2):
    return tuple([p1[p2[i] - 1] for i in range(len(p1))])

def pinv(p):
    return tuple(p.index(i) + 1 for i in range(1, len(p) + 1))

def pinv_dot(p, otup):
    pinv = pinv(p)
    ot = tuple(otup[pinv[i] - 1] for i in range(len(otup)))
    return ot

def w_mul(w1, w2):
    o1, p1 = w1
    o2, p2 = w2
    o3 = o1 + pinv_dot(p1, o2)
    p3 = px_mul(p1, p2)
    return o3, p3

def str2wtup(ostr, pstr):
    otup = tuple(int(i) for i in ostr)
    ptup = tuple(int(i) for i in pstr)
    return otup, ptup

def w_nbrs(g):
    otups, ptups = zip(*PYRAMINX_GENERATORS)
    return [px_wreath_mul(o, p, g[0], g[1], 2) for o, p in zip(otups, ptups)]

class WreathDF:
    def __init__(self, fname):
        self.df = self.load_df(fname)
        self.dist_dict = self.load_dist_dict()
        self.nbr_func = w_nbrs
        self.max_dist = self.df['dist'].max()

    def load_df(self, fname):
        st = time.time()
        df = pd.read_csv(fname, header=None, dtype={0: str, 1:str, 2: int})
        df.columns = ['otup', 'ptup', 'dist']
        return df

    def __getitem__(self, gtup):
        return self.dist_dict[gtup]

    def __call__(self, gtup):
        return self.dist_dict[gtup]

    def benchmark(self, gtups=None):
        if gtups is None:
            gtups =  list(self.dist_dict.keys())

        dist_probs = {}
        probs = []
        for p in gtups:
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

    def load_dist_dict(self):
        return {str2wtup(row[0], row[1]): row['dist'] for s, row in self.df.iterrows()}

    def nbr_values(self, gtup, func=None):
        if func is None:
            func = self.__call__

        vals = {}
        gtup_nbrs = self.nbr_func(gtup)

        if hasattr(func, 'forward_tup') and hasattr(func, 'nout') and func.nout > 1:
            res = func.forward_tup([gtup])
            for i, ntup in enumerate(gtup_nbrs):
                vals[ntup] = res[0, i].item()
            return vals

        for ntup in gtup_nbrs:
            if hasattr(func, 'forward_tup'):
                vals[ntup] = func.forward_tup([ntup])
            else:
                vals[ntup] = func(ntup)

        return vals

    def benchmark_policy(self, gtups, policy):
        if len(gtups) == 0:
            return -1

        with torch.no_grad():
            ncorrect = 0
            for g in gtups:
                ncorrect += int(self.opt_nbr(g, policy))
            return ncorrect / len(gtups)

    def opt_nbr(self, gtup, policy):
        true_vals = self.nbr_values(gtup)
        opt_val = min(true_vals.values())
        opt_nbrs = [n for n, dist in true_vals.items() if dist == opt_val]

        pol_vals = self.nbr_values(gtup, policy)
        opt_pol_nbr = max(pol_vals, key=pol_vals.get)
        return opt_pol_nbr in opt_nbrs

    def prop_corr_by_dist(self, policy):
        dist_corr = {}
        dist_cnts = {}
        ncorrect = 0
        for ptup, dist in self.dist_dict.items():
            correct = int(self.opt_nbr(ptup, policy))
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

    def random_state(self, dist, cnt):
        subdf = self.df[self.df['dist'] == dist]
        if len(subdf) > cnt:
            subdf = subdf.sample(n=cnt)
        tups = [str2wtup(row['otup'], row['ptup']) for _, row in subdf.iterrows()]
        return tups

    def forward_tup(self, gtup):
        return [self.dist_dict[g] for g in gtup]

    def all_states(self):
        return self.dist_dict.keys()

def main():
    alpha = (4, 2)
    parts = ((3, 1), (1, 1))
    fname = '/local/hopan/pyraminx/dists.txt'
    wdf = WreathDF(fname)
    tup = ((1, 0, 1, 0, 0, 0), (5, 2, 1, 4, 3, 6))
    print(wdf.nbr_values(tup, wdf))
    print(wdf.random_state(1, 3))
    print(wdf.opt_nbr(tup, wdf))

if __name__ == '__main__':
    main()
