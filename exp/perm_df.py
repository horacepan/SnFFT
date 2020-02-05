import pdb
import numpy as np
import pandas as pd
import torch
from utility import S8_GENERATORS

def str2ptup(s):
    return tuple(int(i) for i in s)

class PermDF:
    '''
    Container class for holding mapping from perm tuple -> distance
    Use it to evaluate policies (mapping from perm tuple -> number).
    '''
    def __init__(self, fname, nbr_func):
        self.df = self.load_df(fname)
        self.dist_dict = self.load_dist_dict()
        self.nbr_func = nbr_func

    def load_df(self, fname):
        df = pd.read_csv(fname, header=None, dtype={0: str, 1: int})
        df = df.set_index(0)
        df.columns = ['dist']
        return df

    def load_dist_dict(self):
        return {str2ptup(s): row['dist'] for s, row in self.df.iterrows()}

    def __getitem__(self, gtup):
        return self.dist_dict[gtup]

    def benchmark(self, gtups):
        if len(gtups) == 0:
            return -1

        probs = []
        for p in gtups:
            true_vals = self.nbr_values(p)
            opt_val = min(true_vals.values())
            opt_nbrs = [n for n, dist in true_vals.items() if dist == opt_val]
            probs.append(len(opt_nbrs) / len(true_vals))
        return np.mean(probs)

    def opt_nbr(self, gtup, policy, rev=False):
        '''
        See if the optimal move given by the policy coicides with the trust
        dist's optimal move
        '''
        true_vals = self.nbr_values(gtup)
        opt_val = min(true_vals.values())
        opt_nbrs = [n for n, dist in true_vals.items() if dist == opt_val]

        pol_vals = self.nbr_values(gtup, policy)
        if rev:
            opt_pol_nbr = max(pol_vals, key=pol_vals.get)
        else:
            opt_pol_nbr = min(pol_vals, key=pol_vals.get)
        return opt_pol_nbr in opt_nbrs

    def nbr_values(self, gtup, func=None):
        if func is None:
            func = self.__getitem__

        vals = {}
        for n in self.nbr_func(gtup):
            vals[n] = func(n)

        return vals

    # TODO: this is not a proper way doing train test split for perm problem
    def train_test(self, test_ratio):
        perms = list(self.dist_dict.keys())
        np.random.shuffle(perms)
        k = int(test_ratio * len(self.dist_dict))
        test_perms = perms[:k]
        train_perms = perms[k:]
        test_y = np.array([self.dist_dict[p] for p in test_perms])
        train_y = np.array([self.dist_dict[p] for p in train_perms])
        return train_perms, train_y, test_perms, test_y

    def benchmark_policy(self, gtups, policy, rev=False):
        if len(gtups) == 0:
            return -1

        with torch.no_grad():
            ncorrect = 0
            for g in gtups:
                ncorrect += int(self.opt_nbr(g, policy, rev))
            return ncorrect / len(gtups)

    def opt_move_tup(self, tup):
        dist = self.dist_dict[tup]

def px_mult(p1, p2):
    return tuple([p1[p2[x] - 1] for x in range(len(p1))])

def nbrs(p):
    return [px_mult(g, p) for g in S8_GENERATORS]

def test():
    fname = '/home/hopan/github/idastar/s8_dists_red.txt'
    eye = (1, 2, 3, 4, 5, 6, 7, 8)
    pdf = PermDF(fname, nbrs)
    policy = lambda g: g.index(8)
    print(pdf.opt_nbr(eye, policy))

if __name__ == '__main__':
    test()
