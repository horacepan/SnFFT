import pdb
import numpy as np
import pandas as pd

GENS_RED = [
    (8, 1, 3, 4, 5, 6, 2, 7),
    (2, 7, 3, 4, 5, 6, 8, 1),
    (2, 3, 4, 1, 5, 6, 7, 8),
    (4, 1, 2, 3, 5, 6, 7, 8),
    (1, 2, 3, 4, 7, 5, 8, 6),
    (1, 2, 3, 4, 6, 8, 5, 7)
]

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
        probs = []
        for p in gtups:
            true_vals = self.nbr_values(p)
            opt_val = min(true_vals.values())
            opt_nbrs = [n for n, dist in true_vals.items() if dist == opt_val]
            probs.append(len(opt_nbrs) / len(true_vals))
        return np.mean(probs)

    def opt_nbr(self, gtup, policy):
        '''
        See if the optimal move given by the policy coicides with the trust
        dist's optimal move
        '''
        true_vals = self.nbr_values(gtup)
        opt_val = min(true_vals.values())
        opt_nbrs = [n for n, dist in true_vals.items() if dist == opt_val]

        pol_vals = self.nbr_values(gtup, policy)
        opt_pol_nbr = min(pol_vals, key=pol_vals.get)
        return opt_pol_nbr in opt_nbrs

    '''
    def pol_correct(self, gtup, policy):
        true_opts = self.opt_nbr(gtup, self._dist_dict)
        pol_opts = self.opt_nbr(gtup, policy)
        return pol_opts[0] in true_opts
    '''

    def nbr_values(self, gtup, func=None):
        if func is None:
            func = self.__getitem__

        vals = {}
        for n in self.nbr_func(gtup):
            vals[n] = func(n)

        return vals

    def train_test_split(self, test_ratio):
        perms = list(self.dist_dict.keys())
        np.random.shuffle(perms)
        test_perms = perms[: int(test_ratio * len(self.dist_dict))]
        train_dict = {}
        test_dict = {}

        for p in test_perms:
            vals = self.nbr_values(p)
            vals[p] = self.dist_dict[p]
            test_dict.update(vals)

        for p, val in self.dist_dict.items():
            if p not in test_dict:
                train_dict[p] = val

        train_p, train_y = zip(*train_dict.items())
        test_p, test_y = zip(*test_dict.items())
        return list(train_p), list(train_y), list(test_p), list(test_y)

def px_mult(p1, p2):
    return tuple([p1[p2[x] - 1] for x in range(len(p1))])

def nbrs(p):
    return [px_mult(g, p) for g in GENS_RED]

def test():
    fname = '/home/hopan/github/idastar/s8_dists_red.txt'
    eye = (1, 2, 3, 4, 5, 6, 7, 8)
    pdf = PermDF(fname, nbrs)
    policy = lambda g: g.index(8)
    print(pdf.opt_nbr(eye, policy))

if __name__ == '__main__':
    test()
