import os
import numpy as np
from itertools import product
import sys
sys.path.append('../')
from utils import partitions, weak_partitions
import pandas as pd

PYRAMINX_GROUP_SIZE = 11520 * 4
def alpha_parts():
    irreps = []
    for alpha in weak_partitions(6, 2):
        for parts in product(partitions(alpha[0]), partitions(alpha[1])):
            irreps.append((alpha, parts))
    return irreps

def pyraminx_dists(fname):
    dist_dict = {}

    with open(fname, 'r') as f:
        for line in f.readlines():
            opart, ppart, dist = line.strip().split(',')
            otup = tuple(int(x) for x in opart)
            perm = tuple(int(x) for x in ppart)
            dist = int(dist)
            dist_dict[(otup, perm)] = dist

    return dist_dict

def dist_df(fname):
    df = pd.read_csv(fname, header=None, dtype={0: str, 1: str, 2: int})
    df[0] = df[0].map(lambda x: tuple(int(i) for i in x))
    df[1] = df[1].map(lambda x: tuple(int(i) for i in x))
    return df

def load_rep_mat(alpha, parts, prefix='/local/hopan/pyraminx/irreps_mat/'):
    fname = os.path.join(prefix, str(alpha), str(parts) + '.npy')
    return np.load(fname)

def load_rep_mat_sample(alpha, parts, nsample, prefix='/local/hopan/pyraminx/irreps_mat/'):
    reps = load_rep_mat(alpha, parts, prefix)
    idx = np.random.randint(len(reps), size=nsample)
    return reps[idx, :, :]

def load_mat_ift(alpha, parts, prefix='/local/hopan/pyraminx/fourier_eval/'):
    fname = os.path.join(prefix, str(alpha), str(parts) + '.npy')
    return np.load(fname)

def load_ft_sample(alpha, parts, nsample, prefix='/local/hopan/pyraminx/fourier_sample/'):
    fname = os.path.join(prefix, str(nsample), str(alpha), str(parts) + '.npy')
    return np.load(fname)
