import pdb
import time
import math
from itertools import permutations
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from utility import S8_GENERATORS, load_yor, check_memory


def ptup_df(fname):
    df = pd.read_csv(fname, header=None, dtype={0: str, 1:int})
    df[0] = df[0].apply(lambda x: tuple(int(i) for i in x))
    df.columns = ['perm', 'dist']
    return df

def gtup_to_vec(gtup, yor_dicts, irreps):
    vecs = []
    for irr in irreps:
        mat = yor_dicts[irr][gtup]
        vecs.append(torch.from_numpy(mat.reshape(1, -1)) * math.sqrt(mat.shape[0]))
    return torch.cat(vecs, dim=1).float()

def yor_tensor(irreps, yor_prefix, gtups):
    '''
    irreps: list of tuples (which denote irreps)
    yor_prefix: directory prefix containing the yor pkls
    gtups: list of tuples (perm)
    Returns: tensor where each row is the vectorized concated irep of the corresponding
        group element
    '''
    n = sum(irreps[0])
    gsize = math.factorial(n)
    yors = {irr: load_yor(irr, yor_prefix) for irr in irreps}
    eye_tup = tuple(i for i in range(1, n+1))
    vec_size = 0

    for irr, dic in yors.items():
        vec_size += (dic[eye_tup].shape[0] ** 2)

    tensor = torch.zeros(len(gtups), vec_size)

    #for idx, gtup in tqdm(enumerate(permutations(eye_tup))):
    for idx, gtup in tqdm(enumerate(gtups)):
        tensor[idx] = gtup_to_vec(gtup, yors, irreps)

    return tensor

def train_test_datasets(df_fname, yor_prefix, irreps, test_ratio):
    df = ptup_df(df_fname)
    df = df.sample(frac=1) # shuffle
    test_size = int(len(df) * test_ratio)
    train_df = df.iloc[test_size:]
    test_df = df.iloc[:test_size]

    train_x = yor_tensor(irreps, yor_prefix, train_df['perm'])
    test_x  = yor_tensor(irreps, yor_prefix, test_df['perm'])
    train_y = torch.FloatTensor(np.array(train_df['dist']))
    test_y  = torch.FloatTensor(np.array(test_df['dist']))

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    return train_dataset, test_dataset


class YorConverter:
    def __init__(self, irreps, yor_prefix, perms):
        self.irreps = irreps
        self.yors = {irr: load_yor(irr, yor_prefix) for irr in irreps}
        self.pdict = self.cache_perms(perms)
        self.dim = self.pdict[tuple(i for i in range(1, 1+ sum(irreps[0])))]

    def cache_perms(self, perms):
        '''
        perms: list of tuples
        Returns a dictionary mapping tuple to its tensor representation
        '''
        pdict = {}
        for p in perms:
            pdict[p] = torch.from_numpy(self.to_irrep(p)).float()
        self.pdict = pdict
        return pdict

    def to_irrep(self, gtup):
        '''
        Loop over irreps -> cat the irrep (via the yor dicts) -> reshape
        gtup: perm tuple
        '''
        irrep_vecs = []
        for irr in self.irreps:
            rep = self.yors[irr][gtup]
            dim = rep.shape[0]
            vec = rep.reshape(1, -1) * np.sqrt(dim)
            irrep_vecs.append(vec)
        irrep_vecs.append(np.array([[1]]))
        return np.hstack(irrep_vecs)

    def __call__(self, perm):
        return self.pdict[perm]

    def __getitem__(self, perm):
        return self.pdict[perm]

if __name__ == '__main__':
    perms = list(permutations((1,2,3,4,5,6,7,8)))
    tens = YorConverter([(3,2,2,1)], '/local/hopan/irreps/s_8', perms)
    print(tens[perms[3]])
