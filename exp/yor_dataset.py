import time
import math
from itertools import permutations
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utility import S8_GENERATORS, load_yor, check_memory


def gtup_to_vec(gtup, yor_dicts, irreps):
    vecs = []
    for irr in irreps:
        mat = yor_dicts[irr][gtup]
        vecs.append(torch.from_numpy(mat.reshape(1, -1)))
    return torch.cat(vecs, dim=1).float()

def yor_tensor_dataset(df_fname, irreps, yor_prefix):
    '''
    df_fname: loc of where csv is located
    irreps: list of tuples (which denote irreps)
    yor_prefix: directory prefix containing the yor pkls
    Returns: TensorDataset where each group element has been converted
        into its vectorized yor rep
    '''
    df = pd.read_csv(df_fname, header=None, dtype={0: str, 1: int})
    n = sum(irreps[0])
    gsize = math.factorial(n)
    yors = {irr: load_yor(irr, yor_prefix)for irr in irreps}
    df = pd.read_csv(df_fname, header=None, dtype={0: str, 1: int})
    eye_tup = tuple(i for i in range(1, n+1))
    vec_size = 0

    for irr, dic in yors.items():
        vec_size += (dic[eye_tup].shape[0] ** 2)

    # loop over permutations
    tensor = torch.zeros(gsize, vec_size)
    dists = torch.FloatTensor(df[1])

    for idx, gtup in tqdm(enumerate(permutations(eye_tup))):
        tensor[idx] = gtup_to_vec(gtup, yors, irreps)

    return tensor, dists

if __name__ == '__main__':
    st = time.time()
    irreps = [(3,2,2, 1), (8,), (4, 2, 2)]
    #irreps = [(8,)]
    yor_pref = '/local/hopan/irreps/s_8/'
    df_fname = '/home/hopan/github/idastar/s8_dists_red.txt'
    tdataset = yor_tensor_dataset(df_fname, irreps, yor_pref)
    print('Load dataset time: {:.2f}s'.format(time.time() - st))
    check_memory()
