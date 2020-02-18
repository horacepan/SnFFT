import sys
import pdb
import time
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import load_yor, cg_mat, S8_GENERATORS, th_kron
from cg_utility import proj, compute_rhs_block, compute_reduced_block
from logger import get_logger

sys.path.append('../')
from utils import check_memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pyraminx_irreps(alpha, parts, prefix):
    pkl_name = os.path.join(prefix, str(alpha), str(parts) + '.pkl')
    return pickle.load(open(pkl_name, 'rb'))

class WreathPolicy(nn.Module):
    def __init__(self, irreps, pkl_prefix, rep_dict=None, pdict=None):
        super(WreathPolicy, self).__init__()
        fnames = [load_pyraminx_irreps(alpha, parts, pkl_prefix) for (alpha, parts) in irreps]
        self.irreps = irreps

        if rep_dict is not None:
            self.rep_dict = rep_dict
        else:
            self.rep_dict = {(alpha, parts): load_pyraminx_irreps(alpha, parts, pkl_prefix)
                              for (alpha, parts) in irreps}

        if pdict is not None:
            self.pdict = pdict
        else:
            self.pdict = self.cache_perms()

        self.w_torch = nn.Parameter(torch.rand(self.get_dim() + 1, 1))
        self.optmin = False

    def get_dim(self):
        n = sum(self.irreps[0][0])
        ident = (tuple(0 for _ in range(n)), tuple(range(1, n+1)))
        tot = 0
        for irr, dic in self.rep_dict.items():
            tot += (dic[ident].shape[0] ** 2)
        return tot

    def to_irrep(self, gtup):
        irrep_vecs = []
        for irr in self.irreps:
            rep = self.rep_dict[irr][gtup]
            dim = rep.shape[0]
            vec = rep.reshape(1, -1) * np.sqrt(dim)
            irrep_vecs.append(vec)
        irrep_vecs.append(np.array([[1]]))
        return np.hstack(irrep_vecs)

    def cache_perms(self, perms=None):
        if perms is None:
            perms = self.rep_dict[self.irreps[0]].keys()

        pdict = {}
        for p in perms:
            pdict[p] = torch.from_numpy(self.to_irrep(p)).float()
        self.pdict = pdict
        return pdict

    def compute_loss(self, perms, y):
        y_pred = self.forward_tup(perms)
        return F.mse_losss(y, y_pred)

    def _reshaped_mats(self):
        '''
        Return dict mapping irrep tuple -> square tensor
        '''
        fhats = {}
        idx = 0
        n = sum(self.irreps[0][0])
        ident = (tuple(0 for _ in range(n)), tuple(range(1, n+1)))

        for irr in self.irreps:
            size = self.rep_dict[irr][ident].shape[0]
            fhats[irr] = self.w_torch[idx: idx + (size * size), :1].reshape(size, size)
            idx += (size * size)
        return fhats

    def to_tensor(self, perms):
        X = torch.cat([self.pdict[p] for p in perms], dim=0).to(device)
        return X

    def forward(self, tensor):
        y_pred = tensor.matmul(self.w_torch)
        return y_pred

    def forward_tup(self, perms):
        X_th = self.to_tensor(perms).to(device)
        y_pred = X_th.matmul(self.w_torch)
        return y_pred

    def opt_move(self, x):
        output= self.forward(x)
        if self.optmin:
            return output.argmin()
        return output.argmax()

    def opt_move_tup(self, tup_nbrs):
        tens_nbrs = self.to_tensor(tup_nbrs).to(device)
        return self.opt_move(tens_nbrs)

    def eval_opt_nbr(self, nbrs, nnbrs):
        nbr_eval = self.forward_tup(nbrs).reshape(-1, nnbrs)
        max_nbr_vals = nbr_eval.max(dim=1, keepdim=True)[0]
        return max_nbr_vals

    def set_fhats(self, fhat_dict):
        idx = 0
        for irr, fhat in fhat_dict.items():
            dsq = fhat.shape[0] ** 2
            coeff = fhat.shape[0] ** 0.5
            mat = torch.from_numpy(fhat).float().reshape(-1, 1) * coeff
            self.w_torch.data[idx: idx + dsq] = mat
            idx += dsq
        self.optmin = True

def main():
    irreps = [((4, 2), ((2, 2), (1, 1)))]
    pol = WreathPolicy(irreps, '/local/hopan/pyraminx/irreps/')

if __name__ == '__main__':
    main()
