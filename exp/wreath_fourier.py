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
sys.path.append('../cube/')
from utils import check_memory
from cube_irrep import Cube2Irrep
from complex_utils import cmm
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
        self.nout = 1

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

    def cache_perms(self):
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
        return output.argmax()

    def opt_move_tup(self, tup_nbrs):
        tens_nbrs = self.to_tensor(tup_nbrs).to(device)
        return self.opt_move(tens_nbrs)

    def eval_opt_nbr(self, nbrs, nnbrs):
        nbr_eval = self.forward_tup(nbrs).reshape(-1, nnbrs)
        max_nbr_vals, idx = nbr_eval.max(dim=1, keepdim=True)
        return max_nbr_vals.detach(), idx

    def set_fhats(self, fhat_dict):
        idx = 0
        for irr, fhat in fhat_dict.items():
            dsq = fhat.shape[0] ** 2
            coeff = fhat.shape[0] ** 0.5
            mat = torch.from_numpy(fhat).float().reshape(-1, 1) * coeff
            self.w_torch.data[idx: idx + dsq] = mat
            idx += dsq
        self.optmin = True

class CubePolicy(nn.Module):
    '''
    Needs to implement:
    - to_tensor
    - forward
    - nout
    '''
    def __init__(self, irreps, std=0.1, irrep_loaders=None):
        super(CubePolicy, self).__init__()
        self.irreps = irreps
        self.alphas = [i[0] for i in irreps]
        self.parts = [i[1] for i in irreps]
        if irrep_loaders is None:
            self.irrep_loaders = self.load_all_irreps()
        else:
            self.irrep_loaders = irrep_loaders

        self.dim = self._get_dim()
        self.nout = 1
        self.wi = nn.Parameter(torch.zeros(self.dim, 1))
        self.wr = nn.Parameter(torch.zeros(self.dim, 1))
        self.init_params(std)

    def init_params(self, std):
        for p in self.parameters():
            p.data.normal_(std)

    def _get_dim(self):
        ident = ((0,) * 8, tuple(range(1, 9)))
        re, im = self.to_tensor([ident])
        return re.numel()

    def load_all_irreps(self):
        irrep_loaders = []
        print('Starting load')
        st = time.time()
        for a, p in self.irreps:
            irrep_loaders.append(Cube2Irrep(a, p, sparse=True))
            print(f'done loading {a}, {p} | Elapsed: {time.time()-st:.2f}s')

        return irrep_loaders

    def to_tensor(self, gs):
        res = []
        ims = []
        for g in gs:
            g_res = []
            g_ims = []
            for loader in self.irrep_loaders:
                re, im = loader.tup_to_irrep_sp(g[0], g[1])
                g_res.append(re)
                g_ims.append(im)

            res.append(torch.cat(g_res, dim=1))
            ims.append(torch.cat(g_ims, dim=1))

        return torch.cat(res, dim=0).to(device), torch.cat(ims, dim=0).to(device)

    def forward_complex(self, xr, xi):
        return cmm(xr, xi, self.wr, self.wi)

    def forward(self, re_im_tensor):
        xr, xi = re_im_tensor
        yr, yi = self.forward_complex(xr, xi)
        return yr

class CubePolicyNoimag(CubePolicy):
    def __init__(self, irreps, std=0.1, irrep_loaders=None):
        super(CubePolicyNoimag, self).__init__(irreps, std, irrep_loaders)

    def to_tensor(self, gs):
        res = []
        ims = []
        for g in gs:
            g_res = []
            g_ims = []
            for loader in self.irrep_loaders:
                #re, im = loader.tup_to_irrep_sp(g[0], g[1])
                re, im = loader.tup_to_irrep_sp_noimag(g[0], g[1])
                g_res.append(re)
                g_ims.append(im)

            res.append(torch.cat(g_res, dim=1))
            ims.append(torch.cat(g_ims, dim=1))

        return torch.cat(res, dim=0).to(device), torch.cat(ims, dim=0).to(device)

class CubePolicyLowRank(CubePolicy):
    def __init__(self, irreps, rank, std=0.1, irrep_loaders=None):
        super(CubePolicyLowRank, self).__init__(irreps, std=std, irrep_loaders=irrep_loaders)
        # this is kind of hacky...
        if hasattr(self, 'wr'):
            del self.wr
        if hasattr(self, 'wi'):
            del self.wi

        self.irreps = irreps
        self.alphas = [i[0] for i in irreps]
        self.parts = [i[1] for i in irreps]
        if irrep_loaders is None:
            self.irrep_loaders = self.load_all_irreps()
        else:
            self.irrep_loaders = irrep_loaders

        self.dim = self._get_dim()
        self._n = int(self.dim ** 0.5)
        self.nout = 1
        self.uvecs_r = nn.Parameter(torch.zeros(self._n, rank))
        self.uvecs_i = nn.Parameter(torch.zeros(self._n, rank))
        self.vvecs_r = nn.Parameter(torch.zeros(self._n, rank))
        self.vvecs_i = nn.Parameter(torch.zeros(self._n, rank))
        #self.sigmas = nn.Parameter(torch.zeros(rank))
        self.init_params(std)

    def _get_w(self):
        #svvecs_r = self.sigmas * self.vvecs_r
        #svvecs_i = self.sigmas * self.vvecs_i
        svvecs_r = self.vvecs_r
        svvecs_i = self.vvecs_i

        wr, wi = cmm(self.uvecs_r, self.uvecs_i, svvecs_r.T, svvecs_i.T)
        return wr, wi

    def forward_complex(self, xr, xi):
        wr, wi = self._get_w()
        return cmm(xr, xi, wr.view(self.dim, 1), wi.view(self.dim, 1))

class CubePolicyLowRankEig(CubePolicy):
    def __init__(self, irreps, rank, std=0.1, irrep_loaders=None):
        super(CubePolicyLowRankEig, self).__init__(irreps, std=std, irrep_loaders=irrep_loaders)
        # this is kind of hacky...
        if hasattr(self, 'wr'):
            del self.wr
        if hasattr(self, 'wi'):
            del self.wi

        self.irreps = irreps
        self.alphas = [i[0] for i in irreps]
        self.parts = [i[1] for i in irreps]
        if irrep_loaders is None:
            self.irrep_loaders = self.load_all_irreps()
        else:
            self.irrep_loaders = irrep_loaders

        self.dim = self._get_dim()
        self._n = int(self.dim ** 0.5)
        self.nout = 1
        self.uvecs_r = nn.Parameter(torch.zeros(self._n, rank))
        self.uvecs_i = nn.Parameter(torch.zeros(self._n, rank))
        self.init_params(std)

    def _get_w(self):
        wr, wi = cmm(self.uvecs_r, self.uvecs_i, self.uvecs_r.T, self.uvecs_i.T)
        return wr, wi

    def forward_complex(self, xr, xi):
        wr, wi = self._get_w()
        return cmm(xr, xi, wr.view(self.dim, 1), wi.view(self.dim, 1))


def main():
    irreps = [
        ((2, 3, 3), ((2,), (1, 1, 1), (1, 1, 1))),
        ((2, 3, 3), ((2,), (3,), (1, 1, 1)))
    ]
    #irreps = [((4, 2), ((2, 2), (1, 1)))]
    #pol = WreathPolicy(irreps, '/local/hopan/pyraminx/irreps/')
    ident = ((0,) * 8, tuple(range(1, 9)))
    w = CubePolicy(irreps)
    w.to_tensor(ident)

if __name__ == '__main__':
    main()
