import torch
import torch.nn as nn
import torch.nn.functional as F

def cmm(t1r, t1i, t2r, t2i):
    rr = t1r.mm(t2r)
    ri = t1r.mm(t2i)
    ir = t1i.mm(t2r)
    ii = t1i.mm(t2i)
    return (rr-ii, ri + ir)

def cmm_sparse(sparse_r, sparse_i, wr, wi):
    rr = torch.sparse.mm(sparse_r, wr)
    ri = torch.sparse.mm(sparse_r, wi)
    ir = torch.sparse.mm(sparse_i, wr)
    ii = torch.sparse.mm(sparse_i, wi)
    return (rr - ii, ri + ir)

def cmse(ytr, yti, yr, yi):
    real_diff = ytr - yr
    im_diff = yti - yi
    loss = (real_diff.pow(2) + im_diff.pow(2)).mean()
    return loss

def cmse_min_imag(ytr, yti, yr, yi):
    '''
    The real parts should be close. The imaginary parts should also be close
    and they should be small.
    '''
    real_diff = ytr - yr
    im_diff = yti - yi
    loss = (real_diff.pow(2) + im_diff.pow(2) + yti.pow(2) + yi.pow(2)).mean()
    return loss

def cmse_real(ytr, yti, yr, yi):
    real_diff = ytr - yr
    return real_diff.pow(2).mean()
