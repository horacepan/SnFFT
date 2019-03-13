import torch
import torch.nn as nn
import torch.nn.functional as F

def cmm(t1r, t1i, t2r, t2i):
    rr = t1r.mm(t2r)
    ri = t1r.mm(t2i)
    ir = t1i.mm(t2r)
    ii = t1i.mm(t2i)
    return (rr-ii, ri + ii)


def cmse(y_true, yr, yi):
    real_diff = y_true - yr
    loss = (real_diff.pow(2) + yi.pow(2)).mean()
    return loss

def cmse_real(ytr, yti, yr, yi):
    '''
    The real parts should be close. The imaginary parts should also be close
    and they should be small.
    '''
    real_diff = ytr - yr
    im_diff = yti - yi
    loss = (real_diff.pow(2) + im_diff.pow(2) + yti.pow(2) + yi.pow(2)).mean()
    return loss
