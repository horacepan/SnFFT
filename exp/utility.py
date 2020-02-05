import pdb
import os
import psutil
import pickle
import numpy as np
import torch

S8_GENERATORS = [
    (8, 1, 3, 4, 5, 6, 2, 7),
    (2, 7, 3, 4, 5, 6, 8, 1),
    (2, 3, 4, 1, 5, 6, 7, 8),
    (4, 1, 2, 3, 5, 6, 7, 8),
    (1, 2, 3, 4, 7, 5, 8, 6),
    (1, 2, 3, 4, 6, 8, 5, 7)
]
def px_mult(p1, p2):
    return tuple([p1[p2[x] - 1] for x in range(len(p1))])

def nbrs(p):
    return [px_mult(g, p) for g in S8_GENERATORS]

def s8_move(ptup, gidx):
    '''
    Returns g \dot ptup, where g is the group element correspondig to gidx
    '''
    g = S8_GENERATORS[gidx]
    return px_mult(g, ptup)

def get_batch(xs, ys, size):
    idx = np.random.choice(len(xs), size=size)
    return [xs[i] for i in idx], np.array([ys[i] for i in idx]).reshape(-1, 1)

def load_yor(irrep, prefix):
    '''
    irrep: tuple
    prefix: directory to load from
    Assumption: the pkl files seprate the parts of the tuple with an underscore
    Ex: (2, 2) -> 2_2.pkl
    '''
    fname = os.path.join(prefix, '_'.join(str(i) for i in irrep) + '.pkl')
    pkl = pickle.load(open(fname, 'rb'))
    return pkl

def load_np(irrep, prefix):
    fname = os.path.join(prefix, str(irrep) + '.npy')
    return np.load(fname)

def cg_mat(p1, p2, p3, prefix='/local/hopan/irreps/s_8/cg'):
    if not os.path.exists('/local/hopan/irreps/s_8/cg'):
        prefix = '/scratch/hopan/irreps/s_8/cg'

    def sflat(tup):
        return ''.join(str(x) for x in tup)
    fname = os.path.join(prefix, '{}_{}_{}.npy'.format(
        sflat(p1), sflat(p2), sflat(p3)
    ))
    return np.load(fname)


def th_kron(a, b):
    """
    Attribution: https://gist.github.com/yulkang/4a597bcc5e9ccf8c7291f8ecb776382d
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)

def check_memory(verbose=True):
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    if verbose:
        print("Consumed {:.2f}mb memory".format(mem))
    return mem

def perm_onehot(perms, batchdim=True):
    n = len(perms)
    d = len(perms[0]) ** 2
    tensor = torch.zeros(n, d)
    k = 0

    for idx in range(n):
        perm = perms[idx]
        k = 0
        for i in perm:
            tensor[idx, i + k - 1] = 1
            k += n

    return tensor

class ReplayBuffer:
    def __init__(self, state_size, capacity):
        self.states = torch.zeros(capacity, state_size)
        self.next_states = torch.zeros(capacity, state_size)
        self.rewards = torch.zeros(capacity, 1)
        self.dones = torch.zeros(capacity, 1)
        self.capacity = capacity
        self.filled = 0
        self._idx = 0

    def push(self, state, next_state, reward, done):
        self.states[self._idx] = state
        self.next_states[self._idx] = next_state
        self.rewards[self._idx] = reward
        self.dones[self._idx] = done
        self.filled = min(self.capacity, self.filled + 1)
        self._idx = (self._idx + 1) % self.capacity

    def sample(self, batch_size):
        '''
        Returns a tuple of the state, next state, reward, dones
        '''
        size = min(self.filled, batch_size)
        idxs = np.random.choice(self.filled, size=size)
        return (self.states[idxs], self.next_states[idxs], self.rewards[idxs], self.dones[idxs])

def debug_mem():
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device)
        except:
            pass
