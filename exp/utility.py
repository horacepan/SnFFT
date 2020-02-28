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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def str_val_results(dic):
    dstr = ''
    for i, num in dic.items():
        dstr += '{}: {:.2f} |'.format(i, num)
    return dstr

def px_mult(p1, p2):
    return tuple([p1[p2[x] - 1] for x in range(len(p1))])

def nbrs(p):
    return [px_mult(g, p) for g in S8_GENERATORS]

def can_solve(state, policy, max_moves, env, perm_df=None):
    '''
    state: tuple
    policy: nn policy
    max_moves: int
    Returns: True if policy solves(gets to a finished state) within max_moves
    '''
    visited = {}
    curr_state = state
    for _ in range(max_moves):
        visited[curr_state] = visited.get(curr_state, 0) + 1
        neighbors = env.nbrs(curr_state)
        opt_move = policy.opt_move_tup(neighbors)
        curr_state = env.step(curr_state, opt_move)
        if env.is_done(curr_state) or perm_df.distance(curr_state) == 1:
            return True

    return False

def val_model(policy, max_dist, perm_df, cnt=100, env=None):
    '''
    To validate a model need:
    - transition function
    - means to generate states (or pass them in)
    - policy to evluate
    - check if policy landed us in a done state
    '''
    # generate k states by taking a random walk of length d
    # up to size
    nsolves = {}
    for dist in range(1, max_dist + 1):
        d_states = perm_df.random_states(dist, cnt)
        solves = 0
        for state in d_states:
            solves += can_solve(state, policy, 15, env, perm_df)
        nsolves[dist] = solves / len(d_states)
    return nsolves

def test_model(policy, scramble_len, cnt, max_moves, perm_df, env):
    states = [env.random_element(scramble_len) for _ in range(cnt)]
    return _test_model(policy, states, max_moves, perm_df, env)

def _test_model(policy, states, max_moves, perm_df, env):
    stats =  {}
    dists = {}
    solves = 0
    for s in states:
        solved =  int(can_solve(s, policy, max_moves, env, perm_df))
        solves += solved
        d = perm_df.distance(s) # should probably avoid this sort of api
        dists[d] = dists.get(d, 0) + 1
        stats[d] = stats.get(d, 0) + solved

    for d in dists.keys():
        stats[d] = stats[d] / dists[d]

    return solves/len(states), dists, stats

def test_all_states(policy, max_moves, perm_df, env):
    states = perm_df.all_states()
    return _test_model(policy, states, max_moves, perm_df, env)

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

def perm_onehot(perms):
    '''
    List of tuples (perms) to create one hot vector tensor for
    '''
    d = len(perms[0])
    tensor = torch.zeros(len(perms), d * d)
    k = 0

    for idx in range(len(perms)):
        perm = perms[idx]
        k = 0
        for i in perm:
            tensor[idx, i + k - 1] = 1
            k += d

    return tensor.to(device)

def wreath_onehot(wtups, wcyc):
    otups, ptups = zip(*wtups)
    n = len(perm_tups[0])
    perm_part = perm_onehot(perm_tups)
    or_part = torch.zeros(len(otups), wcyc * n)
    for idx, otup in enumerate(otups):
        k = 0
        for ori in otups:
            tensor[idx, ori + k] = 1
            k += wcyc

    return torch.cat([or_part, perm_part], dim=1)

class ReplayBuffer:
    def __init__(self, state_size, capacity):
        self.state_size = state_size
        self.states = torch.zeros(capacity, state_size)
        self.next_states = torch.zeros(capacity, state_size)
        self.state_tups = [None] * capacity
        self.next_state_tups = [None] * capacity
        self.rewards = torch.zeros(capacity, 1)
        self.actions = torch.zeros(capacity, 1)
        self.dones = torch.zeros(capacity, 1)
        self.capacity = capacity
        self.filled = 0
        self._idx = 0

    def push(self, state, action, next_state, reward, done, state_tup, next_state_tup):
        self.states[self._idx] = state
        self.actions[self._idx] = action
        self.next_states[self._idx] = next_state
        self.rewards[self._idx] = reward
        self.dones[self._idx] = done
        self.filled = min(self.capacity, self.filled + 1)
        self.state_tups[self._idx] = state_tup
        self.next_state_tups[self._idx] = next_state_tup
        self._idx = (self._idx + 1) % self.capacity

    def sample(self, batch_size, device):
        '''
        Returns a tuple of the state, next state, reward, dones
        '''
        size = min(self.filled, batch_size)
        idxs = np.random.choice(self.filled, size=size)
        tups = [self.state_tups[i] for i in idxs]
        next_tups = [self.next_state_tups[i] for i in idxs]

        bs = self.states[idxs].to(device)
        ba = self.actions[idxs].to(device)
        bns = self.next_states[idxs].to(device)
        br = self.rewards[idxs].to(device)
        bd = self.dones[idxs].to(device)
        return bs, ba, bns, br, bd, tups, next_tups

# Source: https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def update_params(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def debug_mem():
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device)
        except:
            pass
