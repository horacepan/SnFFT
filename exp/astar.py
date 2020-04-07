import pdb
from queue import PriorityQueue
import time
import random
import os
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from perm_df import WreathDF
from rlmodels import MLP, MLPResModel
from wreath_fourier import CubePolicy
from utility import wreath_onehot
from logger import get_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def a_star(state, done_state, perm_df, heuristic, max_explorable):
    to_visit = PriorityQueue()
    nodes_explored = 0
    to_visit.put((0, 0, state, perm_df.distance(state)))
    icnt = 0
    visited = set()

    while not to_visit.empty():
        val, dist, curr_node, true_dist = to_visit.get()
        nodes_explored += 1
        if curr_node == done_state:
            return True, nodes_explored

        for n in perm_df.nbrs(curr_node):
            if n == done_state:
                return True, nodes_explored
            elif n in visited:
                continue

            fitness = heuristic(n)
            to_visit.put((fitness, dist + 1, n, perm_df.distance(n)))

        visited.add(curr_node)
        icnt += 1
        if icnt > max_explorable:
            return False, False

    return False, False

def test_heuristic(perm_df, func, _states=None, max_explorable=100):
    if _states is None:
        _states = [perm_df.random_state(1000 + (1 if random.random() < 0.5 else 0)) for _ in range(1000)]

    done_state = perm_df.random_state(0)
    astar_solves = []
    states = []
    astar_explored = []
    st = time.time()

    for state in tqdm(_states):
        solved, explored = a_star(state, done_state, perm_df, func, max_explorable=max_explorable)
        astar_solves.append(solved)
        states.append(state)
        astar_explored.append(explored)
    end = time.time()
    #print('Elapsed: {:.2f}s | Solves: {}'.format(end - st, sum(astar_solves)))
    return astar_solves, _states, astar_explored

def gen_mlp_model(model):
    def func(x):
        return -model.forward(model.to_tensor([x])).item()
    return func

def gen_irrep_heuristic(mlp):
    def mlp_heuristic(tup):
        re, im =  mlp.to_tensor([tup])
        return -mlp.forward_complex(re, im)[0].item()
    return mlp_heuristic

def get_model(irreps, fn):
    model = CubePolicy(irreps)
    model.to(device)
    sd = torch.load(fn, map_location=device)
    model.load_state_dict(sd)
    return model

def main():
    log = get_logger(None, stdout=True, tofile=False)
    res_fn = '/scratch/hopan/cube/irreps/cube2res/nres1_512_1024_25len/seed_1/model_270000.pt'
    irr1_fn = '/home/hopan//github/SnFFT/exp/logs/cube2test/233_2111111_100k/seed_0/model_{e}.pt'
    irr2_fn = '/home/hopan//github/SnFFT/exp/logs/cube2test/2111111_41111_200kcap_gpu_nodone/seed_1/model_80000.pt'

    to_wreath = lambda g: wreath_onehot(g, 3).to(device)
    res_mlp = MLPResModel(88, 512, 1024, 1, nres=1, to_tensor=to_wreath)
    res_mlp.to(device)
    res_mlp.load_state_dict(torch.load(res_fn, map_location=device))
    log.info('Loaded res mlp model')

    # load irreps
    irrep1 = [((2, 3, 3), ((2,), (1, 1, 1), (1, 1, 1)))]
    irrep2 = [((4, 2, 2), ((4,), (1, 1), (1, 1))), ((2, 3, 3), ((2,), (1, 1, 1), (1, 1, 1)))]
    irrep1_net = get_model(irrep1, irr1_fn)
    log.info('Loaded 1 irrep models')
    irrep2_net = get_model(irrep2, irr2_fn)
    log.info('Loaded 2 irrep models')

    prefix = 'scratch' if os.path.exists('/scratch/hopan/cube/cube_sym_mod_tup.txt') else 'local'
    perm_df = WreathDF(f'/{prefix}/hopan/cube/cube_sym_mod_tup.txt', 6, 3, f'/{prefix}/hopan/cube/cube_sym_mod_tup.pkl')
    log.info('Loaded perm df')
    heuristic = gen_mlp_model(res_mlp)
    irr1_heuristic = gen_irrep_heuristic(irrep1_net)
    irr2_heuristic = gen_irrep_heuristic(irrep2_net)

    cnt = 1000
    max_explorable = 1000
    _states = [perm_df.random_state(1000 + (1 if random.random() < 0.5 else 0)) for _ in range(cnt)]

    st = time.time()
    log.info('Starting 2 irreps...')
    solve_status_2, _, explored2 = test_heuristic(perm_df, irr2_heuristic, _states, max_explorable=100)
    log.info('2 irrep | Solves: {:.3f} | Avg: {:.2f} | Elapsed: {:.2f}mins'.format(sum(solve_status_2) / len(_states), np.mean(explored2), (time.time()-st) / 60))

    st = time.time()
    log.info('Starting 1 irreps...')
    solve_status_1, _, explored1 = test_heuristic(perm_df, irr1_heuristic, _states, max_explorable=100)
    log.info('1 irrep | Solves: {:.3f} | Avg: {:.2f} | Elapsed: {:.2f}mins'.format(sum(solve_status_1) / len(_states), np.mean(explored1), (time.time()-st) / 60))

    st = time.time()
    log.info('Starting res with max exp budget = {} | cnt = {} ...'.format(max_explorable, cnt))
    solve_status_res, _states, explored = test_heuristic(perm_df, heuristic, _states, max_explorable=1000)
    log.info('Res net | Solves: {:.3f} | Elapsed: {:.2f}mins'.format(sum(solve_status_res) / len(_states), (time.time() - st) / 60))

    pdb.set_trace()

if __name__ == '__main__':
    main()
