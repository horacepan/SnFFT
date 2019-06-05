import os
import argparse
import numpy as np
from mlp_main import mlp_get_action
from tile_env import *
from tile_irrep_env import *
import torch
import pandas as pd
from tile_utils import get_true_df, tup_to_str
from main import get_action

#TRUE_DISTS = get_true_df()
def eval_model(model, env, trials, max_iters, get_action_func, irrep=False):
    successes = []
    move_cnt = []
    puzzles = []
    tot_wins = 0
    tot_trials = 0
    shuffle_distances = [20 * x for x in range(1, 21)]
    for d in shuffle_distances:
        wins = 0
        for e in range(trials):
            states = env.shuffle(d)
            grid_state = env.grid
            state = grid_to_onehot(env.grid)
            start_tup = onehot_to_tup(state)
            puzzles.append(start_tup)
            for i in range(max_iters):
                if irrep:
                    action = get_action_func(model, env, grid_state, e)
                    new_state, reward, done, info = env.step(action)
                    grid_state = info['grid']
                else:
                    action = get_action_func(model, env, state, e)
                    new_state, reward, done, _ = env.step(action)
                state = new_state
                if done:
                    successes.append(True)
                    move_cnt.append(i + 1)
                    #true_dist = TRUE_DISTS.loc[tup_to_str(start_tup)].distance
                    #print('Solved {} | True dist: {} | Net dist: {}'.format(
                    #    start_tup, true_dist, i+1
                    #))
                    wins += 1
                    break

            tot_trials += 1
            if not done:
                successes.append(False)

        print('Shuffle distance: {:3} | Solves: {:.3f}'.format(d, wins / trials))
        tot_wins += wins

    print('Validation | {} Trials | Solves: {}'.format(tot_trials, tot_wins))
    return puzzles, successes, move_cnt

def test(path, partitions):
    path = os.path.join('.', path)
    print('loading from: {}'.format(path))
    model = torch.load(path)
    trials = 100
    max_iters = 35
    irrep = False
    if 'irrep_models' in path:
        func = get_action
        env = TileIrrepEnv(3, partitions)
        irrep = True
    else:
        func = mlp_get_action
        env = TileEnv(3, one_hot=True)
        irrep = False 

    eval_model(model, env, trials, max_iters, func, irrep)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='models/80k30k_64.pt')
    parser.add_argument('--partitions', type=str, default='[(8, 1)]')
    args = parser.parse_args() 
    partitions = eval(args.partitions)

    test(args.path, partitions)
