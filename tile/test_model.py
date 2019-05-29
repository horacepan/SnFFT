import numpy as np
from mlp_main import mlp_get_action
from tile_env import *
import torch
import pandas as pd
from tile_utils import get_true_df, tup_to_str
from main import get_action

#TRUE_DISTS = get_true_df()
def eval_model(model, env, trials, max_iters, get_action_func):
    successes = []
    move_cnt = []
    puzzles = []
    tot_wins = 0
    tot_trials = 0
    shuffle_distances = [20 * x for x in range(1, 20)]
    for d in shuffle_distances:
        wins = 0
        for e in range(trials):
            states = env.shuffle(d)
            state = grid_to_onehot(env.grid)
            start_tup = onehot_to_tup(state)
            puzzles.append(start_tup)
            for i in range(max_iters):
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

def test(fname):
    model = torch.load('./models/{}'.format(fname))
    env = TileEnv(3, one_hot=True)
    trials = 100
    max_iters = 35
    eval_model(model, env, trials, max_iters, mlp_get_action)

if __name__ == '__main__':
    fname = '80k30k_64.pt'
    test(fname)
