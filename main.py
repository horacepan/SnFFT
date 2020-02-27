import os
import logging
import time
import random
import argparse
from collections import namedtuple
import json
import pdb
import sys
sys.path.append('./cube')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from complex_utils import cmm, cmse, cmse_min_imag, cmse_real, cmm_sparse
from irrep_env import Cube2IrrepEnv
from utils import check_memory
import str_cube
from tensorboardX import SummaryWriter

from dqn import * #ReplayMemory, IrrepLinreg, get_logger, update, explore_rate
from memory import ReplayMemory


IRREP_SIZE = {
    ((2, 3, 3), ((2,), (1, 1, 1), (1, 1, 1))): 560,
    ((2, 3, 3), ((1, 1), (3,), (3,))): 560,
    ((2, 3, 3), ((2,), (2, 1), (2, 1))): 2240,
    ((2, 3, 3), ((1, 1), (2, 1), (2, 1))): 2240,
    ((4, 2, 2), ((4,), (1, 1), (1, 1))) : 420,
    ((4, 2, 2), ((2, 2), (2,), (2,))) : 840,
    ((3, 1, 4), ((2, 1), (1,), (2, 1, 1))) : 1680,
    ((3, 1, 4), ((2, 1), (1,), (3, 1))) : 1680,
    ((8, 0, 0), ((8,), (), ())) : 1,
}
if os.path.exists('/local/hopan'):
    NP_IRREP_FMT = '/local/hopan/cube/fourier_unmod/{}/{}.npy'
    PREFIX = '/local/hopan/cube'
elif os.path.exists('/scratch/hopan'):
    NP_IRREP_FMT = '/scratch/hopan/cube/fourier_unmod/{}/{}.npy'
    PREFIX = '/scratch/hopan/cube'
elif os.path.exists('/project2/risi/'):
    NP_IRREP_FMT = '/project2/risi/cube/fourier_unmod/{}/{}.npy'
    PREFIX = '/project2/risi/cube'

def random_walk(max_len, env):
    states = [env.reset_solved()]
    actions = [i for i in range(env.actions)]
    for _ in range(max_len):
        action = random.choice(actions)
        ns, _, _, _ = env.step(action)
        states.append(ns)

    return states

def main(hparams):
    logfname = get_logdir(hparams['logdir'], hparams['savename'])
    if not os.path.exists(hparams['logdir']):
        os.makedirs(hparams['logdir'])
    savedir = get_logdir(hparams['logdir'], hparams['savename'])
    os.makedirs(savedir)
    sumdir = os.path.join(savedir, 'logs')
    os.makedirs(sumdir)
    logfile = os.path.join(savedir, 'log.txt')
    logger = SummaryWriter(sumdir)

    with open(os.path.join(savedir, 'args.json'), 'w') as f:
        json.dump(hparams, f, indent=4)

    log = get_logger(logfile)
    log.info('Saving log in {}'.format(savedir))
    log.info('hparams: {}'.format(hparams))

    torch.manual_seed(hparams['seed'])
    random.seed(hparams['seed'])

    alpha = eval(hparams['alpha'])
    parts = eval(hparams['parts'])
    log.info('alpha: {} | parts: {}'.format(alpha, parts))
    size = IRREP_SIZE[(alpha, parts)]
    pol_net = IrrepLinreg(size * size)
    targ_net = IrrepLinreg(size * size)

    if not hparams['init']:
        log.info('Loading fourier')
        pol_net.loadnp(NP_IRREP_FMT.format(str(alpha), str(parts)))
        targ_net.loadnp(NP_IRREP_FMT.format(str(alpha), str(parts)))
    else:
        log.info('Init model using mode: {}'.format(hparams['init']))
        pol_net.normal_init(hparams['noise_std'])
        targ_net.normal_init(hparams['noise_std'])

    if hparams['noise_std'] > 0:
        log.info('Adding noise: {}'.format(hparams['noise_std']))
        pol_net.add_gaussian_noise(0, hparams['noise_std'])
        targ_net.add_gaussian_noise(0, hparams['noise_std'])

    env = Cube2IrrepEnv(alpha, parts, solve_rew=hparams['solve_rew'])
    log.info('env solve reward: {}'.format(env.solve_rew))
    optimizer = torch.optim.SGD(pol_net.parameters(), lr=hparams['lr'], momentum=hparams['momentum'])
    memory = ReplayMemory(hparams['capacity'])
    niter = 0
    nupdates = 0
    seen_states = set()
    max_prop = 0

    log.info('Before any training:')
    val_avg, val_prop, val_time, solve_lens, solves_by_dist = val_model(pol_net, env, hparams)
    log.info('Validation | avg solve length: {:.4f} | solve prop: {:.4f} | time: {:.2f}s'.format(
        val_avg, val_prop, val_time
    ))
    log.info('{}'.format(solves_by_dist))
    prop_correct, dist_correct, dists = val_prop_correct(pol_net, env, hparams)
    log.info('Validation | Prop correct: {:.3f} | {} | Dists: {}'.format(
        prop_correct, str_fmt_dict(dist_correct), str_fmt_dict(dists)
    ))

    if len(solve_lens) > 0:
        log.info('Validation | LQ: {:.3f} | MQ: {:.3f} | UQ: {:.3f} | Max: {}'.format(
            np.percentile(solve_lens, 25),
            np.percentile(solve_lens, 50),
            np.percentile(solve_lens, 75),
            max(solve_lens)
        ))
    else:
        log.info('Validation | No solves!')

    for e in range(hparams['epochs']):
        states = random_walk(hparams['trainsteps'], env)
        seen_states.update(states)

        for state in states:
            if hparams['norandom']:
                action = get_action(env, pol_net, state)
            elif random.random() < explore_rate(e, hparams['epochs'] * hparams['explore_proportion'], hparams['minexp']):
                action = random.randint(0, env.action_space.n - 1)
            else:
                action = get_action(env, pol_net, state)

            ns, rew, done, _ = env.step(action, irrep=False)
            done = 1 if done == 1 else -1
            memory.push(state, action, ns, rew, done)
            niter += 1

            if (not hparams['noupdate']) and niter > 0 and niter % hparams['updateint'] == 0:
                sample = memory.sample(hparams['batch_size'])
                _loss = train_batch(env, pol_net, targ_net, sample, optimizer, hparams, logger, nupdates)
                logger.add_scalar('loss', _loss, nupdates)
                nupdates += 1

        if e % hparams['logint'] == 0:
            prop_correct, dist_correct, dists = val_prop_correct(pol_net, env, hparams)
            max_prop = max(max_prop, prop_correct)
            log.info('{:7} | Corr: {:.3f} | {} Updates: {} seen: {}'.format(
                e, prop_correct, str_fmt_dict(dist_correct), nupdates, len(seen_states)))
            try:
                if prop_correct > 0.80:
                    log.info('Saving model!')
                    torch.save(pol_net.state_dict(), os.path.join(savedir, 'model_{}_{}.pt'.format(e, prop_correct)))
            except:
                pdb.set_trace()


        #if e % hparams['updatetarget'] == 0 and e > 0:
        #    targ_net.load_state_dict(pol_net.state_dict())

    log.info('Total updates: {}'.format(nupdates))
    logger.export_scalars_to_json(os.path.join(savedir, 'summary.json'))
    logger.close()
    #torch.save(pol_net, os.path.join(savedir, 'model.pt'))
    check_memory()

    hparams['val_size'] = 10 * hparams['val_size']
    val_avg, val_prop, val_time, solve_lens, solves_by_dist = val_model(pol_net, env, hparams)
    log.info('Validation avg solve length: {:.4f} | solve prop: {:.4f} | time: {:.2f}s'.format(
        val_avg, val_prop, val_time
    ))
    log.info('{}'.format(solves_by_dist))
    log.info('Opt prop: {:.4f}'.format(max_prop))
    log.info('Saving in {}'.format(savedir))
    try:
        torch.save(pol_net.state_dict(), os.path.join(savedir, 'model.pt'))
    except:
        pdb.set_trace()

def val_prop_correct(pol_net, env, hparams):
    '''
    Generate some random smaple of states, see how many of the moves are correct
    '''
    dists = {}
    dists_correct = {}
    ncorrect = 0
    for _ in range(hparams['val_size']):
        pone = 1 if random.random() > 0.5 else 0
        states = random_walk(hparams['max_dist'] + pone, env)
        state = states[-1]
        #state = env.reset_fixed(max_dist=hparams['max_dist'])
        d = env.distance(state)
        corr = correct_move(env, pol_net, state)

        dists[d] = dists.get(d, 0) + 1
        dists_correct[d] = dists_correct.get(d, 0) + corr
        ncorrect += corr

    for d, val in dists_correct.items():
        dists_correct[d] /= dists[d]

    for d, val in dists.items():
        dists[d] = dists[d] / hparams['val_size']

    prop_correct = ncorrect / hparams['val_size']
    return prop_correct, dists_correct, dists

def str_fmt_dict(dic):
    dstr = ''
    max_key = max(dic.keys())
    #for i in range(0, max_key + 1):
    for i in range(0, 15):
        dstr += '{:2d}: {:+.2f} |'.format(i, dic.get(i, -1))

    return dstr

def val_model(pol_net, env, hparams):
    '''
    Returns proportion of validation cubes solved, avg solved len of those solved
    '''
    start = time.time()
    solved_lens = []
    solves_by_dist = {}
    all_dists = {}
    for e in tqdm(range(hparams['val_size'])):
        state = env.reset_fixed(max_dist=hparams['max_dist'])
        start_state = state
        true_dist = env.distance(start_state)
        all_dists[true_dist] = all_dists.get(true_dist, 0) + 1
        for i in range(hparams['maxsteps']):
            action = get_action(env, pol_net, state)
            ns, rew, done, _ = env.step(action, irrep=False)
            state = ns

            if done:
                solved_lens.append(i + 1)
                true_dist = env.distance(start_state)
                solves_by_dist[true_dist] = solves_by_dist.get(true_dist, 0) + 1
                break

    avg_solve = -1 if len(solved_lens) == 0 else np.mean(solved_lens)
    prop_solve = len(solved_lens) / hparams['val_size_fullsolve']
    for d in solves_by_dist.keys():
        solves_by_dist[d] /= all_dists[d]
    elapsed = time.time() - start

    return avg_solve, prop_solve, elapsed, solved_lens, solves_by_dist

if __name__ == '__main__':
    _prefix = 'local' if os.path.exists('/local/hopan/cube/pickles_sparse') else 'scratch'
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_dist', type=int, default=100)
    parser.add_argument('--val_size', type=int, default=200)
    parser.add_argument('--val_size_fullsolve', type=int, default=1000)
    parser.add_argument('--explore_proportion', type=float, default=0.2)
    parser.add_argument('--minexp', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--maxsteps', type=int, default=15)
    parser.add_argument('--trainsteps', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--capacity', type=int, default=10000)
    parser.add_argument('--updateint', type=int, default=50)
    parser.add_argument('--updatetarget', type=int, default=100)
    parser.add_argument('--usetarget', action='store_true', default=False)
    parser.add_argument('--logint', type=int, default=1000)
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--savename', type=str, default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--loadnp', type=str, default=NP_IRREP_FMT)
    parser.add_argument('--logdir', type=str, default=f'/{_prefix}/hopan/cube/logs/')
    parser.add_argument('--noupdate', action='store_true')
    parser.add_argument('--norandom', action='store_true')
    parser.add_argument('--init', type=str, default=None)
    parser.add_argument('--noise_std', type=float, default=0)
    parser.add_argument('--lossfunc', type=str, default='cmse')
    parser.add_argument('--solve_rew', type=int, default=1)
    parser.add_argument('--alpha', type=str, default='(2, 3, 3)')
    parser.add_argument('--parts', type=str, default='((2,), (2, 1), (2, 1))')
    args = parser.parse_args()
    hparams = vars(args)

    try:
        main(hparams)
    except KeyboardInterrupt:
        print('Keyboard escape!')
        check_memory()
