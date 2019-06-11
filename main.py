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

def curriculum_dist(max_dist, curr_epoch, tot_epochs):
    '''
    Scale this so that initially you start 0 (or 1) away and gradually
    have further scrambles?
    Or maybe it shoud be probabalistic?! Equally likely in 0-k, where k gradually
    increases?
    '''
    if curr_epoch >= tot_epochs / 2:
        return max_dist
    else:
        # pick a random distance in 1 - max_dist * curr_epoch / (tot_epochs/2)
        return random.randint(1, 1 + int(max_dist * curr_epoch / (tot_epochs/2)))

def init_memory(memory, env):
    state = ns = 'GGGGBBBBRRRRMMMMWWWWYYYY'
    action = 0
    rew = env.solve_rew
    done = True
    for _ in range(memory.capacity):
        memory.push(state, action, ns, rew, done)

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
    log.debug('Saving in {}'.format(savedir))
    log.debug('hparams: {}'.format(hparams))

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
        pol_net.init(hparams['init'])
        targ_net.init(hparams['init'])
        log.info('Init model using mode: {}'.format(hparams['init']))

    if hparams['noise']:
        log.info('Adding noise: {}'.format(hparams['noise']))
        mu = torch.zeros(pol_net.wr.size())
        std = torch.zeros(pol_net.wr.size()) + hparams['noise']
        wr_noise = torch.normal(mu, std)
        wi_noise = torch.normal(mu, std)
        pol_net.wr.data.add_(wr_noise)
        pol_net.wi.data.add_(wi_noise)

        wr_noise = torch.normal(mu, std)
        wi_noise = torch.normal(mu, std)
        targ_net.wr.data.add_(wr_noise)
        targ_net.wi.data.add_(wi_noise)

    env = Cube2IrrepEnv(alpha, parts, solve_rew=hparams['solve_rew'])
    log.info('env solve reward: {}'.format(env.solve_rew))
    if hparams['opt'] == 'sgd':
        log.info('Using sgd')
        optimizer = torch.optim.SGD(pol_net.parameters(), lr=hparams['lr'], momentum=hparams['momentum'])
    elif hparams['opt'] == 'rms':
        log.info('Using rmsprop')
        optimizer = torch.optim.RMSprop(pol_net.parameters(), lr=hparams['lr'], momentum=hparams['momentum'])
    memory = ReplayMemory(hparams['capacity'])
    if hparams['meminit']:
        init_memory(memory, env)
    niter = 0
    nupdates = 0
    totsolved = 0
    solved_lens = []
    rewards = np.zeros(hparams['logint'])

    log.info('Before any training:')
    val_avg, val_prop, val_time, solve_lens = val_model(pol_net, env, hparams)
    log.info('Validation | avg solve length: {:.4f} | solve prop: {:.4f} | time: {:.2f}s'.format(
        val_avg, val_prop, val_time
    ))
    log.info('Validation | LQ: {:.3f} | MQ: {:.3f} | UQ: {:.3f} | Max: {}'.format(
        np.percentile(solve_lens, 25),
        np.percentile(solve_lens, 50),
        np.percentile(solve_lens, 75),
        max(solve_lens)
    ))
    scramble_lens = []
    for e in range(hparams['epochs']):
        if hparams['curric']:
            dist = curriculum_dist(hparams['max_dist'], e, hparams['epochs'])
        else:
            dist = hparams['max_dist']
        state = env.reset_fixed(max_dist=dist)
        epoch_rews = 0
        scramble_lens.append(dist)

        for i in range(hparams['maxsteps']):
            if hparams['norandom']:
                action = get_action(env, pol_net, state)
            elif random.random() < explore_rate(e, hparams['epochs'] * hparams['explore_proportion'], hparams['eps_min']):
                action = random.randint(0, env.action_space.n - 1)
            else:
                action = get_action(env, pol_net, state)

            ns, rew, done, _ = env.step(action, irrep=False)
            memory.push(state, action, ns, rew, done)
            epoch_rews += rew
            state = ns
            niter += 1

            if (not hparams['noupdate']) and niter > 0 and niter % hparams['update_int'] == 0:
                sample = memory.sample(hparams['batch_size'])
                _loss = update(env, pol_net, targ_net, sample, optimizer, hparams, logger, nupdates)
                logger.add_scalar('loss', _loss, nupdates)
                nupdates += 1

            if done:
                solved_lens.append(i + 1)
                totsolved += 1
                break

        rewards[e%len(rewards)] = epoch_rews
        logger.add_scalar('reward', epoch_rews, e)

        if e % hparams['logint'] == 0 and e > 0:
            val_avg, val_prop, val_time, _ = val_model(pol_net, env, hparams)
            logger.add_scalar('last_{}_solved'.format(hparams['logint']), len(solved_lens) / hparams['logint'], e)
            if len(solved_lens) > 0:
                logger.add_scalar('last_{}_solved_len'.format(hparams['logint']), np.mean(solved_lens), e)
            logger.add_scalar('val_solve_avg', val_avg, e)
            logger.add_scalar('val_prop', val_prop, e)
            log.info('{:7} | dist: {:4.1f} | avg rew: {:5.2f} | solve prop: {:5.3f}, len: {:5.2f} | exp: {:.2f} | ups {:7} | val avg {:.3f} prop {:.3f}'.format(
                e,
                np.mean(scramble_lens),
                np.mean(rewards),
                len(solved_lens) / hparams['logint'],
                0 if len(solved_lens) == 0 else np.mean(solved_lens),
                explore_rate(e, hparams['epochs'] * hparams['explore_proportion'], hparams['eps_min']),
                nupdates,
                val_avg,
                val_prop,
            ))
            solved_lens = []
            scramble_lens = []

        if e % hparams['updatetarget'] == 0 and e > 0:
            targ_net.load_state_dict(pol_net.state_dict())

    log.info('Total updates: {}'.format(nupdates))
    log.info('Total solved: {:8} | Prop solved: {:.4f}'.format(totsolved, totsolved / hparams['epochs']))
    logger.export_scalars_to_json(os.path.join(savedir, 'summary.json'))
    logger.close()
    torch.save(pol_net, os.path.join(savedir, 'model.pt'))
    check_memory()

    hparams['val_size'] = 10 * hparams['val_size']
    val_avg, val_prop, val_time, _ = val_model(pol_net, env, hparams)
    log.info('Validation avg solve length: {:.4f} | solve prop: {:.4f} | time: {:.2f}s'.format(
        val_avg, val_prop, val_time
    ))

def val_model(pol_net, env, hparams):
    '''
    Returns proportion of validation cubes solved, avg solved len of those solved
    '''
    start = time.time()
    solved_lens = []
    for e in range(hparams['val_size']):
        state = env.reset_fixed(max_dist=hparams['max_dist'])
        for i in range(hparams['maxsteps']):
            action = get_action(env, pol_net, state)
            ns, rew, done, _ = env.step(action, irrep=False)
            state = ns

            if done:
                solved_lens.append(i + 1)
                break

    avg_solve = -1 if len(solved_lens) == 0 else np.mean(solved_lens)
    prop_solve = len(solved_lens) / hparams['val_size']
    elapsed = time.time() - start

    return avg_solve, prop_solve, elapsed, solved_lens

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_dist', type=int, default=100)
    parser.add_argument('--val_size', type=int, default=1000)
    parser.add_argument('--explore_proportion', type=float, default=0.2)
    parser.add_argument('--eps_min', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--maxsteps', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--capacity', type=int, default=10000)
    parser.add_argument('--update_int', type=int, default=50)
    parser.add_argument('--updatetarget', type=int, default=100)
    parser.add_argument('--logint', type=int, default=1000)
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--savename', type=str, default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--loadnp', type=str, default=NP_IRREP_FMT)
    parser.add_argument('--logdir', type=str, default='/local/hopan/cube/logs/')
    parser.add_argument('--noupdate', action='store_true')
    parser.add_argument('--norandom', action='store_true')
    parser.add_argument('--init', type=str, default=None)
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--lossfunc', type=str, default='cmse')
    parser.add_argument('--curric', action='store_true')
    parser.add_argument('--meminit', action='store_true')
    parser.add_argument('--solve_rew', type=int, default=1)
    parser.add_argument('--alpha', type=str, default='(8, 0, 0)')
    parser.add_argument('--parts', type=str, default='((8,), (), ())')
    args = parser.parse_args()
    hparams = vars(args)

    try:
        main(hparams)
    except KeyboardInterrupt:
        print('Keyboard escape!')
        check_memory()
