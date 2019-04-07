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

    alpha = (2, 3, 3)
    parts = ((2,), (1, 1, 1), (1, 1, 1))
    pol_net = IrrepLinreg(560 * 560)
    targ_net = IrrepLinreg(560 * 560)

    if hparams['loadnp'] and (not hparams['noinit']):
        pol_net.loadnp(hparams['loadnp'])
        targ_net.loadnp(hparams['loadnp'])

    env = Cube2IrrepEnv(alpha, parts)
    if hparams['opt'] == 'sgd':
        print('Using sgd')
        optimizer = torch.optim.SGD(pol_net.parameters(), lr=hparams['lr'], momentum=hparams['momentum'])
    elif hparams['opt'] == 'rms':
        print('Using rmsprop')
        optimizer = torch.optim.RMSprop(pol_net.parameters(), lr=hparams['lr'], momentum=hparams['momentum'])
    memory = ReplayMemory(hparams['capacity'])
    niter = 0
    nupdates = 0
    totsolved = 0
    solved_lens = []
    rewards = np.zeros(hparams['logint'])
    seen_states = set()

    for e in range(hparams['epochs']):
        state = env.reset_fixed(max_dist=hparams['max_dist'])
        epoch_rews = 0
        for i in range(hparams['maxsteps']):
            if hparams['norandom']:
                action = get_action(env, pol_net, state)
            elif random.random() < explore_rate(e, hparams['epochs'] * hparams['explore_proportion'], hparams['eps_min']):
                action = random.randint(0, env.action_space.n - 1)
            else:
                action = get_action(env, pol_net, state)

            seen_states.add(state)
            ns, rew, done, _ = env.step(action, irrep=False)
            memory.push(state, action, ns, rew, done)
            epoch_rews += rew
            state = ns
            niter += 1

            if (not hparams['noupdate']) and niter > 0 and niter % hparams['update_int'] == 0:
                sample = memory.sample(hparams['batch_size'])
                _loss = update(env, pol_net, targ_net, sample, optimizer, hparams, logger, nupdates)
                logger.add_scalar('loss', _loss, nupdates)
                #if nupdates % 100 == 0:
                #    logger.add_histogram('real_weights', pol_net.wr.detach().numpy(), nupdates)
                #    logger.add_histogram('imag_weights', pol_net.wi.detach().numpy(), nupdates)
                nupdates += 1

            if done:
                solved_lens.append(i + 1)
                totsolved += 1
                break

        rewards[e%len(rewards)] = epoch_rews
        logger.add_scalar('reward', epoch_rews, e)

        if e % hparams['logint'] == 0 and e > 0:
            val_avg, val_prop, val_time = val_model(pol_net, env, hparams)
            logger.add_scalar('last_{}_solved'.format(hparams['logint']), len(solved_lens) / hparams['logint'], e)
            logger.add_scalar('val_solve_avg', val_avg, e)
            logger.add_scalar('val_prop', val_avg, e)
            log.info('Epoch {:7} | avg rew: {:5.2f} | logint solve prop: {:5.2f} | explore: {:.2f} | nupdates {:7} | Val avg {:.3f}, prop {:.3f}, time {:.2f}s'.format(
                e,
                np.mean(rewards),
                -1 if len(solved_lens) == 0 else np.mean(solved_lens),
                explore_rate(e, hparams['epochs'] * hparams['explore_proportion'], hparams['eps_min']),
                nupdates,
                val_avg,
                val_prop,
                val_time
            ))
            solved_lens = []

        if e % hparams['updatetarget'] == 0 and e > 0:
            targ_net.load_state_dict(pol_net.state_dict())

    log.info('Total updates: {}'.format(nupdates))
    log.info('Total solved: {:8} | Prop solved: {:.4f}'.format(totsolved, totsolved / hparams['epochs']))
    logger.export_scalars_to_json(os.path.join(savedir, 'summary.json'))
    logger.close()
    torch.save(pol_net, os.path.join(savedir, 'model.pt'))
    check_memory()

def test_model(hparams):
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


    totsolved = 0
    alpha = (2, 3, 3)
    parts = ((2,), (1, 1, 1), (1, 1, 1))
    model = IrrepLinreg(560 * 560)
    env = Cube2IrrepEnv(alpha, parts)
    log.info('Done loading environment irreps')

    rewards = np.zeros(hparams['logint'])
    if hparams['loadnp'] and (not hparams['noinit']):
        log.info('Loading fourier matrix')
        model.loadnp(hparams['loadnp'])

    if hparams['noise']:
        log.info('Adding noise: {}'.format(hparams['noise']))
        mu = torch.zeros(model.wr.size())
        std = torch.zeros(model.wr.size()) + hparams['noise']
        wr_noise = torch.normal(mu, std)
        wi_noise = torch.normal(mu, std)
        model.wr.data.add_(wr_noise)
        model.wi.data.add_(wi_noise)

    solved_sum = 0
    for e in range(hparams['epochs'] + 1):
        state = env.reset_fixed(max_dist=hparams['max_dist'])
        epoch_rews = 0

        # allow twice as many moves as the max scramble distance
        for i in range(30):
            action = get_action(env, model, state)
            ns, rew, done, _ = env.step(action, irrep=False)
            state = ns
            epoch_rews += rew

            if done:
                solved_sum += (i + 1)
                totsolved += 1
                break

        rewards[e%len(rewards)] = epoch_rews
        if e % hparams['logint'] == 0 and e > 0:
            log.info('Epoch {:7} | avg traj rew: {:5.2f} | solved avg: {:5.2f} | tot solved: {:4} | last {} solved: {:4}'.format(
                e,
                np.mean(rewards),
                -1 if len(solved_sum) == 0 else np.mean(solved_sum),
                totsolved, #/ e,
                hparams['logint'],
                len(solved_sum),# / hparams['logint'],
            ))
            solved_sum = 0

    log.info('Total solved: {:8} | Prop solved: {:.4f}'.format(totsolved, totsolved / hparams['epochs']))

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
    return avg_solve, prop_solve, elapsed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_dist', type=int, default=100)
    parser.add_argument('--val_size', type=int, default=1000)
    parser.add_argument('--explore_proportion', type=float, default=0.2)
    parser.add_argument('--eps_min', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--maxsteps', type=int, default=30)
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
    parser.add_argument('--loadnp', type=str, default=NP_TOP_IRREP_LOC)
    parser.add_argument('--logdir', type=str, default='/local/hopan/cube/logs/')
    parser.add_argument('--noupdate', action='store_true')
    parser.add_argument('--norandom', action='store_true')
    parser.add_argument('--noinit', action='store_true')
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--lossfunc', type=str, default='cmse')
    parser.add_argument('--curric', action='store_true')
    args = parser.parse_args()
    hparams = vars(args)

    try:
        if hparams['test']:
            test_model(hparams)
        else:
            main(hparams)
    except KeyboardInterrupt:
        print('Keyboard escape!')
        check_memory()
