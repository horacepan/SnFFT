from GPUtil import showUtilization as gpu_usage
import pdb
import os
import time
import random
from itertools import permutations
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from perm_df import WreathDF
from rlmodels import MLP, MLPResModel
from complex_policy import ComplexLinear
from wreath_fourier import WreathPolicy, CubePolicy
from fourier_policy import FourierPolicyCG
from utility import ReplayBuffer, update_params, str_val_results, test_model, test_all_states, log_grad_norms, check_memory, wreath_onehot
from replay_buffer import ReplayBufferMini
from logger import get_logger
from tensorboardX import SummaryWriter
import sys
sys.path.append('../')
from complex_utils import cmse, cmse_real
import pyr_irreps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def full_benchmark(policy, perm_df, to_tensor, log):
    score, dists, stats = test_all_model(policy, 20, perm_df, to_tensor)
    log.info(f'Full Prop solves: {score:.4f} | stats: {str_val_results(stats)} | dists: {dists}')
    return {'score': score, 'dists': dists, 'stats': stats}

def try_load_weights(sumdir, policy, target):
    files = os.listdir(sumdir)
    models = [f for f in files if 'model_' in f]
    if len(models) == 0:
        return False, 0

    max_ep = max([int(f[6:f.index('.')]) for f in models])
    fname = os.path.join(sumdir, f'model_{max_ep}.pt')
    sd = torch.load(fname, map_location=device)

    policy.load_state_dict(sd)
    target.load_state_dict(sd)
    return True, max_ep

def main(args):
    sumdir = os.path.join(f'{args.savedir}', f'{args.sumdir}', f'{args.notes}/seed_{args.seed}')
    if not os.path.exists(sumdir) and args.savelog:
        os.makedirs(sumdir)
    if args.savelog:
        swr = SummaryWriter(sumdir)
        json.dump(args.__dict__, open(os.path.join(sumdir, 'args.json'), 'w'))
        logfile = os.path.join(sumdir, 'output.log')
        _cnt = 1
        while os.path.exists(logfile):
            logfile = os.path.join(sumdir, f'output{_cnt}.log')
            _cnt += 1
    else:
        logfile = args.logfile
    log = get_logger(logfile, stdout=not args.nostdout, tofile=args.savelog)

    log.info(f'Starting ... Saving logs in: {logfile} | summary writer: {sumdir}')
    log.info('Args: {}'.format(args))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    seen_states = set()

    if args.env == 'pyraminx':
        perm_df = WreathDF(args.fname, 8, cyc_size=2)
        ident = ((0, 0, 0,0, 0, 0), tuple(range(1, 7)))
        irreps = pyr_irreps.get_topk_irreps(args.num_pyr_irreps)
    elif args.env == 'cube2':
        log.info('Starting to load cube')
        perm_df = WreathDF(args.cubefn, 6, 3, args.tup_pkl)
        ident = ((0,) * 8, tuple(range(1, 9)))
        irreps = eval(args.cube_irreps)
        log.info('done loading cube')
    elif args.env == 'cubetest':
        log.info('Starting to load test cube')
        perm_df = WreathDF(args.cubefn, 3, 3, args.tup_pkl)
        ident = ((0,) * 8, tuple(range(1, 9)))
        irreps = eval(args.cube_irreps)

    if args.convert == 'onehot' and args.env == 'cube2':
        to_tensor = lambda g: wreath_onehot(g, 3)
    elif args.convert == 'onehot' and args.env == 'pyraminx':
        to_tensor = lambda g: wreath_onehot(g, 2)
    elif args.convert == 'onehot' and args.env == 'cubetest':
        to_tensor = lambda g: wreath_onehot(g, 3)
    elif args.convert == 'irrep':
        to_tensor = None
    else:
        raise Exception('Must pass in convert string')

    if args.model == 'linear':
        log.info(f'Policy using Irreps: {irreps}')
        policy = CubePolicy(irreps, std=args.std)
        target = CubePolicy(irreps, std=args.std, irrep_loaders=policy.irrep_loaders)
        to_tensor = lambda g: policy.to_tensor(g)
        log.info('Cube policy dim: {}'.format(policy.dim))
    elif args.model == 'dvn':
        log.info('Using MLP DVN')
        policy = MLP(to_tensor([ident]).numel(), args.nhid, 1, layers=args.layers, to_tensor=to_tensor, std=args.std)
        target = MLP(to_tensor([ident]).numel(), args.nhid, 1, layers=args.layers, to_tensor=to_tensor, std=args.std)
    elif args.model == 'res':
        log.info('Using Residual Network')
        policy = MLPResModel(to_tensor([ident]).numel(), args.resfc1, args.resfc2, 1, args.nres, std=args.std, to_tensor=to_tensor)
        target = MLPResModel(to_tensor([ident]).numel(), args.resfc1, args.resfc2, 1, args.nres, std=args.std, to_tensor=to_tensor)

    policy.to(device)
    target.to(device)
    start_epochs = 0
    loaded, start_epochs = try_load_weights(sumdir, policy, target)
    log.info(f'Loaded from old: {loaded}')

    if len(args.loadsaved) > 0:
        sd = torch.load(args.loadsaved, map_location=device)
        policy.load_state_dict(sd)
        target.load_state_dict(sd)
        log.info('Loaded model from: {}'.format(args.loadsaved))

    if args.use_mask and args.convert =='irrep':
        masks = {}
        if args.random_mask:
            log.info(f'Random Masks with p = {args.mask_p}')
            masks['wr'] = (torch.rand(policy.wr.data.shape) > (args.mask_p / 100)).float().to(device)
            masks['wi'] = (torch.rand(policy.wr.data.shape) > (args.mask_p / 100)).float().to(device)
        else:
            re_fn = os.path.join(args.maskdir, f'real_{args.mask_p}.th')
            im_fn = os.path.join(args.maskdir, f'im_{args.mask_p}.th')
            log.info(f'Loading Masks from {re_fn}:')
            masks['wr'] = torch.load(re_fn).to(device)
            masks['wi'] = torch.load(im_fn).to(device)

        policy.wr.data *= masks['wr']
        policy.wi.data *= masks['wi']

        target.wr.data *= masks['wr']
        target.wi.data *= masks['wi']

    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    if args.convert == 'onehot':
        replay = ReplayBuffer(to_tensor([ident]).numel(), args.capacity)
    elif args.convert == 'irrep':
        log.info('Creating mini replay')
        replay = ReplayBufferMini(args.capacity)

    max_benchmark = 0
    max_rand_benchmark = 0
    icnt = 0
    updates = 0
    bps = 0
    nactions = perm_df.num_nbrs
    dist_vals = {i: [] for i in range(0, 10)}
    update_pairs = {}
    exp_moves = 0
    pol_moves = 0
    pushes = {}
    npushes = 0

    stats_dict = {
        'save_epochs': [],
        'save_prop_corr': [],
        'save_solve_corr': [],
        'save_seen': [],
        'save_updates': []
    }
    check_memory()

    for e in range(start_epochs, start_epochs  + args.epochs + 1):
        states = perm_df.random_walk(args.eplen)
        #for state in states:
        for idx, state in enumerate(states):
            _nbrs = perm_df.nbrs(state)
            move = np.random.choice(nactions) # TODO
            next_state = _nbrs[move]

            done = 1 if (perm_df.is_done(state)) else 0
            reward = 1 if done else -1
            if args.convert == 'onehot':
                replay.push(to_tensor([state]), move, to_tensor([next_state]), reward, done, state, next_state, idx+1)
            else:
                replay.push(state, move, next_state, reward, done)
            d1 = perm_df.distance(state)
            d2 = perm_df.distance(next_state)
            pushes[(d1, d2, done)] = pushes.get((d1, d2, done), 0) + 1
            npushes += 1

            icnt += 1
            if icnt % args.update == 0 and icnt > 0:
                optim.zero_grad()
                if args.convert == 'onehot':
                    bs, ba, bns, br, bd, bs_tups, bns_tups, bidx = replay.sample(args.minibatch, device)
                else:
                    bs_tups, ba, bns_tups, br, bd = replay.sample(args.minibatch, device)

                seen_states.update(bs_tups)
                if args.convert == 'irrep':
                    bs_re, bs_im = to_tensor(bs_tups)
                    val_re, val_im = policy.forward_complex(bs_re, bs_im)

                    # backprop on a random sample!
                    if args.dorandom and (random.random() < args.minexp):
                        bns_re, bns_im = to_tensor(bns_tups)
                        opt_nbr_vals_re, opt_nbr_vals_im = target.forward_complex(bns_re, bns_im)
                    else:
                        bs_nbrs = [n for tup in bs_tups for n in perm_df.nbrs(tup)]
                        bs_nbrs_re, bs_nbrs_im = to_tensor(bs_nbrs)
                        nr, ni = target.forward_complex(bs_nbrs_re, bs_nbrs_im)
                        opt_nbr_vals_re, opt_idx = nr.reshape(-1, nactions).max(dim=1, keepdim=True)
                        opt_nbr_vals_im = ni.reshape(-1, nactions).gather(1, opt_idx)

                    if args.use_done:
                        loss = cmse(val_re, val_im,
                                    args.discount * opt_nbr_vals_re.detach() + br,
                                    args.discount * opt_nbr_vals_im.detach())
                    else:
                        loss = cmse(val_re, val_im,
                                    (1 - bd) * args.discount * opt_nbr_vals_re.detach() + br,
                                    (1 - bd) * args.discount * opt_nbr_vals_im.detach())

                    if args.use_mask and args.convert == 'irrep':
                        if hasattr(policy, 'wr'):
                            policy.wr.grad *= masks['wr']
                        if hasattr(policy, 'wi'):
                            policy.wi.grad *= masks['wi']
                elif args.convert == 'onehot' and (args.model == 'dvn' or args.model == 'res'):
                    if args.dorandom and (random.random() < args.minexp):
                        opt_nbr_vals = target.forward(bns).detach()
                    else:
                        bs_nbrs = [n for tup in bs_tups for n in perm_df.nbrs(tup)]
                        bs_nbrs_tens = to_tensor(bs_nbrs)
                        opt_nbr_vals, _ = target.forward(bs_nbrs_tens).detach().reshape(-1, nactions).max(dim=1, keepdim=True)

                    loss = F.mse_loss(policy.forward(bs),
                                      args.discount * (1 - bd) * opt_nbr_vals + br)
                if args.lognorms and bps % args.normiters == 0:
                    log_grad_norms(swr, policy, e)
                loss.backward()
                optim.step()
                bps += 1
                if args.savelog:
                    swr.add_scalar('loss', loss.item(), bps)

        if e % args.targetupdate == 0 and e > 0:
            update_params(target, policy)
            updates += 1

        if e % args.logiters == 0:
            exp_rate = 1.0
            distance_check = range(1, 13)
            cnt = 200
            benchmark, val_results = perm_df.prop_corr_by_dist(policy, to_tensor, distance_check, cnt)
            val_corr, distr, distr_stats = test_model(policy, 1000, 1000, 20, perm_df, to_tensor)
            max_benchmark = max(max_benchmark, benchmark)
            str_dict = str_val_results(val_results)

            if args.savelog:
                swr.add_scalar('prop_correct/overall', benchmark, e)
                swr.add_scalar('solve_prop', val_corr, e)
                for ii in distance_check:
                    rand_states = perm_df.random_states(ii, 100)
                    rand_tensors = to_tensor(rand_states)
                    vals = policy.forward(rand_tensors)
                    swr.add_scalar(f'values_median/states_{ii}', vals.median().item(), e)
                    swr.add_scalar(f'prop_correct/dist_{ii}', val_results[ii], e)

                stats_dict['save_epochs'].append(e)
                stats_dict['save_updates'].append(updates)
                stats_dict['save_prop_corr'].append(benchmark)
                stats_dict['save_solve_corr'].append(val_corr)
                stats_dict['save_seen'].append(len(seen_states))

            log.info(f'Epoch {e:5d} | Dist corr: {benchmark:.3f} | solves: {val_corr:.3f} | val: {str_dict}' + \
                     f'Updates: {updates}, bps: {bps} | seen: {len(seen_states)}')

        if e % args.saveiters == 0 and e > 0:
            torch.save(policy.state_dict(), os.path.join(sumdir, f'model_{e}.pt'))

    log.info('Max benchmark prop corr move attained: {:.4f}'.format(max_benchmark))
    log.info(f'Done training | log saved in: {logfile}')
    score, dists, stats = test_model(policy, 1000, 10000, 20, perm_df, to_tensor)
    bench_results.update(stats_dict)
    log.info('Final solve corr: {}'.format(bench_results['score']))

    if args.savelog:
        json.dump(bench_results, open(os.path.join(sumdir, 'stats.json'), 'w'))
        torch.save(policy.state_dict(), os.path.join(sumdir, 'model_final.pt'))
    print(f'Done with: {args.cube_irreps}')
    return bench_results

def get_args():
    _prefix = 'local' if os.path.exists('/local/hopan/irreps') else 'scratch'
    parser = argparse.ArgumentParser()
    # log params
    parser.add_argument('--nostdout', action='store_true', default=False)
    parser.add_argument('--savedir', type=str, default='/scratch/hopan/cube/irreps')
    parser.add_argument('--sumdir', type=str, default='test')
    parser.add_argument('--savelog', action='store_true', default=False)
    parser.add_argument('--logfile', type=str, default=f'./logs/rl/{time.time()}.log')
    parser.add_argument('--skipvalidate', action='store_true', default=False)
    parser.add_argument('--benchlog', type=int, default=50000)
    parser.add_argument('--saveiters', type=int, default=50000)
    parser.add_argument('--lognorms', action='store_true', default=False)
    parser.add_argument('--normiters', type=int, default=2000)
    parser.add_argument('--logiters', type=int, default=1000)

    # file related params
    parser.add_argument('--fname', type=str, default='/home/hopan/github/idastar/s8_dists_red.txt')
    parser.add_argument('--yorprefix', type=str, default=f'/{_prefix}/hopan/irreps/s_8/')
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--env', type=str, default='cube2')
    parser.add_argument('--pyrprefix', type=str, default=f'/{_prefix}/hopan/pyraminx/irreps/')
    parser.add_argument('--cubefn', type=str, default=f'/{_prefix}/hopan/cube/cube_sym_mod_tup.txt')
    parser.add_argument('--tup_pkl', type=str, default=f'/{_prefix}/hopan/cube/cube_sym_mod_tup.pkl')

    # model params
    parser.add_argument('--convert', type=str, default='irrep')
    parser.add_argument('--irreps', type=str, default='[]')
    parser.add_argument('--cube_irreps', type=str, default='[((2, 3, 3), ((2,), (1, 1, 1), (1, 1, 1)))]')
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--nhid', type=int, default=32)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--std', type=float, default=0.1)

    # hparams
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--capacity', type=int, default=100000)
    parser.add_argument('--eplen', type=int, default=15)
    parser.add_argument('--minexp', type=float, default=1.0)
    parser.add_argument('--dorandom', action='store_true', default=False)
    parser.add_argument('--update', type=int, default=25)
    parser.add_argument('--minibatch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--discount', type=float, default=1)
    parser.add_argument('--targetupdate', type=int, default=100)
    parser.add_argument('--use_done', action='store_true', default=False)

    # env specific
    parser.add_argument('--num_pyr_irreps', type=int, default=1)
    parser.add_argument('--br', type=int, default=-1)

    # zero mask
    parser.add_argument('--use_mask', action='store_true', default=False)
    parser.add_argument('--random_mask', action='store_true', default=False)
    parser.add_argument('--maskdir', type=str, default='/scratch/hopan/cube/masks/top1')
    parser.add_argument('--mask_p', type=int, default='50')

    # res params
    parser.add_argument('--resfc1', type=int, default='1024')
    parser.add_argument('--resfc2', type=int, default='256')
    parser.add_argument('--nres', type=int, default='1')

    parser.add_argument('--loadsaved', type=str, default='')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)
