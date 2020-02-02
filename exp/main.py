import pdb
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from  yor_dataset import train_test_datasets
from torch.utils.data import DataLoader
from logger import get_logger

def compute_loss(net, test_x, test_y):
    pred_y = net(test_x)
    return F.mse_loss(pred_y, test_y).item()

def main(args):
    log = get_logger('./logs/test.log')
    log.info('Starting script with dataloader')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    irreps = [(3, 2, 2, 1), (4, 2, 2)]
    train_dataset, test_dataset = train_test_datasets(args.dfname, args.pklprefix, irreps, args.testratio)
    train_loader = DataLoader(train_dataset, batch_size=args.minibatch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.minibatch)
    log.info('Done loading data')

    size = 8036
    net = nn.Linear(size, 1, bias=True)
    net.weight.data.normal_(std=0.2)
    net.bias.data[0] = 5.328571428571428

    #optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    optim = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    train_x, train_y = train_dataset.tensors
    test_x, test_y = test_dataset.tensors

    for e in range(args.epochs):
        losses = []
        for k in range(160):
        #for bx, by in train_loader:
            optim.zero_grad()
            perm = np.random.choice(len(train_x), size=args.minibatch)
            bx, by = train_x[perm], train_y[perm]
            #bx = bx.to(device)
            #by = by.to(device)

            bpred = net(bx)
            loss = F.mse_loss(bpred, by)
            loss.backward()
            optim.step()

            losses.append(loss.item())
        test_loss = F.mse_loss(net(test_x), test_y)
        log.info(f'Done {e+1:3d} epochs | Epoch mse: {np.mean(losses):.4f} | ' + \
                 f'Test loss: {test_loss:.4f} | bpred mean: {bpred.mean().item():.2f}, std: {bpred.std().item():.2f}')

    log.info('Done!')
    pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--testratio', type=float, default=0.05)
    parser.add_argument('--dfname', type=str, default='/home/hopan/github/idastar/s8_dists_red.txt')
    parser.add_argument('--pklprefix', type=str, default='/local/hopan/irreps/s_8')
    parser.add_argument('--fhatprefix', type=str, default='/local/hopan/s8cube/fourier/')
    parser.add_argument('--logfile', type=str, default=f'/logs/{time.time()}.log')
    parser.add_argument('--minibatch', type=int, default=128)
    parser.add_argument('--maxiters', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--cgiters', type=int, default=250)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logiters', type=int, default=1000)
    parser.add_argument('--mode', type=str, default='torch')

    args = parser.parse_args()

    main(args)
