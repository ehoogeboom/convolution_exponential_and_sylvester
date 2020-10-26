# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import time
import numpy as np

import torch
import torch.utils.data

from optimization.training import evaluate, plot_samples
from utils.load_data import load_dataset
from os.path import join

parser = argparse.ArgumentParser(description='PyTorch Discrete Normalizing flows')


parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'imagenet32', 'imagenet64', 'svhn'],
                    metavar='DATASET',
                    help='Dataset choice.')


parser.add_argument('-bs', '--batch_size', type=int, default=1000, metavar='BATCH_SIZE',
                    help='input batch size for training (default: 100)')


parser.add_argument('--snap_dir', type=str, default='')


def main():
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    args.break_epoch = False

    snap_dir = args.snap_dir = join('snapshots', args.snap_dir) + '/'

    train_loader, val_loader, test_loader, args = load_dataset(args)

    final_model = torch.load(snap_dir + 'a.model', map_location='cpu')
    if args.cuda:
        final_model = final_model.cuda()

    # Just for timing at the moment.
    with torch.no_grad():
        final_model.eval()

        timing_results = []

        for i in range(100):
            torch.cuda.synchronize()
            start = time.time()
            x_sample = final_model.sample(n_samples=100)
            torch.cuda.synchronize()
            duration = time.time() - start
            timing_results.append(duration)

        print('Timings: ', timing_results)
        print('Mean time:', np.mean(timing_results))

        plot_samples(final_model, args, epoch=9999, bpd=0.0)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        final_model = torch.nn.DataParallel(final_model, dim=0)
    test_bpd = evaluate(test_loader, final_model, args)

    with open(snap_dir + 'log.txt', 'a') as ff:
        msg = 'FINAL \ttest negative elbo bpd {:.4f}'.format(
                test_bpd)
        print(msg)
        print(msg, file=ff)

    test_bpd = evaluate(test_loader, final_model, args, iw_samples=1000)

    with open(snap_dir + 'log.txt', 'a') as ff:
        msg = 'FINAL \ttest negative log_px bpd {:.4f}'.format(
            test_bpd)
        print(msg)
        print(msg, file=ff)


if __name__ == '__main__':
    main()