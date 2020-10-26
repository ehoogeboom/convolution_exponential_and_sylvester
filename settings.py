import argparse
import torch
import numpy as np
import random


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Discrete Normalizing flows')

    parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'imagenet32', 'imagenet64', 'svhn'],
                        metavar='DATASET',
                        help='Dataset choice.')

    parser.add_argument('-nc', '--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--manual_seed', type=int, help='manual seed, if not given resorts to random seed.')

    parser.add_argument('-li', '--log_interval', type=int, default=20, metavar='LOG_INTERVAL',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate_interval_epochs', type=int, default=25,
                        help='Evaluate per how many epochs')

    parser.add_argument('-od', '--out_dir', type=str, default='snapshots', metavar='OUT_DIR',
                        help='output directory for model snapshots etc.')

    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('-te', '--testing', action='store_true', dest='testing',
                    help='evaluate on test set after training')
    fp.add_argument('-va', '--validation', action='store_false', dest='testing',
                    help='only evaluate on validation set')
    parser.set_defaults(testing=True)

    # optimization settings
    parser.add_argument('-e', '--epochs', type=int, default=1000, metavar='EPOCHS',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('-es', '--early_stopping_epochs', type=int, default=300, metavar='EARLY_STOPPING',
                        help='number of early stopping epochs')

    parser.add_argument('-bs', '--batch_size', type=int, default=256, metavar='BATCH_SIZE',
                        help='input batch size for training (default: 100)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, metavar='LEARNING_RATE',
                        help='learning rate')
    parser.add_argument('--warmup', type=int, default=10,
                        help='number of warmup epochs')

    parser.add_argument('--data_augmentation_level', type=int, default=2,
                        help='data augmentation level')

    parser.add_argument('--variable_type', type=str, default='discrete',
                        help='variable type of data distribution: discrete'
                             'continuous',
                        choices=['discrete', 'continuous'])
    parser.add_argument('--dequantize_distribution', type=str, default='uniform',
                        choices=['uniform', 'flow'])
    parser.add_argument('--n_levels', type=int, default=2,
                        help='number of flows per level')
    parser.add_argument('--n_subflows', type=int, default=4,
                        help='number of flows per level')
    parser.add_argument('--n_context', type=int, default=32,
                        help='number of flows per level')
    parser.add_argument('--n_intermediate_channels', type=int, default=64,
                        help='number of flows per level')
    parser.add_argument('--densenet_depth', type=int, default=8,
                        help='number of flows per level')
    parser.add_argument('--densenet_growth', type=int, default=64,
                        help='number of flows per level')
    parser.add_argument('--n_densenets', type=int, default=1,
                        help='number of densenets per residual dense block')
    parser.add_argument('--n_scales', type=int, default=1,
                        help='number of scales in multiscale dense block')
    parser.add_argument('--use_gated_conv', action='store_true', default=False)
    parser.add_argument('--use_splitprior', action='store_true', default=False)
    parser.add_argument('--mixing', type=str, default='1x1',
                        choices=['1x1', 'convexp', 'emerging', 'woodbury'])
    parser.add_argument('--convexp_coeff', type=float, default=0.95,
                        help='Spectral norm coefficient.')

    parser.add_argument('--attention', type=str, default='none',
                        choices=['none', 'sparse', 'full'])

    parser.add_argument('--dropout_p', type=float, default=0.0,
                        help='Dropout probability')

    parser.add_argument('--dequantize_subflows', type=int, default=4,
                        help='number of flows per level')
    parser.add_argument('--dequantize_levels', type=int, default=1,
                        help='number of dequantize levels')

    parser.add_argument('--exp_name', type=str, default='',
                        help='Prefix for the experiment folder')

    parser.add_argument('--break_epoch', action='store_true', default=False,
                        help='Used to test all code by breaking after first'
                             'iteration.')

    # ---------------- SETTINGS CONCERNING NETWORKS -------------
    # ---------------- ----------------------------- -------------

    # ---------------- SETTINGS CONCERNING COUPLING LAYERS -------------
    # ---------------- ----------------------------------- -------------

    parser.add_argument('--lr_decay', default=0.995, type=float,
                        help='Learning rate')

    return parser
