# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import time
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
import math
import random
import tensorboardX
from os.path import join

import os

import datetime

from models.model import DiscreteLowerboundModel
from optimization.training import train, evaluate, plot_samples
from utils.load_data import load_dataset

from models.distributions.flow import Flow
from models.distributions.uniform import Uniform
from models.architectures.densenet import DenseNet
from models.transformations.normalize import Normalize_without_ldj
from models.distributions import TemplateDistribution
from models.transformations.sigmoid import Sigmoid
from models.transformations import ReverseTransformation
from utils.distributions import log_sum_exp
from settings import get_parser

parser = get_parser()

parser.add_argument('--model_type', type=str, default='sylvester',
                    choices=[
                        'sylvester', 'residual_flow', 'coupling_flow',
                        'ablation_nobasis', 'ablation_nogeneralized'])

parser.add_argument('--truncation_convexp', type=float, default=0)

parser.add_argument('--n_internal_channels', type=int, default=528)
parser.add_argument('--weight_decay', type=float, default=0)

parser.add_argument('--restart_from_path', type=str, default=None)
parser.add_argument('--restart_from_epoch', type=int, default=None)

parser.add_argument('--squeeze_first', type=int, default=1)

parser.add_argument('--lipschitz_sylvester', type=float, default=1.5)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.manual_seed is None:
    args.manual_seed = random.randint(1, 100000)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
np.random.seed(args.manual_seed)


kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}


def setup(args):
    print('\nMODEL SETTINGS: \n', args, '\n')
    print("Random Seed: ", args.manual_seed)

    if 'imagenet' in args.dataset and args.evaluate_interval_epochs > 5:
        args.evaluate_interval_epochs = 5

    # ==================================================================================================================
    # SNAPSHOTS
    # ==================================================================================================================
    args.model_signature = str(datetime.datetime.now())[0:16].replace(' ', '_')
    args.model_signature = args.model_signature.replace(':', '_')

    if args.restart_from_path is not None:
        args.model_signature = args.model_signature + '_restart'
        args.restart_from_path = args.restart_from_path + '/'

    snapshots_path = args.out_dir
    snap_dir = snapshots_path + '/'

    snap_dir += args.exp_name + args.dataset + '_' + 'flows_' + str(
        args.n_subflows)

    snap_dir = snap_dir + '_' + args.model_signature + '/'

    args.snap_dir = snap_dir

    if not os.path.exists(snap_dir):
        os.makedirs(snap_dir)

    with open(snap_dir + 'log.txt', 'a') as ff:
        print('\nMODEL SETTINGS: \n', args, '\n', file=ff)

    # SAVING
    try:
        torch.save(args, snap_dir + '.config')
    except:
        print('Error during saving.')

    # ==================================================================================================================
    # SELECT MODEL
    # ==================================================================================================================
    # flow parameters and architecture choice are passed on to model through
    # args

    if args.restart_from_path is None:

        from models.distributions.specific_models.sylvester_model import \
            SylvesterModel
        from models.distributions.specific_models.residualflow_model import \
            ResidualFlowModel
        from models.distributions.specific_models.coupling_model import \
            CouplingModel

        if args.model_type == 'sylvester':
            model_pv = SylvesterModel(
                args, args.input_size, n_levels=args.n_levels,
                n_subflows=args.n_subflows,
                use_splitprior=args.use_splitprior,
                normalize_translation=128.,
                normalize_scale=256.)
        elif args.model_type == 'residual_flow':
            model_pv = ResidualFlowModel(
                args, args.input_size, n_levels=args.n_levels,
                n_subflows=args.n_subflows,
                use_splitprior=args.use_splitprior,
                normalize_translation=128.,
                normalize_scale=256.)
        elif args.model_type == 'coupling_flow':
            model_pv = CouplingModel(
                args, args.input_size, n_levels=args.n_levels,
                n_subflows=args.n_subflows,
                use_splitprior=args.use_splitprior,
                normalize_translation=128.,
                normalize_scale=256.)
        elif 'ablation' in args.model_type:
            from models.distributions.specific_models.ablations_model import \
                Ablation
            model_pv = Ablation(
                args, args.input_size, n_levels=args.n_levels,
                n_subflows=args.n_subflows,
                use_splitprior=args.use_splitprior,
                normalize_translation=128.,
                normalize_scale=256.)
        else:
            raise ValueError(args.model_type)

        if args.dequantize_distribution == 'uniform':
            model_hx = None
            model_qu_x = Uniform(args.input_size)

        elif args.dequantize_distribution == 'flow':
            model_hx = torch.nn.Sequential(
                Normalize_without_ldj(translation=128., scale=256.),
                DenseNet(
                    args,
                    input_size=(3, args.input_size[1], args.input_size[2]),
                    n_inputs=3, n_outputs=args.n_context, depth=4, growth=32,
                    dropout_p=args.dropout_p),
                torch.nn.Conv2d(args.n_context, args.n_context,
                                kernel_size=2, stride=2, padding=0),
                DenseNet(
                    args, n_inputs=args.n_context,
                    input_size=(3, args.input_size[1], args.input_size[2]),
                    n_outputs=args.n_context,
                    depth=4, growth=32, dropout_p=args.dropout_p),
            )
            model_qu_x = TemplateDistribution(
                transformations=[ReverseTransformation(Sigmoid())],
                distribution=Flow(
                    args, args.input_size, n_levels=args.dequantize_levels,
                    n_subflows=args.dequantize_subflows,
                    n_context=args.n_context,
                    use_splitprior=False,
                    normalize_translation=0.,
                    normalize_scale=1.,
                    parametrize_inverse=True))
        else:
            raise ValueError

        model = DiscreteLowerboundModel(model_pv, model_qu_x, model_hx)

        print(model)

    else:
        assert args.restart_from_epoch is not None
        print('Loading from checkpoint...')
        model = torch.load(args.restart_from_path + 'a.model')

    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(args.device)
    model_sample = model
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, dim=0)

    def lr_lambda(epoch):
        factor = np.power(min(1., (epoch + 1) / args.warmup), 2) \
                 * np.power(args.lr_decay, epoch)
        print('Learning rate factor: {:.4f}'.format(factor))
        return factor

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.99), weight_decay=args.weight_decay, eps=1.e-7)

    if args.restart_from_path is None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda,
                                                      last_epoch=-1)
    else:
        # if loaded directly links to params are broken.
        optimizer_from_file = torch.load(args.restart_from_path + 'a.optimizer')
        optimizer.load_state_dict(optimizer_from_file.state_dict())
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=args.restart_from_epoch-1)

    # Log the number of params.
    number_of_params = np.sum(
        [np.prod(tensor.size()) for tensor in model.parameters()])
    fn = snap_dir + 'log.txt'
    with open(fn, 'a') as ff:
        msg = 'Number of Parameters: {}. Note: for masked convolutions this is an overestimate.'.format(
            number_of_params)
        print(msg, file=ff)
        print(msg)

    return model, model_sample, optimizer, scheduler


def run(args, kwargs):

    # Only for the residual networks (resflow/sylvester) comparison.
    args.grad_norm_enabled = True

    # ==================================================================================================================
    # LOAD DATA
    # ==================================================================================================================
    train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)

    model, model_sample, optimizer, scheduler = setup(args)

    writer = tensorboardX.SummaryWriter(logdir=args.snap_dir)

    snap_dir = args.snap_dir
    # ==================================================================================================================
    # TRAINING
    # ==================================================================================================================
    train_bpd = []
    val_bpd = []

    # for early stopping
    best_val_bpd = np.inf
    best_train_bpd = np.inf
    epoch = 0

    train_times = []

    model.eval()
    model.train()

    starting_epoch = 1
    if args.restart_from_epoch is not None:
        starting_epoch = args.restart_from_epoch

    for epoch in range(starting_epoch, args.epochs + 1):
        t_start = time.time()
        tr_bpd = train(epoch, train_loader, model, optimizer, args)
        scheduler.step()
        train_bpd.append(tr_bpd)
        writer.add_scalar('train bpd', train_bpd[-1], epoch)
        train_times.append(time.time()-t_start)
        print('One training epoch took %.2f seconds' % (time.time()-t_start))

        if epoch in [1, 5, 10] or epoch % args.evaluate_interval_epochs == 0:
            tr_bpd = evaluate(train_loader, model, args, iw_samples=1)
            v_bpd = evaluate(val_loader, model, args, iw_samples=1)

            # Logging message.
            with open(snap_dir + 'log.txt', 'a') as ff:
                msg = 'epoch {}\ttrain bpd {:.3f}\tval bpd {:.3f}\t'.format(
                        epoch, tr_bpd, v_bpd)
                print(msg, file=ff)

            plot_samples(model_sample, args, epoch, v_bpd)

            val_bpd.append(v_bpd)
            writer.add_scalar('val bpd', v_bpd, epoch)

            # Model save based on val performance
            if v_bpd < best_val_bpd:
                best_train_bpd = tr_bpd
                best_val_bpd = v_bpd

                try:
                    if hasattr(model, 'module'):
                        torch.save(model.module, snap_dir + 'a.model')
                    else:
                        torch.save(model, snap_dir + 'a.model')
                    torch.save(optimizer, snap_dir + 'a.optimizer')
                    print('->model saved<-')
                except:
                    print('Saving was unsuccessful.')

            print('(BEST: train bpd {:.4f}, val bpd {:.4f})\n'.format(
                best_train_bpd, best_val_bpd))

            if math.isnan(v_bpd):
                raise ValueError('NaN encountered!')

    # training time per epoch
    train_times = np.array(train_times)
    mean_train_time = np.mean(train_times)
    std_train_time = np.std(train_times, ddof=1)
    print('Average train time per epoch: {:.2f} +/- {:.2f}'.format(
        mean_train_time, std_train_time))

    # ========================================================================
    # EVALUATION
    # ========================================================================
    final_model = torch.load(snap_dir + 'a.model')

    test_bpd = evaluate(test_loader, final_model, args)

    with open(snap_dir + 'log.txt', 'a') as ff:
        msg = 'epoch {}\ttest negative elbo bpd {:.4f}'.format(
                epoch, test_bpd)
        print(msg, file=ff)

    if 'residual' in args.model_type:
        print('Importance weighted eval needs exact determinants.')

    else:
        test_bpd = evaluate(test_loader, final_model, args, iw_samples=1000)

        with open(snap_dir + 'log.txt', 'a') as ff:
            msg = 'epoch {}\ttest negative log_px bpd {:.4f}'.format(
                epoch, test_bpd)
            print(msg, file=ff)


if __name__ == "__main__":
    run(args, kwargs)
