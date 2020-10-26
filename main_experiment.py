# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
import math
import tensorboardX
import random
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


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.manual_seed is None:
    args.manual_seed = random.randint(1, 100000)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
np.random.seed(args.manual_seed)


kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}


def run(args, kwargs):
    # Would probably help, but experiments were done before.
    args.grad_norm_enabled = False

    print('\nMODEL SETTINGS: \n', args, '\n')
    print("Random Seed: ", args.manual_seed)

    if 'imagenet' in args.dataset and args.evaluate_interval_epochs > 5:
        args.evaluate_interval_epochs = 5

    # ==================================================================================================================
    # SNAPSHOTS
    # ==================================================================================================================
    args.model_signature = str(datetime.datetime.now())[0:16].replace(' ', '_')
    args.model_signature = args.model_signature.replace(':', '_')

    snapshots_path = args.out_dir
    snap_dir = snapshots_path + '/'

    snap_dir += args.exp_name + args.dataset + '_' + 'flows_' + str(args.n_subflows)

    snap_dir = snap_dir + '_' + args.model_signature + '/'

    args.snap_dir = snap_dir

    if not os.path.exists(snap_dir):
        os.makedirs(snap_dir)

    with open(snap_dir + 'log.txt', 'a') as ff:
        print('\nMODEL SETTINGS: \n', args, '\n', file=ff)

    writer = tensorboardX.SummaryWriter(logdir=snap_dir)

    # SAVING
    torch.save(args, snap_dir + '.config')

    # ==================================================================================================================
    # LOAD DATA
    # ==================================================================================================================
    train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)

    # ==================================================================================================================
    # SELECT MODEL
    # ==================================================================================================================
    # flow parameters and architecture choice are passed on to model through
    # args

    model_pv = Flow(
        args, args.input_size, n_levels=args.n_levels,
        n_subflows=args.n_subflows,
        use_splitprior=args.use_splitprior, n_context=None,
        normalize_translation=128.,
        normalize_scale=256.)

    if args.dequantize_distribution == 'uniform':
        model_hx = None
        model_qu_x = Uniform(args.input_size)

    elif args.dequantize_distribution == 'flow':
        model_hx = torch.nn.Sequential(
            Normalize_without_ldj(translation=128., scale=256.),
            DenseNet(
                args, input_size=(3, args.input_size[1], args.input_size[2]),
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

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(args.device)
    model_sample = model
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, dim=0)

    def lr_lambda(epoch):
        factor = min(1., (epoch + 1) / args.warmup) * np.power(args.lr_decay, epoch)
        print('Learning rate factor:', factor)
        return factor
    optimizer = optim.Adamax(model.parameters(), lr=args.learning_rate, eps=1.e-7)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    # Log the number of params.
    number_of_params = np.sum([np.prod(tensor.size()) for tensor in model.parameters()])
    fn = snap_dir + 'log.txt'
    with open(fn, 'a') as ff:
        msg = 'Number of Parameters: {}'.format(number_of_params)
        print(msg, file=ff)
        print(msg)

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

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        tr_bpd = train(epoch, train_loader, model, optimizer, args)
        scheduler.step()
        train_bpd.append(tr_bpd)
        writer.add_scalar('train bpd', train_bpd[-1], epoch)
        train_times.append(time.time()-t_start)
        print('One training epoch took %.2f seconds' % (time.time()-t_start))

        if epoch < 25 or epoch % args.evaluate_interval_epochs == 0:
            tr_bpd = evaluate(train_loader, model, args, iw_samples=1)
            v_bpd = evaluate(val_loader, model, args, iw_samples=1)

            # Logging message.
            with open(snap_dir + 'log.txt', 'a') as ff:
                msg = 'epoch {}\ttrain bpd {:.3f}\tval bpd {:.3f}\t'.format(
                        epoch, tr_bpd, v_bpd)
                print(msg, file=ff)

            # Sample and time sampling.
            torch.cuda.synchronize()
            start_sample = time.time()
            plot_samples(model_sample, args, epoch, v_bpd)
            torch.cuda.synchronize()
            print('Sampling took {} seconds'.format(time.time() - start_sample))

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

    test_bpd = evaluate(test_loader, final_model, args, iw_samples=1000)

    with open(snap_dir + 'log.txt', 'a') as ff:
        msg = 'epoch {}\ttest negative log_px bpd {:.4f}'.format(
            epoch, test_bpd)
        print(msg, file=ff)


if __name__ == "__main__":
    run(args, kwargs)
