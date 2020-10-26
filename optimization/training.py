from __future__ import print_function
import torch

import torchvision
from utils.distributions import log_sum_exp

import numpy as np
import os
import gc


def update_lipschitz(model):
    from models.transformations.iresnet.conv2d import InducedNormConv2d
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, InducedNormConv2d):
                m.compute_weight(update=True)


def train(epoch, train_loader, model, opt, args):
    model.train()
    train_bpd = 0.
    num_data = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, *args.input_size)
        data = data.to(args.device)
        elbo = model(data)

        bpd = torch.mean(-elbo) / np.prod(args.input_size) / np.log(2)

        bpd.backward()

        # Grad norm is only enabled for the residual flow experiments.
        # The simple reason is that all the other experiments were already done
        # at this point.
        if args.grad_norm_enabled:
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), 1.)
        else:
            grad_norm = 0

        opt.step()
        opt.zero_grad()

        bpd = bpd.item()
        train_bpd += bpd * len(data)
        num_data += len(data)

        if batch_idx % args.log_interval == 0:
            perc = 100. * batch_idx / len(train_loader)

            msg = 'Epoch: {} [{}/{} ({:2.0f}%)]\tbpd: {:8.6f}\t gradnorm: {:.2f}'.format(
                epoch, num_data, len(train_loader.sampler), perc, bpd,
                grad_norm)
            print(msg)

        if args.break_epoch:
            break

        del data
        torch.cuda.empty_cache()
        gc.collect()

    train_bpd = train_bpd / num_data

    print('Epoch: {:3d} Average bpd: {:.4f}'.format(epoch, train_bpd))

    return train_bpd


def evaluate(data_loader, model, args, iw_samples=1):
    model.eval()

    total_bpd = 0.
    number_items = 0
    with torch.no_grad():
        for data, _ in data_loader:
            if args.cuda:
                data = data.cuda()
            data = data.view(-1, *args.input_size)

            # Elbos contains all iw samples.
            elbos = torch.zeros((data.size(0), iw_samples))
            for iw_idx in range(iw_samples):
                elbo = model(data)
                elbos[:, iw_idx] = elbo

            # For iw-sampling, take logsumexp.
            batch_log_px = log_sum_exp(elbos, dim=1) \
                - np.log(iw_samples)

            batch_bpd = -torch.sum(batch_log_px).item() \
                / np.prod(args.input_size) / np.log(2)
            total_bpd += batch_bpd
            number_items += data.size(0)

            if args.break_epoch:
                break

    bpd = total_bpd / number_items
    return bpd


def plot_samples(model_sample, args, epoch, bpd):
    sample_d = 8

    with torch.no_grad():
        model_sample.eval()
        if not os.path.exists(args.snap_dir + 'samples/'):
            os.makedirs(args.snap_dir + 'samples/')

        bpd_string = 'bpd_{:.3f}'.format(bpd)
        fname = args.snap_dir + 'samples/' + str(epoch) + '_' + bpd_string + '_.png'
        x_sample = model_sample.sample(n_samples=sample_d*sample_d)
        print('min value {} max value {} mean value {}'.format(
            x_sample.min().item(),
            x_sample.max().item(),
            x_sample.mean().item()
        ))

        # x_sample = x_sample + x_sample.min()
        # x_sample = x_sample / x_sample.max()

        torchvision.utils.save_image(
            x_sample / 256., fname, nrow=sample_d, padding=2,
            normalize=False, range=None, scale_each=False, pad_value=0,
            format=None)
