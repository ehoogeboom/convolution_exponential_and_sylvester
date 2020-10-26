import numpy as np
import torch
import torch.nn.functional as F
import models.transformations.convexp.functional as F_convexp

from models.architectures.dense_in_residual import MultiscaleDenseInResidualNet, DenseInResidualNet
from models.transformations import BaseTransformation
from models.architectures.densenet import DenseNet
from models.transformations.conv1x1 import Conv1x1Householder
from models.transformations.convexp.spectral import spectral_norm_conv
from models.architectures.layers import MaskedConv2d
from models.transformations.sylvester.masked_spectral import \
            masked_spectral_norm_conv


class GeneralizedSylvester(BaseTransformation):
    def __init__(self, args, input_size, n_channels):
        super().__init__()
        self.n_channels = n_channels

        kernel_size = [input_size[0], input_size[0], 3, 3]

        self.kernel = torch.nn.Parameter(
            torch.randn(kernel_size) / np.prod(kernel_size))

        self.stride = (1, 1)
        self.padding = (1, 1)

        self.truncation_convexp = args.truncation_convexp

        self.conv1x1 = Conv1x1Householder(input_size[0], input_size[0])

        spectral_norm_conv(
            self, coeff=args.convexp_coeff, input_dim=input_size, name='kernel',
            n_power_iterations=1, eps=1e-12)

        self.nn = torch.nn.Sequential(
            MaskedConv2d(
                input_size[0], n_channels, size_kernel=(3, 3),
                diagonal_zeros=False, bias=True),
            torch.nn.ReLU(inplace=True),
            MaskedConv2d(
                n_channels, n_channels, size_kernel=(1, 1),
                diagonal_zeros=False, bias=True),
            torch.nn.ReLU(inplace=True)
        )

        self.linear_t1 = MaskedConv2d(
            n_channels, input_size[0], size_kernel=(3, 3),
            diagonal_zeros=True, bias=True)
        self.linear_s1 = MaskedConv2d(
            n_channels, input_size[0], size_kernel=(3, 3),
            diagonal_zeros=True, bias=True)
        self.linear_t2 = MaskedConv2d(
            n_channels, input_size[0], size_kernel=(3, 3),
            diagonal_zeros=True, bias=True)
        self.linear_s2 = MaskedConv2d(
            n_channels, input_size[0], size_kernel=(3, 3),
            diagonal_zeros=True, bias=True)

        with torch.no_grad():
            self.linear_t1.weight.data.zero_()
            self.linear_s1.weight.data.zero_()

        masked_spectral_norm_conv(
            self.nn[0], coeff=args.lipschitz_sylvester, input_dim=input_size,
            name='weight', maskname='mask',
            n_power_iterations=1, eps=1e-12)

        masked_spectral_norm_conv(
            self.nn[2], coeff=args.lipschitz_sylvester, input_dim=(n_channels, *input_size[1:]),
            name='weight', maskname='mask',
            n_power_iterations=1, eps=1e-12)

        masked_spectral_norm_conv(
            self.linear_t1, coeff=args.lipschitz_sylvester,
            input_dim=(n_channels, *input_size[1:]),
            name='weight', maskname='mask',
            n_power_iterations=1, eps=1e-12)

        masked_spectral_norm_conv(
            self.linear_s1, coeff=args.lipschitz_sylvester,
            input_dim=(n_channels, *input_size[1:]),
            name='weight', maskname='mask',
            n_power_iterations=1, eps=1e-12)

        masked_spectral_norm_conv(
            self.linear_t2, coeff=args.lipschitz_sylvester,
            input_dim=(n_channels, *input_size[1:]),
            name='weight', maskname='mask',
            n_power_iterations=1, eps=1e-12)

        masked_spectral_norm_conv(
            self.linear_s2, coeff=args.lipschitz_sylvester,
            input_dim=(n_channels, *input_size[1:]),
            name='weight', maskname='mask',
            n_power_iterations=1, eps=1e-12)

    def der_tanh(self, a):
        return 1. - torch.tanh(a) ** 2

    def f_AR_nn(self, x):
        h = self.nn(x)

        h_t1 = self.linear_t1(h)
        h_s1 = self.linear_s1(h)
        h_t2 = self.linear_t2(h)
        h_s2 = self.linear_s2(h)

        t1 = h_t1
        s1 = torch.tanh(h_s1)

        t2 = h_t2
        s2 = .5 * torch.tanh(h_s2)

        return t1, s1, t2, s2

    def f_AR(self, x):
        t1, s1, t2, s2 = self.f_AR_nn(x)

        # alternative transform.
        s1_x = s1 * x + t1
        out = s2 * torch.tanh(s1_x) + t2
        diag = s2 * self.der_tanh(s1_x) * s1

        return out, diag

    def forward(self, x, ldj, context, reverse=False):
        assert context is None

        zeros_placeholder = torch.zeros_like(ldj)

        trucation = self.truncation_convexp if self.training else 0
        terms = 9 if self.training else 20

        if not reverse:
            z, _ = self.conv1x1(x, zeros_placeholder, None)

            Mz = F_convexp.conv_exp(
                z, self.kernel, terms=terms, dynamic_truncation=trucation)

            f_ar_Mz, diagonal = self.f_AR(Mz)

            Minv_f_ar_Mz = F_convexp.inv_conv_exp(
                f_ar_Mz, self.kernel, terms=terms, dynamic_truncation=trucation)

            Minv_f_ar_Mz, _ = self.conv1x1(
                Minv_f_ar_Mz, zeros_placeholder, None, reverse=True)

            out = x + Minv_f_ar_Mz

            delta_ldj = torch.log(1. + diagonal).sum(dim=(1, 2, 3))

            ldj = ldj + delta_ldj

        else:
            v, _ = self.conv1x1(x, torch.zeros_like(ldj), None)
            v = F_convexp.conv_exp(
                v, self.kernel, terms=terms, dynamic_truncation=trucation)

            u = self.autoregressive_fixed_point_iteration(v)

            u = F_convexp.inv_conv_exp(
                u, self.kernel, terms=terms, dynamic_truncation=trucation)

            out, _ = self.conv1x1(u, torch.zeros_like(ldj), None, reverse=True)

            _, n_ldj = self.forward(out, -ldj, None)

            ldj = n_ldj

        return out, ldj

    def autoregressive_fixed_point_iteration(self, y):
        x = y.clone()

        n_iterations = int(np.prod(y.size()[1:]))

        converged_at = None

        atol = 1e-4

        oldx = x
        with torch.no_grad():
            for iteration in range(1, n_iterations):
                y_, diagonal = self.f_AR(x)
                # Solve 1d case exact if conditioning checks out.
                newx = (y - y_)
                # y = x + f(x)

                # diff = y - (x * (1 + scale) + t)
                diff = newx - x

                if diff.abs().max().item() < atol:
                    print(diff.abs().max().item())
                    converged_at = iteration
                    break

                oldx = x  # oldx is only used for logging.
                x = newx

        if converged_at is None:
            message = 'Did not converge in {} iterations'.format(n_iterations)
        else:
            message = 'Converged at iteration {}'.format(iteration)
        print(message)

        diff = (x - oldx).abs()
        factor = (diff < atol).float().sum() / np.prod(x.size())
        meandiff = diff.mean().item()
        maxdiff = diff.max().item()

        print('{:.2f} dimensions converged, meandiff {} / maxdiff {}'.format(
            factor, meandiff, maxdiff))

        return x

    def reverse(self, z, ldj, context):
        return self(z, ldj, context, reverse=True)


if __name__ == '__main__':
    x = torch.randn(128, 4, 8, 8)
    ldj = torch.zeros(128)

    class Args:
        convexp_coeff = 0.9
        truncation_convexp = 0

    args = Args()

    input_size = (4, 30, 30)
    layer = GeneralizedSylvester(args, input_size, 128)
    layer.eval()

    z, ldj_z = layer(x, ldj, None)

    x_recon, ldj_recon = layer(z, ldj, None, reverse=True)

    print(torch.mean((x - x_recon)**2))
