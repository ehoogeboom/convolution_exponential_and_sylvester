import numpy as np
import torch
import torch.nn.functional as F
import models.transformations.convexp.functional as F_convexp

from models.transformations import BaseTransformation
from models.transformations.conv1x1 import Conv1x1Householder
from models.transformations.convexp.spectral import spectral_norm_conv
from models.architectures.layers import MaskedConv2d
from models.transformations.sylvester.masked_spectral import \
            masked_spectral_norm_conv


def triu_conv_mask(n_in, n_out, kernel_size, diagonal_zeros):
    assert n_out % n_in == 0 or n_in % n_out == 0, "%d - %d" % (n_in, n_out)

    # Build autoregressive mask
    l = (kernel_size[0] - 1) // 2
    m = (kernel_size[1] - 1) // 2
    mask = np.ones((n_out, n_in, kernel_size[0], kernel_size[1]),
                   dtype=np.float32)
    mask[:, :, :l, :] = 0
    mask[:, :, l, :m] = 0

    if n_out >= n_in:
        k = n_out // n_in
        for i in range(n_in):
            mask[i * k:(i + 1) * k, i + 1:, l, m] = 0
            if diagonal_zeros:
                mask[i * k:(i + 1) * k, i:i + 1, l, m] = 0
    else:
        k = n_in // n_out
        for i in range(n_out):
            mask[i:i + 1, (i + 1) * k:, l, m] = 0
            if diagonal_zeros:
                mask[i:i + 1, i * k:(i + 1) * k:, l, m] = 0

    return mask


def add_conv_diagonal(kernel, diagonal):
    C_out, C, K1, K2 = kernel.size()

    assert C_out == C
    assert diagonal.size(0) == C

    m1 = (K1 - 1) // 2
    m2 = (K2 - 1) // 2

    kernel[torch.arange(0, C), torch.arange(0, C), m1, m2] = diagonal

    return kernel


def get_conv_diagonal(kernel):
    C_out, C, K1, K2 = kernel.size()

    assert C_out == C

    m1 = (K1 - 1) // 2
    m2 = (K2 - 1) // 2

    diagonal = kernel[torch.arange(0, C), torch.arange(0, C), m1, m2]

    return diagonal


class AblationNoBasis(BaseTransformation):
    def __init__(self, args, input_size, n_channels):
        super().__init__()
        self.n_channels = n_channels

        kernel_size = [input_size[0], input_size[0], 3, 3]

        self.stride = (1, 1)
        self.padding = (1, 1)

        self.truncation_convexp = args.truncation_convexp

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
            self.nn[0], coeff=1.5, input_dim=input_size,
            name='weight', maskname='mask',
            n_power_iterations=1, eps=1e-12)

        masked_spectral_norm_conv(
            self.nn[2], coeff=1.5, input_dim=(n_channels, *input_size[1:]),
            name='weight', maskname='mask',
            n_power_iterations=1, eps=1e-12)

        masked_spectral_norm_conv(
            self.linear_t1, coeff=1.5,
            input_dim=(n_channels, *input_size[1:]),
            name='weight', maskname='mask',
            n_power_iterations=1, eps=1e-12)

        masked_spectral_norm_conv(
            self.linear_s1, coeff=1.5,
            input_dim=(n_channels, *input_size[1:]),
            name='weight', maskname='mask',
            n_power_iterations=1, eps=1e-12)

        masked_spectral_norm_conv(
            self.linear_t2, coeff=1.5,
            input_dim=(n_channels, *input_size[1:]),
            name='weight', maskname='mask',
            n_power_iterations=1, eps=1e-12)

        masked_spectral_norm_conv(
            self.linear_s2, coeff=1.5,
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
            f_ar_x, diagonal = self.f_AR(x)

            out = x + f_ar_x

            delta_ldj = torch.log(1. + diagonal).sum(dim=(1, 2, 3))

            ldj = ldj + delta_ldj

        else:
            v = x
            u = self.autoregressive_fixed_point_iteration(v)

            out = u

            _, n_ldj = self.forward(out, -ldj, None)

            ldj = n_ldj

        return out, ldj

    def autoregressive_fixed_point_iteration(self, y):
        x = y.clone()

        n_iterations = int(np.prod(y.size()[1:]))

        converged_at = None

        atol = 1e-6

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


class AblationWithoutGeneralization(BaseTransformation):
    def __init__(self, args, input_size, n_channels):
        super().__init__()
        kernel_size = [input_size[0], input_size[0], 3, 3]

        self.kernel = torch.nn.Parameter(
            torch.randn(kernel_size) / np.prod(kernel_size[1:]))

        self.stride = (1, 1)
        self.padding = (1, 1)

        self.conv1x1 = Conv1x1Householder(input_size[0], input_size[0])

        spectral_norm_conv(
            self, coeff=args.convexp_coeff, input_dim=input_size, name='kernel',
            n_power_iterations=1, eps=1e-12)

        # Masks needed for triangular r1 and r2.
        triu_mask = torch.from_numpy(triu_conv_mask(
            input_size[0], input_size[0], (3, 3), diagonal_zeros=True))
        print()
        self.register_buffer('triu_mask', triu_mask)

        self.d = torch.nn.Parameter(torch.zeros(kernel_size))
        with torch.no_grad():
            self.d.data.normal_(
                0, 1. / (n_channels * n_channels * 3 * 3))

        self.diag1 = torch.nn.Parameter(torch.randn(input_size[0]))
        self.diag2 = torch.nn.Parameter(torch.zeros(input_size[0]))

        self.b = torch.nn.Parameter(torch.zeros(input_size[0]))

    def h(self, x):
        return torch.tanh(x)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1. - self.h(x) ** 2

    def f_AR_nn(self, x):
        h = self.nn(x)

        t = self.linear_mean(h)
        h_s = self.linear_scale(h)

        scale = 0.5 * torch.tanh(h_s)

        return t, scale

    def f_AR(self, u):
        """
        In this case f_AR is a matrix multiplication:
        f_AR(x) = R1 h (R2 x + b)
        """
        d = self.d

        diag1 = torch.tanh(self.diag1)
        diag2 = torch.tanh(self.diag2)
        b = self.b
        r1 = d * self.triu_mask
        r1 = add_conv_diagonal(r1, diag1)
        r2 = d.permute(1, 0, 3, 2) * self.triu_mask
        r2 = add_conv_diagonal(r2, diag2)

        u = u + b.view(1, u.size(1), 1, 1)

        r1_u = F.conv2d(u, r1, stride=self.stride, padding=self.padding)

        h_r1_u = self.h(r1_u)

        r2_h_r1_u = \
            F.conv2d(h_r1_u, r2, stride=self.stride, padding=self.padding)

        out = r2_h_r1_u
        diagonal = diag1[None, :, None, None] * diag2[None, :, None, None] * self.der_h(r1_u)

        return out, diagonal

    def forward(self, x, ldj, context, reverse=False):
        assert context is None

        zeros_placeholder = torch.zeros_like(ldj)

        z, _ = self.conv1x1(x, zeros_placeholder, None)

        Mz = F_convexp.conv_exp(z, self.kernel, terms=10)

        f_ar_Mz, diagonal = self.f_AR(Mz)

        Minv_f_ar_Mz = F_convexp.inv_conv_exp(f_ar_Mz, self.kernel)

        Minv_f_ar_Mz, _ = self.conv1x1(
            Minv_f_ar_Mz, zeros_placeholder, None, reverse=True)

        out = x + Minv_f_ar_Mz

        delta_ldj = torch.log(1. + diagonal).sum(dim=(1, 2, 3))

        ldj = ldj + delta_ldj

        return out, ldj

    def reverse(self, z, ldj, context):
        print('Warning: Reverse is not implemented')
        return z, ldj


if __name__ == '__main__':
    x = torch.randn(128, 4, 8, 8)
    ldj = torch.zeros(128)

    class Args:
        convexp_coeff = 0.9

    args = Args()

    input_size = (4, 30, 30)
    layer = AblationWithoutGeneralization(args, input_size, 128)

    z, ldj_z = layer(x, ldj, None)

    x_recon, ldj_recon = layer(z, ldj, None, reverse=True)

    print(torch.mean((x - x_recon)**2))