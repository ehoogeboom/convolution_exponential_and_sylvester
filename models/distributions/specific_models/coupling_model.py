import torch

from models.distributions import BaseDistribution
from models.distributions.diagonalgaussian import DiagonalGaussian
from models.transformations import BaseTransformation
from models.transformations.squeeze import Squeeze
from models.transformations.normalize import Normalize, ActNorm
from models.distributions.splitprior import SplitPrior
from models.transformations.conv1x1 import Conv1x1


class PlainCoupling(BaseTransformation):
    def __init__(self, n_channels, n_intermediate_channels):
        super().__init__()
        self.n_channels = n_channels

        self.use_context = False

        self.nn = torch.nn.Sequential(
            torch.nn.Conv2d(
                n_channels // 2, n_intermediate_channels, kernel_size=3,
                padding=1, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                n_intermediate_channels, n_intermediate_channels, kernel_size=1,
                padding=0, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                n_intermediate_channels, n_channels, kernel_size=3, padding=1,
                bias=True),
        )

        with torch.no_grad():
            self.nn[-1].weight.data.zero_()
            self.nn[-1].bias.data.zero_()

    def forward(self, x, ldj, context, reverse=False):
        assert self.use_context == (context is not None)
        x1 = x[:, :self.n_channels // 2, :, :]
        x2 = x[:, self.n_channels // 2:, :, :]

        if self.use_context:
            h = self.nn(torch.cat([x1, context], dim=1))
        else:
            h = self.nn(x1)
        h_s, t = h[:, ::2], h[:, 1::2]
        log_s = 0.5 * torch.tanh(h_s)

        if not reverse:
            z2 = x2 * torch.exp(log_s) + t
            ldj = ldj + torch.sum(log_s, dim=(1, 2, 3))
        else:
            z2 = (x2 - t) * torch.exp(-log_s)
            ldj = ldj - torch.sum(log_s, dim=(1, 2, 3))

        z = torch.cat([x1, z2], dim=1)

        return z, ldj

    def reverse(self, z, ldj, context):
        return self(z, ldj, context, reverse=True)


class CouplingModel(BaseDistribution):
    def __init__(self, args, input_size, n_levels, n_subflows, use_splitprior,
                 normalize_translation, normalize_scale):
        super(CouplingModel, self).__init__()

        self.transformations = torch.nn.ModuleList()

        current_size = input_size

        if args.squeeze_first:
            self.transformations.append(
                Squeeze()
            )
            current_size = (
                current_size[0] * 4, current_size[1] // 2, current_size[2] // 2)

        for level in range(n_levels):
            for i in range(n_subflows):
                self.transformations.append(
                    ActNorm(current_size[0])
                )

                self.transformations.append(
                    Conv1x1(current_size[0])
                )

                self.transformations.append(
                    PlainCoupling(
                        current_size[0],
                        n_intermediate_channels=args.n_internal_channels
                    )
                )

            if level < n_levels - 1:
                if use_splitprior:
                    # do not factor out if the feature maps have not been
                    # squeezed ye.
                    if level >= 1 or not args.squeeze_first:
                        self.transformations.append(
                            SplitPrior(args, current_size, None))
                        # Update current_size
                        current_size = (
                            current_size[0] // 2,
                            current_size[1],
                            current_size[2]
                        )

                # Even if splitprior is not used, we always want to squeeze.
                self.transformations.append(
                    Squeeze()
                )
                current_size = (
                    current_size[0] * 4,
                    current_size[1] // 2,
                    current_size[2] // 2
                )

        self.base = DiagonalGaussian(current_size, is_convolutional=True)

    def inference(self, x, context):
        ldj = torch.zeros_like(x[:, 0, 0, 0])

        for layer in self.transformations:
            if isinstance(layer, SplitPrior):
                x, log_px2 = layer.inference(x, context)
                ldj = ldj + log_px2

            else:
                x, ldj = layer(x, ldj, context)

        log_pz = self.base.inference(x, context)
        log_px = log_pz + ldj
        return log_px

    def sample(self, context, n_samples):
        z, log_pz = self.base.sample(context, n_samples)

        ldj = torch.zeros_like(z[:, 0, 0, 0])

        for layer in reversed(self.transformations):
            if isinstance(layer, SplitPrior):
                z, log_px2 = layer.sample(z, context, n_samples)
                ldj = ldj - log_px2

            else:
                z, ldj = layer.reverse(z, ldj, context)

        x = z
        log_px = log_pz - ldj

        return x, log_px


if __name__ == '__main__':
    input_size = (3, 32, 32)
    x = torch.randn(127, *input_size)
    ldj = torch.zeros_like(x[:, 0, 0, 0])

    class Args:
        n_scales = 1
        n_densenets = 1
        n_intermediate_channels = 48
        densenet_depth = 4
        densenet_growth = 64
        dropout_p = 0.
        use_gated_conv = True
        use_convexp = False
        convexp_coeff = 0.9


    args = Args()

    flow = SylvesterModel(
        args, input_size, n_levels=2, n_subflows=1,
        use_splitprior=False, normalize_translation=128., normalize_scale=256.)

    opt = torch.optim.Adam(flow.parameters())

    x_in = x.clone()

    # for layer in flow.transformations:
    #     x, ldj = layer(x, ldj, context=None)
    #
    # loss = torch.sum(x**2)
    # loss.backward()
    # opt.step()

    x = x_in

    for layer in flow.transformations:
        x, ldj = layer(x, ldj, context=None)

    z = x

    temp = z.clone()

    for layer in reversed(flow.transformations):
        temp, ldj = layer.reverse(temp, ldj, context=None)

    x_recon = temp

    print(torch.max(torch.abs(x_in - x_recon)))
