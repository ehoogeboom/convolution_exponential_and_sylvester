import torch

from models.distributions import BaseDistribution
from models.distributions.diagonalgaussian import DiagonalGaussian
from models.transformations.flip2d import Flip2d
from models.transformations.squeeze import Squeeze
from models.transformations.normalize import Normalize, ActNorm
from models.distributions.splitprior import SplitPrior
from models.transformations.sylvester.sylvester import GeneralizedSylvester


class SylvesterModel(BaseDistribution):
    def __init__(self, args, input_size, n_levels, n_subflows, use_splitprior, normalize_translation, normalize_scale):
        super(SylvesterModel, self).__init__()

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
                    GeneralizedSylvester(
                        args, current_size, args.n_internal_channels)
                )

                self.transformations.append(
                    Flip2d()
                )

            if level < n_levels - 1:
                if use_splitprior:
                    # do not factor out if the feature maps have not been
                    # squeezed ye.
                    if level >= 1 or args.squeeze_first:
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
        n_internal_channels = 48
        densenet_depth = 4
        densenet_growth = 64
        dropout_p = 0.
        use_gated_conv = True
        use_convexp = False
        convexp_coeff = 0.9
        truncation_convexp = 0


    args = Args()

    flow = SylvesterModel(
        args, input_size, n_levels=2, n_subflows=10,
        use_splitprior=False, normalize_translation=0., normalize_scale=1.)

    with torch.no_grad():

        # Make sure that everything is spectral normalized.
        flow.train()
        flow(x, context=None)

        flow.eval()

        opt = torch.optim.Adam(flow.parameters())

        x_in = x.clone()
        #
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

        print('forward done')

        temp = z.clone()

        for layer in reversed(flow.transformations):
            temp, ldj = layer.reverse(temp, ldj, context=None)

        x_recon = temp

        print(torch.max(torch.abs(x_in - x_recon)))
