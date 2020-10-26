import torch

from models.distributions import BaseDistribution
from models.distributions.diagonalgaussian import DiagonalGaussian
from models.transformations.coupling import Coupling
from models.transformations.squeeze import Squeeze
from models.transformations.conv1x1 import Conv1x1
from models.transformations.normalize import Normalize
from models.transformations.convexp.convexp_module import ConvExp
from models.distributions.splitprior import SplitPrior
from models.transformations import ReverseTransformation
from models.transformations.woodbury.woodbury_module import Woodbury


def get_mixing_transform(args, mixing, current_size):
    if mixing == '1x1':
        module = Conv1x1(current_size[0])
    elif mixing == 'convexp':
        module = ConvExp(args, current_size)
    elif mixing == 'emerging':
        from models.transformations.emerging.emerging_module import Emerging
        module = Emerging(current_size[0])
    elif mixing == 'woodbury':
        module = Woodbury(current_size)
    else:
        raise ValueError

    return module


class Flow(BaseDistribution):
    def __init__(self, args, input_size, n_levels, n_subflows, use_splitprior, n_context, normalize_translation, normalize_scale, parametrize_inverse=False):
        super(Flow, self).__init__()

        self.transformations = torch.nn.ModuleList()

        current_size = input_size

        self.transformations.append(
            Normalize(normalize_translation, normalize_scale)
        )

        self.transformations.append(
            Squeeze()
        )
        current_size = (
            current_size[0] * 4, current_size[1] // 2, current_size[2] // 2)

        # First start with 1x1 conv to reshuffle channels.
        module = get_mixing_transform(args, args.mixing, current_size)
        if parametrize_inverse:
            module = ReverseTransformation(module)
        self.transformations.append(module)

        for level in range(n_levels):
            for i in range(n_subflows):
                self.transformations.append(
                    Coupling(args, input_size, current_size[0], n_context)
                )
                # First start with 1x1 conv to reshuffle channels.
                module = get_mixing_transform(args, args.mixing, current_size)
                if parametrize_inverse:
                    module = ReverseTransformation(module)
                self.transformations.append(module)

            if level < n_levels - 1:
                if use_splitprior:
                    self.transformations.append(
                        SplitPrior(args, current_size, n_context))
                    # Update current_size
                    current_size = (
                        current_size[0] // 2,
                        current_size[1],
                        current_size[2]
                    )

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
    input_size = (7, 32, 32)
    x = torch.randn(127, *input_size)
    ldj = torch.zeros_like(x[:, 0, 0, 0])

    class Args:
        n_scales = 1
        n_densenets = 1
        n_intermediate_channels = 32
        densenet_depth = 4
        densenet_growth = 64
        dropout_p = 0.
        use_gated_conv = True
        use_convexp = False


    args = Args()

    flow = Flow(args, input_size, n_levels=2, n_subflows=10,
                use_splitprior=False, n_context=None,
                normalize_translation=128., normalize_scale=256.,
                parametrize_inverse=False)

    opt = torch.optim.Adam(flow.parameters())

    x_in = x.clone()

    for layer in flow.transformations:
        x, ldj = layer(x, ldj, context=None)

    loss = torch.sum(x**2)
    loss.backward()
    opt.step()

    x = x_in

    for layer in flow.transformations:
        x, ldj = layer(x, ldj, context=None)

    z = x

    temp = z.clone()

    for layer in reversed(flow.transformations):
        temp, ldj = layer.reverse(temp, ldj, context=None)

    x_recon = temp

    print(torch.max(torch.abs(x_in - x_recon)))
