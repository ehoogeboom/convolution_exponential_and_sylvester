import torch
import torch.nn.functional as F

from models.transformations.coupling import Coupling
from models.distributions.standardgaussian import StandardGaussian


NN = None


class SplitPrior(torch.nn.Module):
    def __init__(self, args, input_size, n_context):
        super().__init__()
        assert len(input_size) == 3
        self.n_channels = n_channels = input_size[0]

        self.transform = Coupling(args, input_size, n_channels, n_context)

        self.base = StandardGaussian(
            (n_channels // 2, input_size[1], input_size[2]))

    def forward(self, x, context):
        return self.inference(x, context)

    def inference(self, x, context):
        ldj = torch.zeros_like(x[:, 0, 0, 0])

        x, ldj = self.transform(x, ldj, context)
        x1 = x[:, :self.n_channels // 2, :, :]
        x2 = x[:, self.n_channels // 2:, :, :]

        log_px2 = self.base.inference(x2, context)

        log_px2 = log_px2 + ldj

        z = x1

        return z, log_px2

    def sample(self, z, context, n_samples):
        x1 = z
        x2, log_px2 = self.base.sample(context, n_samples=n_samples)

        x = torch.cat([x1, x2], dim=1)
        ldj = torch.zeros_like(x[:, 0, 0, 0])

        x, ldj = self.transform.reverse(x, ldj, context)

        log_px = log_px2 - ldj

        return x, log_px
