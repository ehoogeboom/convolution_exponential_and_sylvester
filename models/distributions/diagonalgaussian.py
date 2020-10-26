import torch

from models.distributions import BaseDistribution
from models.distributions.utils import log_normal_standard, sample_normal_standard


class DiagonalGaussian(BaseDistribution):
    def __init__(self, input_size, is_convolutional):
        super(DiagonalGaussian, self).__init__()

        assert len(input_size) == 3

        self.input_size = input_size

        param_size = (1, input_size[0], 1, 1) if is_convolutional else (1,) + (1, input_size[0], input_size[1], input_size[2])
        self.mu = torch.nn.Parameter(torch.zeros(param_size))
        self.log_sigma = torch.nn.Parameter(torch.zeros(param_size))

    def inference(self, x, context):
        z = (x - self.mu) * torch.exp(-self.log_sigma)

        log_pz = torch.sum(log_normal_standard(z), dim=(1, 2, 3))

        log_px = log_pz - torch.sum(self.log_sigma * torch.ones_like(x), dim=(1, 2, 3))

        return log_px

    def sample(self, context, n_samples):
        z = sample_normal_standard(n_samples, self.input_size)
        z = z.to(device=self.mu.device)
        log_pz = torch.sum(
            log_normal_standard(z), dim=(1, 2, 3))

        x = z * torch.exp(self.log_sigma) + self.mu

        log_px = log_pz - torch.sum(
            self.log_sigma * torch.ones_like(x), dim=(1, 2, 3))

        return x, log_px
