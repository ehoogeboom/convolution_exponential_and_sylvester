import torch

from models.distributions import BaseDistribution
from models.distributions.utils import log_normal_standard, sample_normal_standard


class StandardGaussian(BaseDistribution):
    def __init__(self, input_size):
        super(StandardGaussian, self).__init__()

        assert len(input_size) == 3

        self.register_buffer('buffer', torch.ones(1))

        self.input_size = input_size

    def inference(self, x, context):
        log_px = torch.sum(log_normal_standard(x), dim=(1, 2, 3))
        return log_px

    def sample(self, context, n_samples):
        x = sample_normal_standard(n_samples, self.input_size)
        x = x.to(device=self.buffer.device)
        log_px = torch.sum(log_normal_standard(x), dim=(1, 2, 3))
        return x, log_px


if __name__ == '__main__':
    sample = sample_normal_standard(n_samples=32, size=(3, 32, 32))
    log_px = log_normal_standard(sample).sum(dim=(1, 2, 3))

    entropy_estimate = -torch.mean(log_px) / 32 / 32 / 3

    print(entropy_estimate)