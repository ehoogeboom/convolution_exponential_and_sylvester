import torch
from models.distributions import BaseDistribution


class Uniform(BaseDistribution):
    def __init__(self, input_size):
        super(Uniform, self).__init__()

        assert len(input_size) == 3

        self.input_size = input_size

        self.register_buffer('buffer', torch.zeros(1))

    def inference(self, x, context):

        more_than_1 = (x > 1).float()
        less_than_0 = (x < 0).float()

        log_px = torch.sum(
            -100 * more_than_1 * less_than_0, dim=(1, 2, 3))

        return log_px

    def sample(self, context, n_samples):
        x = torch.rand((n_samples, *self.input_size),
                       device=self.buffer.device)
        log_px = torch.zeros_like(x[:, 0, 0, 0])
        return x, log_px
