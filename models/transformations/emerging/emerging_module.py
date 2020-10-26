import torch
import numpy as np
import torch.nn.functional as F
from models.transformations import BaseTransformation
from models.transformations.conv1x1 import Conv1x1
from models.transformations.emerging.masks import get_conv_square_ar_mask


class SquareAutoRegressiveConv2d(BaseTransformation):
    def __init__(self, n_channels):
        super(SquareAutoRegressiveConv2d, self).__init__()
        self.n_channels = n_channels
        kernel_size = [n_channels, n_channels, 2, 2]

        weight = torch.randn(kernel_size) / np.sqrt(np.prod(kernel_size))
        weight[torch.arange(n_channels), torch.arange(n_channels), -1, -1] += 1.
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(torch.zeros(n_channels))

        mask_np = get_conv_square_ar_mask(n_channels, n_channels, 2, 2)
        self.register_buffer('mask', torch.from_numpy(mask_np))

        from models.transformations.emerging.inverse_triang_conv import Inverse
        self.inverse_op = Inverse()

    def delta_ldj(self, x):
        log_abs_diag = torch.log(torch.abs(self.weight[torch.arange(
            self.n_channels), torch.arange(self.n_channels), -1, -1]))

        delta_ldj = torch.sum(log_abs_diag) * x.size(2) * x.size(3)

        return delta_ldj

    def forward(self, x, ldj, context):
        weight = self.weight * self.mask
        z = F.conv2d(x, weight, self.bias, stride=1, padding=1)

        # Slice off last dimensions.
        z = z[:, :, :-1, :-1]

        delta_ldj = self.delta_ldj(x)

        ldj = ldj + delta_ldj

        return z, ldj

    def reverse(self, z, ldj, context):
        weight = self.weight * self.mask

        delta_ldj = self.delta_ldj(z)

        bias = self.bias.view(1, self.n_channels, 1, 1)

        with torch.no_grad():
            x_np = self.inverse_op(
                z.detach().cpu().numpy(),
                weight.detach().cpu().numpy(),
                bias.detach().cpu().numpy())
            x = torch.from_numpy(x_np).to(z.device, z.dtype)

        ldj = ldj - delta_ldj

        return x, ldj
    

class Flip2d(BaseTransformation):
    def __init__(self):
        super(Flip2d, self).__init__()

    def forward(self, x, ldj, context, reverse=False):
        height = x.size(2)
        width = x.size(3)

        x = x[:, :, torch.arange(height-1, -1, -1)]
        x = x[:, :, :, torch.arange(width - 1, -1, -1)]

        return x, ldj

    def reverse(self, z, context, ldj):
        return self(z, context, ldj)


class Emerging(BaseTransformation):
    def __init__(self, n_channels):
        super(Emerging, self).__init__()

        self.transformations = torch.nn.ModuleList([
            Conv1x1(n_channels),
            SquareAutoRegressiveConv2d(n_channels),
            Flip2d(),
            SquareAutoRegressiveConv2d(n_channels),
            Flip2d(),
        ])

    def forward(self, x, logdet, context, reverse=False):
        if not reverse:
            for transform in self.transformations:
                x, logdet = transform(x, logdet, context)

        else:
            for transform in reversed(self.transformations):
                x, logdet = transform.reverse(x, logdet, context)

        return x, logdet

    def reverse(self, x, logdet, context):
        # For this particular reverse it is important that forward is called,
        # as it activates the pre-forward hook for spectral normalization.
        # This situation occurs when a flow is used to sample, for instance
        # in the case of variational dequantization.
        return self(x, logdet, context, reverse=True)


if __name__ == '__main__':
    x = torch.randn(1, 8, 4, 4)
    ldj = torch.zeros(8)
    layer = Emerging(8)

    z, _ = layer(x, ldj, None)

    x_recon, _ = layer.reverse(z, ldj, None)

    print(torch.mean((x - x_recon)**2))