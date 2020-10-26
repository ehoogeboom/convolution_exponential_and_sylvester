import torch
import numpy as np
import torch.nn.functional as F
from models.transformations import BaseTransformation
from models.transformations.conv1x1 import Conv1x1


class WoodburyAxis(BaseTransformation):
    def __init__(self, n_dims, low_rank=24):
        super(WoodburyAxis, self).__init__()

        self.n_dims = n_dims
        self.low_rank = low_rank

        self.U = torch.nn.Parameter(
            torch.randn((n_dims, low_rank)) / n_dims)
        self.V = torch.nn.Parameter(
            torch.randn((low_rank, n_dims)) / n_dims)

    def forward(self, x, ldj, context, reverse=False):
        B, N, K = x.size()

        I_K = torch.eye(self.n_dims, device=x.device, dtype=x.dtype)
        I_low = torch.eye(self.low_rank, device=x.device, dtype=x.dtype)

        UV = torch.matmul(self.U, self.V)
        VU = torch.matmul(self.V, self.U)
        delta_ldj = N * torch.slogdet(I_low + VU)[1]

        if not reverse:
            z = torch.matmul(x, I_K + UV.transpose(-2, -1))
            ldj = ldj + delta_ldj
        else:
            I_VU_inv = torch.inverse(I_low + VU)
            inv = I_K - torch.matmul(self.U, I_VU_inv).matmul(self.V)

            z = torch.matmul(x, inv.transpose(-2, -1))
            ldj = ldj - delta_ldj

        return z, ldj

    def reverse(self, z, ldj, context):
        return self(z, ldj, context, reverse=True)


class Woodbury(BaseTransformation):
    def __init__(self, input_size):
        super(Woodbury, self).__init__()

        self.input_size = input_size
        self.n_channels = input_size[0]
        hw = input_size[1] * input_size[2]

        self.conv1x1 = Conv1x1(self.n_channels)
        self.woodbury_spatial = WoodburyAxis(n_dims=hw)

    def forward(self, x, logdet, context, reverse=False):
        B, C, H, W = x.size()
        if not reverse:
            # We use a standard 1x1 conv to make the comparison more fair.
            x, logdet = self.conv1x1(x, logdet, context)

            # Combine spatial dims.
            x = x.view(B, C, H*W)

            # Spatial woodbury.
            x, logdet = self.woodbury_spatial(x, logdet, context)

            # Inflate height and width:
            x = x.view(B, C, H, W)

        else:
            # Combine spatial dims.
            x = x.view(B, C, H * W)

            # Spatial woodbury.
            x, logdet = self.woodbury_spatial.reverse(x, logdet, context)

            # Inflate height and width:
            x = x.view(B, C, H, W)

            x, logdet = self.conv1x1.reverse(x, logdet, context)

        return x, logdet

    def reverse(self, x, logdet, context):
        # For this particular reverse it is important that forward is called,
        # as it activates the pre-forward hook for spectral normalization.
        # This situation occurs when a flow is used to sample, for instance
        # in the case of variational dequantization.
        return self(x, logdet, context, reverse=True)


if __name__ == '__main__':
    x = torch.randn(3, 31, 8, 8)
    ldj = torch.zeros(31)
    layer = Woodbury(input_size=(31, 8, 8))

    z, _ = layer(x, ldj, None)

    x_recon, _ = layer.reverse(z, ldj, None)

    print(torch.mean((x - x_recon) ** 2))