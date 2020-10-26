import torch
import torch.nn.functional as F
import numpy as np
from models.transformations import BaseTransformation


class PReLU(BaseTransformation):

    def __init__(self, n_channels):
        super().__init__()

        self.log_alpha = torch.nn.Parameter(torch.zeros(1, n_channels, 1, 1))

    def forward(self, x, ldj, context, reverse=False):
        negative_ldj = self.log_alpha * torch.ones_like(x)

        zeros = torch.zeros_like(x)

        delta_ldj = torch.where(x < 0, negative_ldj, zeros)

        if not reverse:
            alpha = torch.exp(self.log_alpha)

            z = torch.max(zeros, x) + torch.min(zeros, alpha * x)

            ldj = ldj + torch.sum(delta_ldj, dim=(1, 2, 3))

        else:
            inv_alpha = torch.exp(-self.log_alpha)

            z = torch.max(zeros, x) + torch.min(zeros, inv_alpha * x)

            ldj = ldj - torch.sum(delta_ldj, dim=(1, 2, 3))

        return z, ldj

    def reverse(self, z, ldj, context):
        return self(z, ldj, context, reverse=True)


if __name__ == '__main__':
    x = torch.randn(128, 4, 8, 8)
    ldj = torch.zeros(128)
    layer = PReLU(4)

    z, ldj = layer(x, ldj, None)

    x_recon, ldj = layer(z, ldj, None, reverse=True)

    print(torch.mean((x - x_recon)**2))
    print(torch.mean(ldj))