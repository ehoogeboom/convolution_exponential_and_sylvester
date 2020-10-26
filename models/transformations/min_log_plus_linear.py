import torch
import torch.nn.functional as F
import numpy as np
from models.transformations import BaseTransformation


class MinLogPlusLinear(BaseTransformation):

    def __init__(self):
        super().__init__()

    def forward(self, x, ldj, context, reverse=False):
        if not reverse:

            negative_part = -torch.log((-x + 1).abs() + 1e-10)
            positive_part = x

            negative_ldj = -torch.log(
                (x - 1).abs() + 1e-10
            )

            z = torch.where(x < 0, negative_part, positive_part)
            delta_ldj = torch.where(x < 0, negative_ldj, torch.zeros_like(x))

            ldj = ldj + torch.sum(delta_ldj, dim=(1, 2, 3))
        else:
            negative_part = -torch.exp(-x) + 1.
            positive_part = x
            z = torch.where(x < 0, negative_part, positive_part)

            negative_ldj = -torch.log(
                (z - 1).abs() + 1e-10
            )
            delta_ldj = torch.where(x < 0, negative_ldj, torch.zeros_like(x))

            ldj = ldj - torch.sum(delta_ldj, dim=(1, 2, 3))

        return z, ldj

    def reverse(self, z, ldj, context):
        return self(z, ldj, context, reverse=True)


if __name__ == '__main__':
    x = torch.randn(128, 4, 8, 8)
    ldj = torch.zeros(128)
    coupling = MinLogPlusLinear()

    z, ldj = coupling(x, ldj, None)

    x_recon, ldj = coupling(z, ldj, None, reverse=True)

    print(torch.mean((x - x_recon)**2))
    print(torch.mean(ldj))