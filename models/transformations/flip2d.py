import torch
import torch.nn.functional as F
import numpy as np
from models.transformations import BaseTransformation


class Flip2d(BaseTransformation):
    def __init__(self):
        super().__init__()

    def forward(self, x, ldj, context, reverse=False):
        B, C, H, W = x.size()

        flip_h = torch.arange(H - 1, -1, -1)
        flip_w = torch.arange(W - 1, -1, -1)

        x = x[:, :, flip_h]
        x = x[:, :, :, flip_w]

        return x, ldj

    def reverse(self, z, ldj, context):
        return self(z, ldj, context, reverse=True)


if __name__ == '__main__':
    x = torch.randn(128, 3, 8, 8)
    ldj = torch.zeros(128)
    conv1x1 = Conv1x1Householder(3, 3)

    z, _ = conv1x1(x, ldj, None)

    x_recon, _ = conv1x1(z, ldj, None, reverse=True)

    print(torch.mean((x - x_recon)**2))
