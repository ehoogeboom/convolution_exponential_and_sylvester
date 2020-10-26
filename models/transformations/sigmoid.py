import torch
import torch.nn.functional as F
import numpy as np
from models.transformations import BaseTransformation


class Sigmoid(BaseTransformation):

    def __init__(self):
        super().__init__()

    def forward(self, x, ldj, context, reverse=False):
        z = torch.sigmoid(x)

        _, C, H, W = x.size()

        ldj = ldj + torch.sum(
            F.logsigmoid(x) + F.logsigmoid(-x), dim=(1, 2, 3))

        return z, ldj

    def reverse(self, z, ldj, context):
        raise NotImplementedError
