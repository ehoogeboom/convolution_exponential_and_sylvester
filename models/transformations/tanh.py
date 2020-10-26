import torch
import torch.nn.functional as F
import numpy as np
from models.transformations import BaseTransformation


class Tanh(BaseTransformation):

    def __init__(self):
        super().__init__()

    def forward(self, x, ldj, context, reverse=False):
        z = torch.tanh(x)

        _, C, H, W = x.size()

        log_derivative = - torch.pow(z, 2) + 1
        ldj = ldj + torch.sum(torch.log(log_derivative), dim=(1, 2, 3))

        return z, ldj

    def reverse(self, z, ldj, context):
        x = 0.5 * (torch.log(z + 1.) - torch.log(-z + 1))

        log_derivative = - torch.pow(z, 2) + 1
        ldj = ldj - torch.sum(torch.log(log_derivative), dim=(1, 2, 3))

        return x, ldj
