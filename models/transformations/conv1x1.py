import torch
import torch.nn.functional as F
import numpy as np
from models.transformations import BaseTransformation


class Conv1x1(BaseTransformation):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

        w_np = np.random.randn(n_channels, n_channels)
        q_np = np.linalg.qr(w_np)[0]

        self.W = torch.nn.Parameter(torch.from_numpy(q_np.astype('float32')))

    def forward(self, x, ldj, context, reverse=False):
        _, _, H, W = x.size()

        w = self.W
        d_ldj = H * W * torch.slogdet(w)[1]

        if not reverse:
            w = w.view(self.n_channels, self.n_channels, 1, 1)

            z = F.conv2d(
                x, w, bias=None, stride=1, padding=0, dilation=1,
                groups=1)

            ldj += d_ldj
            return z, ldj
        else:
            w_inv = torch.inverse(w)
            w_inv = w_inv.view(self.n_channels, self.n_channels, 1, 1)

            z = F.conv2d(
                x, w_inv, bias=None, stride=1, padding=0, dilation=1,
                groups=1)

            ldj -= d_ldj

            return z, ldj

    def reverse(self, z, ldj, context):
        return self(z, ldj, context, reverse=True)


class Conv1x1Householder(BaseTransformation):
    def __init__(self, n_channels, n_reflections):
        super().__init__()
        self.n_channels = n_channels
        self.n_reflections = n_reflections

        v_np = np.random.randn(n_reflections, n_channels)

        self.V = torch.nn.Parameter(torch.from_numpy(v_np.astype('float32')))

    def contruct_Q(self):
        I = torch.eye(self.n_channels, dtype=self.V.dtype, device=self.V.device)
        Q = I

        for i in range(self.n_reflections):
            v = self.V[i].view(self.n_channels, 1)

            vvT = torch.matmul(v, v.t())
            vTv = torch.matmul(v.t(), v)
            Q = torch.matmul(Q, I - 2 * vvT / vTv)

        return Q

    def forward(self, x, ldj, context, reverse=False):
        _, _, H, W = x.size()

        Q = self.contruct_Q()

        if not reverse:
            Q = Q.view(self.n_channels, self.n_channels, 1, 1)

            z = F.conv2d(
                x, Q, bias=None, stride=1, padding=0, dilation=1,
                groups=1)

            return z, ldj
        else:
            Q_inv = Q.t()
            Q_inv = Q_inv.view(self.n_channels, self.n_channels, 1, 1)

            z = F.conv2d(
                x, Q_inv, bias=None, stride=1, padding=0, dilation=1,
                groups=1)

            return z, ldj

    def reverse(self, z, ldj, context):
        return self(z, ldj, context, reverse=True)


if __name__ == '__main__':
    x = torch.randn(128, 3, 8, 8)
    ldj = torch.zeros(128)
    conv1x1 = Conv1x1Householder(3, 3)

    z, _ = conv1x1(x, ldj, None)

    x_recon, _ = conv1x1(z, ldj, None, reverse=True)

    print(torch.mean((x - x_recon)**2))
