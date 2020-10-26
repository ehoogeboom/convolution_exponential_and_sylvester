import torch
import numpy as np
from models.transformations import BaseTransformation


class Normalize(BaseTransformation):
    def __init__(self, translation, scale, learnable=False):
        super().__init__()

        assert scale > 0

        log_scale = np.log(scale).astype('float32')

        if learnable:
            self.translation = torch.nn.Parameter(torch.tensor([translation]))
            self.log_scale = torch.nn.Parameter(torch.tensor([log_scale]))

        else:
            self.register_buffer('translation', torch.tensor([translation]))
            self.register_buffer('log_scale', torch.tensor([log_scale]))

    def forward(self, x, ldj, context, reverse=False):
        _, C, H, W = x.size()

        d_ldj = -C * H * W * self.log_scale

        if not reverse:
            z = (x - self.translation) * torch.exp(-self.log_scale)
            ldj += d_ldj
            return z, ldj
        else:
            z = x * torch.exp(self.log_scale) + self.translation

            ldj -= d_ldj
            return z, ldj

    def reverse(self, z, ldj, context):
        return self(z, ldj, context, reverse=True)


class Normalize_without_ldj(torch.nn.Module):

    def __init__(self, translation, scale):
        super().__init__()
        self.translation = translation
        self.scale = scale

    def forward(self, x, reverse=False):
        if not reverse:
            z = (x - self.translation) / self.scale
            return z
        else:
            z = x * self.scale + self.translation
            return z

    def reverse(self, z):
        return self.forward(z, reverse=True)


class LogitTransform(BaseTransformation):
    """
    Taken from residual flows github.
    The proprocessing step used in Real NVP:
    y = sigmoid(x) - a / (1 - 2a)
    x = logit(a + (1 - 2a)*y)
    """

    def __init__(self, alpha):
        super(LogitTransform, self).__init__()
        self.alpha = alpha

    def forward(self, x, ldj, context, reverse=False):
        if reverse:
            return self.reverse(x, ldj, context)

        s = self.alpha + (1 - 2 * self.alpha) * x
        y = torch.log(s) - torch.log(1 - s)

        ldj = ldj + self._logdetgrad(x).view(x.size(0), -1).sum(1)

        return y, ldj

    def reverse(self, z, ldj, context):
        x = (torch.sigmoid(z) - self.alpha) / (1 - 2 * self.alpha)

        ldj = ldj - self._logdetgrad(x).view(x.size(0), -1).sum(1)

        return x, ldj

    def _logdetgrad(self, x):
        s = self.alpha + (1 - 2 * self.alpha) * x
        logdetgrad = -torch.log(s - s * s) + np.log(1 - 2 * self.alpha)
        return logdetgrad


class ActNorm(BaseTransformation):
    def __init__(self, n_channels):
        super().__init__()

        self.translation = torch.nn.Parameter(torch.zeros(n_channels))
        self.log_scale = torch.nn.Parameter(torch.zeros(n_channels))
        self.register_buffer('initialized', torch.tensor(0))

    def forward(self, x, ldj, context, reverse=False):
        _, C, H, W = x.size()

        if not self.initialized:
            print('initializing')
            with torch.no_grad():
                mean = torch.mean(x, dim=(0, 2, 3))
                log_stddev = torch.log(torch.std(x, dim=(0, 2, 3)) + 1e-8)

                self.translation.data.copy_(mean)
                self.log_scale.data.copy_(log_stddev)

                self.initialized.fill_(1)

        d_ldj = -H * W * self.log_scale.sum()

        translation = self.translation.view(1, C, 1, 1)
        log_scale = self.log_scale.view(1, C, 1, 1)

        if not reverse:
            z = (x - translation) * torch.exp(-log_scale)
            ldj += d_ldj
            return z, ldj
        else:
            z = x * torch.exp(log_scale) + translation

            ldj -= d_ldj
            return z, ldj

    def reverse(self, z, ldj, context):
        assert self.initialized
        return self(z, ldj, context, reverse=True)
