import torch
import torch.nn.functional as F

from models.architectures.dense_in_residual import MultiscaleDenseInResidualNet, DenseInResidualNet
from models.transformations import BaseTransformation
from models.architectures.densenet import DenseNet

NN = None


class Coupling(BaseTransformation):
    def __init__(self, args, input_size, n_channels, n_context):
        super().__init__()
        self.n_channels = n_channels

        in_channels = n_channels // 2
        if n_context is not None:
            in_channels += n_context
            self.use_context = True
        else:
            self.use_context = False

        if args.n_scales == 1:
            self.nn = DenseInResidualNet(
                args, input_size,
                in_channels, n_channels,
                n_densenets=args.n_densenets,
                intermediate_channels=args.n_intermediate_channels,
                depth=args.densenet_depth,
                growth=args.densenet_growth,
                dropout_p=args.dropout_p,
                init_last_zero=True)
        else:
            self.nn = MultiscaleDenseInResidualNet(
                args, input_size, in_channels, n_channels,
                n_scales=args.n_scales,
                n_densenets=args.n_densenets,
                intermediate_channels=args.n_intermediate_channels,
                depth=args.densenet_depth,
                growth=args.densenet_growth,
                dropout_p=args.dropout_p,
                init_last_zero=True
            )

    def forward(self, x, ldj, context, reverse=False):
        assert self.use_context == (context is not None)
        x1 = x[:, :self.n_channels // 2, :, :]
        x2 = x[:, self.n_channels // 2:, :, :]

        if self.use_context:
            h = self.nn(torch.cat([x1, context], dim=1))
        else:
            h = self.nn(x1)
        h_s, t = h[:, ::2], h[:, 1::2]
        logs_range = 2.
        log_s = logs_range * torch.tanh(h_s / logs_range)

        if not reverse:
            z2 = x2 * torch.exp(log_s) + t
            ldj = ldj + torch.sum(log_s, dim=(1, 2, 3))
        else:
            z2 = (x2 - t) * torch.exp(-log_s)
            ldj = ldj - torch.sum(log_s, dim=(1, 2, 3))

        z = torch.cat([x1, z2], dim=1)

        return z, ldj

    def reverse(self, z, ldj, context):
        return self(z, ldj, context, reverse=True)


if __name__ == '__main__':
    x = torch.randn(128, 4, 8, 8)
    ldj = torch.zeros(128)
    coupling = Coupling(None, 4)

    z, _ = coupling(x, ldj, None)

    x_recon, _ = coupling(z, ldj, None, reverse=True)

    print(torch.mean((x - x_recon)**2))