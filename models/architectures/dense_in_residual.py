import torch

from models.architectures.densenet import DenseNet


class DenseInResidualNet(torch.nn.Module):
    def __init__(self, args, input_size, n_inputs, n_outputs, n_densenets,
                 intermediate_channels, depth, growth, dropout_p,
                 init_last_zero=False):
        super(DenseInResidualNet, self).__init__()

        self.first = torch.nn.Conv2d(
            n_inputs, intermediate_channels, kernel_size=1, padding=0)
        dense_nns = []

        self.n_densenets = n_densenets

        for i in range(n_densenets):
            dense_nns.append(
                DenseNet(args, input_size, intermediate_channels,
                         intermediate_channels,
                         depth, growth, dropout_p)
            )

        self.dense_nns = torch.nn.ModuleList(dense_nns)

        self.last = torch.nn.Conv2d(intermediate_channels, n_outputs,
                                    kernel_size=1, padding=0)

        with torch.no_grad():
            if init_last_zero:
                self.last.weight.zero_()
                if hasattr(self.last, 'bias'):
                    self.last.bias.zero_()

    def forward(self, x):
        h = self.first(x)

        for i in range(self.n_densenets):
            # Residual connection.
            h = h + self.dense_nns[i](h)

        return self.last(h)


class MultiscaleDenseInResidualNet(torch.nn.Module):
    def __init__(self, args, input_size, n_inputs, n_outputs, n_scales,
                 n_densenets, intermediate_channels, depth, growth, dropout_p,
                 init_last_zero):
        super(MultiscaleDenseInResidualNet, self).__init__()
        assert n_scales > 1
        self.n_scales = n_scales

        down = [
            DenseInResidualNet(
                args, input_size, n_inputs, intermediate_channels, n_densenets,
                intermediate_channels, depth, growth, dropout_p)]

        for i in range(n_scales - 1):
            down.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        intermediate_channels, intermediate_channels,
                        kernel_size=2, padding=0, stride=2),
                    DenseInResidualNet(args, input_size, intermediate_channels,
                                       intermediate_channels,
                                       n_densenets, intermediate_channels,
                                       depth, growth, dropout_p)
                )
            )

        up = []
        for i in range(n_scales - 1):
            up.append(
                torch.nn.Sequential(
                    DenseInResidualNet(args, input_size, intermediate_channels,
                                       intermediate_channels,
                                       n_densenets, intermediate_channels,
                                       depth, growth, dropout_p),
                    torch.nn.ConvTranspose2d(
                        intermediate_channels, intermediate_channels,
                        kernel_size=2, padding=0, stride=2),
                )
            )
        up.append(
            DenseInResidualNet(args, input_size, intermediate_channels,
                               n_outputs, n_densenets, intermediate_channels,
                               depth, growth, dropout_p,
                               init_last_zero=init_last_zero)
        )

        self.down = torch.nn.ModuleList(down)
        self.up = torch.nn.ModuleList(up)

    def forward(self, x):
        down_hs = []

        # First connection.
        down_hs.append(self.down[0](x))

        for i in range(1, self.n_scales):
            down_hs.append(
                self.down[i](down_hs[i - 1])
            )

        up_hs = [down_hs[-1]]

        for i in range(self.n_scales - 1):
            up_hs.append(
                self.up[i](up_hs[i]) + down_hs[self.n_scales - 2 - i]
            )

        # Last connection.
        final = self.up[-1](up_hs[-1])

        return final
