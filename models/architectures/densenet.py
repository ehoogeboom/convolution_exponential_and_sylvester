import torch


class DenseLayer(torch.nn.Module):
    def __init__(self, args, n_inputs, growth, dropout_p):
        super().__init__()

        nn = []

        nn.extend([
            torch.nn.Conv2d(
                n_inputs, n_inputs, kernel_size=1, stride=1,
                padding=0, bias=True),
            torch.nn.ReLU(inplace=True),
        ])

        if dropout_p > 0.:
            nn.append(torch.nn.Dropout(p=dropout_p))

        nn.extend([
            torch.nn.Conv2d(
                n_inputs, growth, kernel_size=3, stride=1,
                padding=1, bias=True),
            torch.nn.ReLU(inplace=True)
        ])

        self.nn = torch.nn.Sequential(*nn)

    def forward(self, x):
        h = self.nn(x)
        h = torch.cat([x, h], dim=1)
        return h


class GatedConv2d(torch.nn.Module):
    def __init__(self, args, n_inputs, n_outputs, kernel_size, padding):
        super(GatedConv2d, self).__init__()
        self.n_inputs = n_inputs

        self.conv = torch.nn.Conv2d(
            n_inputs, n_outputs * 3, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        h = self.conv(x)

        a, b, c = torch.chunk(h, chunks=3, dim=1)

        return a + b * torch.sigmoid(c)


class DenseNet(torch.nn.Module):
    def __init__(self, args, input_size, n_inputs, n_outputs, depth, growth,
                 dropout_p, init_last_zero=False):
        super(DenseNet, self).__init__()

        nn = []

        total_channels = n_inputs
        for i in range(depth):
            nn.append(
                DenseLayer(args, total_channels, growth, dropout_p)
            )
            total_channels = total_channels + growth

        if args.use_gated_conv:
            nn.append(
                GatedConv2d(
                    args, total_channels, n_outputs, kernel_size=1, padding=0)
            )
        else:
            nn.append(
                torch.nn.Conv2d(
                    total_channels, n_outputs, kernel_size=1, padding=0)
            )

        if init_last_zero:
            nn[-1].weight.zero_()
            if hasattr(nn[-1], 'bias'):
                nn[-1].bias.zero_()

        self.nn = torch.nn.Sequential(*nn)

    def forward(self, x):
        return self.nn(x)
