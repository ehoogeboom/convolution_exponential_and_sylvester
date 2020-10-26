import torch


class BaseDistribution(torch.nn.Module):
    def __init__(self):
        super(BaseDistribution, self).__init__()

    def forward(self, x, context):
        return self.inference(x, context)

    def inference(self, x, context):
        raise NotImplementedError

    def sample(self, context, n_samples):
        raise NotImplementedError


class TemplateDistribution(BaseDistribution):
    def __init__(self, transformations, distribution):
        super(TemplateDistribution, self).__init__()

        self.transformations = torch.nn.ModuleList(transformations)
        self.distribution = distribution

    def inference(self, x, context):
        ldj = torch.zeros_like(x[:, 0, 0, 0])

        for layer in self.transformations:
            x, ldj = layer(x, ldj, context)

        log_pz = self.distribution.inference(x, context)
        log_px = log_pz + ldj
        return log_px

    def sample(self, context, n_samples):
        z, log_pz = self.distribution.sample(context, n_samples)

        ldj = torch.zeros_like(z[:, 0, 0, 0])

        for layer in reversed(self.transformations):
            z, ldj = layer.reverse(z, ldj, context)

        x = z
        log_px = log_pz - ldj

        return x, log_px
