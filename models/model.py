import torch


class DiscreteLowerboundModel(torch.nn.Module):
    def __init__(self, model_pv, model_qu_x, model_context):
        super(DiscreteLowerboundModel, self).__init__()

        self.model_pv = model_pv
        self.model_qu_x = model_qu_x
        self.model_context = model_context

    def forward(self, x):
        x = x.float()
        if self.model_context is not None:
            context = self.model_context(x)
        else:
            context = None

        u, log_qu = self.model_qu_x.sample(context, n_samples=x.size(0))

        v = x + u

        log_pv = self.model_pv(v, context=None)

        return log_pv - log_qu

    def sample(self, n_samples):
        v, log_pv = self.model_pv.sample(context=None, n_samples=n_samples)

        return v


class DiscreteLowerboundAugmentedModel(torch.nn.Module):
    def __init__(self, model_pva, model_qu_x, model_qa_v,
                 model_context_x, model_context_v):
        super(DiscreteLowerboundAugmentedModel, self).__init__()

        self.model_pva = model_pva
        self.model_qu_x = model_qu_x
        self.model_qa_v = model_qa_v
        self.model_context_x = model_context_x
        self.model_context_v = model_context_v

        self.v_channels = None

    def forward(self, x):
        x = x.float()
        context_x = self.model_context_x(x)

        u, log_qu = self.model_qu_x.sample(context_x, n_samples=x.size(0))

        v = x + u

        if self.v_channels is None:
            self.v_channels = v.size(1)

        context_v = self.model_context_v(v)

        a, log_qa_v = self.model_qa_v.sample(context_v, n_samples=x.size(0))

        va = torch.cat([v, a], dim=1)

        log_pva = self.model_pva(va, context=None)

        return log_pva - log_qu - log_qa_v

    def sample(self, n_samples):
        v, log_pv = self.model_pva.sample(context=None, n_samples=n_samples)

        return v[:, :self.v_channels]


class PredictiveDiscreteLowerboundModel(torch.nn.Module):
    def __init__(self, model_pv_x, model_qu_y, model_context_y, model_context_x):
        super(PredictiveDiscreteLowerboundModel, self).__init__()

        self.model_pv_x = model_pv_x
        self.model_qu_y = model_qu_y
        self.model_context_y = model_context_y
        self.model_context_x = model_context_x

    def forward(self, x, y):
        if self.model_context_x is not None:
            context_x = self.model_context_x(x)
        else:
            context_x = None

        if self.model_context_y is not None:
            context_y = self.model_context_y(y)
        else:
            context_y = None

        u, log_pu = self.model_qu_x.sample(context_y, n_samples=x.size(0))

        v = y.float() + u

        log_pv_x = self.model_pv_x(v, context=context_x)

        return log_pv_x - log_pu

    def sample(self, x, n_samples):
        if self.model_context_x is not None:
            context_x = self.model_context_x(x)
        else:
            context_x = None

        v, log_pv = self.model_pv_x.sample(
            context=context_x, n_samples=n_samples)

        return v
