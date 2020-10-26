import torch


class BaseTransformation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, ldj, context):
        raise NotImplementedError

    def reverse(self, z, ldj, context):
        raise NotImplementedError


class ReverseTransformation(BaseTransformation):
    def __init__(self, transform):
        super().__init__()

        self.transform = transform

    def forward(self, x, ldj, context):
        return self.transform.reverse(x, ldj, context)

    def reverse(self, z, ldj, context):
        return self.transform(z, ldj, context)
