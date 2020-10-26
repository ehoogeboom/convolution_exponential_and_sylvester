import torch
import torch.nn.functional as F
import math
from torch._six import container_abcs
from itertools import repeat


def operator_norm_settings(domain, codomain):
    if domain == 1 and codomain == 1:
        # maximum l1-norm of column
        max_across_input_dims = True
        norm_type = 1
    elif domain == 1 and codomain == 2:
        # maximum l2-norm of column
        max_across_input_dims = True
        norm_type = 2
    elif domain == 1 and codomain == float("inf"):
        # maximum l-inf norm of column
        max_across_input_dims = True
        norm_type = float("inf")
    elif domain == 2 and codomain == float("inf"):
        # maximum l2-norm of row
        max_across_input_dims = False
        norm_type = 2
    elif domain == float("inf") and codomain == float("inf"):
        # maximum l1-norm of row
        max_across_input_dims = False
        norm_type = 1
    else:
        raise ValueError('Unknown combination of domain "{}" and codomain "{}"'.format(domain, codomain))


def _max_except_dim(input, dim):
    maxed = input
    for axis in range(input.ndimension() - 1, dim, -1):
        maxed, _ = maxed.max(axis, keepdim=True)
    for axis in range(dim - 1, -1, -1):
        maxed, _ = maxed.max(axis, keepdim=True)
    return maxed


def _norm_except_dim(w, norm_type, dim):
    if norm_type == 1 or norm_type == 2:
        return torch.norm_except_dim(w, norm_type, dim)
    elif norm_type == float('inf'):
        return _max_except_dim(w, dim)


def _ntuple(n):

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


def leaky_elu(x, a=0.3):
    return a * x + (1 - a) * F.elu(x)


def asym_squash(x):
    return torch.tanh(-leaky_elu(-x + 0.5493061829986572)) * 2 + 3


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def projmax_(v):
    """Inplace argmax on absolute value."""
    ind = torch.argmax(torch.abs(v))
    v.zero_()
    v[ind] = 1
    return v


def vector_norm(x, p):
    x = x.view(-1)
    return torch.sum(x**p)**(1 / p)


def normalize_v(v, domain, out=None):
    if not torch.is_tensor(domain) and domain == 2:
        v = F.normalize(v, p=2, dim=0, out=out)
    elif domain == 1:
        v = projmax_(v)
    else:
        vabs = torch.abs(v)
        vph = v / vabs
        vph[torch.isnan(vph)] = 1
        vabs = vabs / torch.max(vabs)
        vabs = vabs**(1 / (domain - 1))
        v = vph * vabs / vector_norm(vabs, domain)
    return v


def normalize_u(u, codomain, out=None):
    if not torch.is_tensor(codomain) and codomain == 2:
        u = F.normalize(u, p=2, dim=0, out=out)
    elif codomain == float('inf'):
        u = projmax_(u)
    else:
        uabs = torch.abs(u)
        uph = u / uabs
        uph[torch.isnan(uph)] = 1
        uabs = uabs / torch.max(uabs)
        uabs = uabs**(codomain - 1)
        if codomain == 1:
            u = uph * uabs / vector_norm(uabs, float('inf'))
        else:
            u = uph * uabs / vector_norm(uabs, codomain / (codomain - 1))
    return u


class InducedNormConv2d(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True, coeff=0.97, domain=2, codomain=2,
        n_iterations=None, atol=None, rtol=None, **unused_kwargs
    ):
        del unused_kwargs
        super(InducedNormConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.domain = domain
        self.codomain = codomain
        self.atol = atol
        self.rtol = rtol
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.register_buffer('initialized', torch.tensor(0))
        self.register_buffer('spatial_dims', torch.tensor([1., 1.]))
        self.register_buffer('scale', torch.tensor(0.))
        self.register_buffer('u', self.weight.new_empty(self.out_channels))
        self.register_buffer('v', self.weight.new_empty(self.in_channels))

    def compute_domain_codomain(self):
        if torch.is_tensor(self.domain):
            domain = asym_squash(self.domain)
            codomain = asym_squash(self.codomain)
        else:
            domain, codomain = self.domain, self.codomain
        return domain, codomain

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def _initialize_u_v(self):
        with torch.no_grad():
            domain, codomain = self.compute_domain_codomain()
            if self.kernel_size == (1, 1):
                self.u.resize_(self.out_channels).normal_(0, 1)
                self.u.copy_(normalize_u(self.u, codomain))
                self.v.resize_(self.in_channels).normal_(0, 1)
                self.v.copy_(normalize_v(self.v, domain))
            else:
                c, h, w = self.in_channels, int(self.spatial_dims[0].item()), int(self.spatial_dims[1].item())
                with torch.no_grad():
                    num_input_dim = c * h * w
                    self.v.resize_(num_input_dim).normal_(0, 1)
                    self.v.copy_(normalize_v(self.v, domain))
                    # forward call to infer the shape
                    u = F.conv2d(
                        self.v.view(1, c, h, w), self.weight, stride=self.stride, padding=self.padding, bias=None
                    )
                    num_output_dim = u.shape[0] * u.shape[1] * u.shape[2] * u.shape[3]
                    # overwrite u with random init
                    self.u.resize_(num_output_dim).normal_(0, 1)
                    self.u.copy_(normalize_u(self.u, codomain))

            self.initialized.fill_(1)

            # Try different random seeds to find the best u and v.
            self.compute_weight(True)
            best_scale = self.scale.clone()
            best_u, best_v = self.u.clone(), self.v.clone()
            if not (domain == 2 and codomain == 2):
                for _ in range(10):
                    if self.kernel_size == (1, 1):
                        self.u.copy_(normalize_u(self.weight.new_empty(self.out_channels).normal_(0, 1), codomain))
                        self.v.copy_(normalize_v(self.weight.new_empty(self.in_channels).normal_(0, 1), domain))
                    else:
                        self.u.copy_(normalize_u(torch.randn(num_output_dim).to(self.weight), codomain))
                        self.v.copy_(normalize_v(torch.randn(num_input_dim).to(self.weight), domain))
                    self.compute_weight(True, n_iterations=200)
                    if self.scale > best_scale:
                        best_u, best_v = self.u.clone(), self.v.clone()
            self.u.copy_(best_u)
            self.v.copy_(best_v)

    def compute_one_iter(self):
        if not self.initialized:
            raise ValueError('Layer needs to be initialized first.')
        domain, codomain = self.compute_domain_codomain()
        if self.kernel_size == (1, 1):
            u = self.u.detach()
            v = self.v.detach()
            weight = self.weight.detach().view(self.out_channels, self.in_channels)
            u = normalize_u(torch.mv(weight, v), codomain)
            v = normalize_v(torch.mv(weight.t(), u), domain)
            return torch.dot(u, torch.mv(weight, v))
        else:
            u = self.u.detach()
            v = self.v.detach()
            weight = self.weight.detach()
            c, h, w = self.in_channels, int(self.spatial_dims[0].item()), int(self.spatial_dims[1].item())
            u_s = F.conv2d(v.view(1, c, h, w), weight, stride=self.stride, padding=self.padding, bias=None)
            out_shape = u_s.shape
            u = normalize_u(u_s.view(-1), codomain)
            v_s = F.conv_transpose2d(
                u.view(out_shape), weight, stride=self.stride, padding=self.padding, output_padding=0
            )
            v = normalize_v(v_s.view(-1), domain)
            weight_v = F.conv2d(v.view(1, c, h, w), weight, stride=self.stride, padding=self.padding, bias=None)
            return torch.dot(u.view(-1), weight_v.view(-1))

    def compute_weight(self, update=True, n_iterations=None, atol=None, rtol=None):
        if not self.initialized:
            self._initialize_u_v()

        if self.kernel_size == (1, 1):
            return self._compute_weight_1x1(update, n_iterations, atol, rtol)
        else:
            return self._compute_weight_kxk(update, n_iterations, atol, rtol)

    def _compute_weight_1x1(self, update=True, n_iterations=None, atol=None, rtol=None):
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol

        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')

        max_itrs = 200
        if n_iterations is not None:
            max_itrs = n_iterations

        u = self.u
        v = self.v
        weight = self.weight.view(self.out_channels, self.in_channels)
        if update:
            with torch.no_grad():
                domain, codomain = self.compute_domain_codomain()
                itrs_used = 0
                for _ in range(max_itrs):
                    old_v = v.clone()
                    old_u = u.clone()

                    u = normalize_u(torch.mv(weight, v), codomain, out=u)
                    v = normalize_v(torch.mv(weight.t(), u), domain, out=v)

                    itrs_used = itrs_used + 1

                    if n_iterations is None and atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                if itrs_used > 0:
                    if domain != 1 and domain != 2:
                        self.v.copy_(v)
                    if codomain != 2 and codomain != float('inf'):
                        self.u.copy_(u)
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight, v))
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight.view(self.out_channels, self.in_channels, 1, 1)

    def _compute_weight_kxk(self, update=True, n_iterations=None, atol=None, rtol=None):
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol

        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')

        max_itrs = 200
        if n_iterations is not None:
            max_itrs = n_iterations

        u = self.u
        v = self.v
        weight = self.weight
        c, h, w = self.in_channels, int(self.spatial_dims[0].item()), int(self.spatial_dims[1].item())
        if update:
            with torch.no_grad():
                domain, codomain = self.compute_domain_codomain()
                itrs_used = 0
                for _ in range(max_itrs):
                    old_u = u.clone()
                    old_v = v.clone()

                    u_s = F.conv2d(v.view(1, c, h, w), weight, stride=self.stride, padding=self.padding, bias=None)
                    out_shape = u_s.shape
                    u = normalize_u(u_s.view(-1), codomain, out=u)

                    v_s = F.conv_transpose2d(
                        u.view(out_shape), weight, stride=self.stride, padding=self.padding, output_padding=0
                    )
                    v = normalize_v(v_s.view(-1), domain, out=v)

                    itrs_used = itrs_used + 1
                    if n_iterations is None and atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                if itrs_used > 0:
                    if domain != 2:
                        self.v.copy_(v)
                    if codomain != 2:
                        self.u.copy_(u)
                    v = v.clone()
                    u = u.clone()

        weight_v = F.conv2d(v.view(1, c, h, w), weight, stride=self.stride, padding=self.padding, bias=None)
        weight_v = weight_v.view(-1)
        sigma = torch.dot(u.view(-1), weight_v)
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight

    def forward(self, input):
        if not self.initialized: self.spatial_dims.copy_(torch.tensor(input.shape[2:4]).to(self.spatial_dims))
        weight = self.compute_weight(update=True)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, 1, 1)

    def extra_repr(self):
        domain, codomain = self.compute_domain_codomain()
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}' ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        s += ', coeff={}, domain={:.2f}, codomain={:.2f}, n_iters={}, atol={}, rtol={}, learnable_ord={}'.format(
            self.coeff, domain, codomain, self.n_iterations, self.atol, self.rtol, torch.is_tensor(self.domain)
        )
        return s.format(**self.__dict__)


class LopConv2d(torch.nn.Conv2d):
    """Lipschitz constant defined using operator norms."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        coeff=0.97,
        domain=float('inf'),
        codomain=float('inf'),
        local_constraint=True,
        **unused_kwargs,
    ):
        del unused_kwargs
        super(LopConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.coeff = coeff
        self.domain = domain
        self.codomain = codomain
        self.local_constraint = local_constraint
        max_across_input_dims, self.norm_type = operator_norm_settings(self.domain, self.codomain)
        self.max_across_dim = 1 if max_across_input_dims else 0
        self.register_buffer('scale', torch.tensor(0.))

    def compute_weight(self):
        scale = _norm_except_dim(self.weight, self.norm_type, dim=self.max_across_dim)
        if not self.local_constraint: scale = scale.max()
        with torch.no_grad():
            self.scale.copy_(scale.max())

        # soft normalization
        factor = torch.max(torch.ones(1).to(self.weight.device), scale / self.coeff)

        return self.weight / factor

    def forward(self, input):
        weight = self.compute_weight()
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, 1, 1)

    def extra_repr(self):
        s = super(LopConv2d, self).extra_repr()
        return s + ', coeff={}, domain={}, codomain={}, local={}'.format(
            self.coeff, self.domain, self.codomain, self.local_constraint
        )


def get_conv2d(
    in_channels, out_channels, kernel_size, stride, padding, bias=True,
        coeff=0.97, domain=None, codomain=None, **kwargs):
    _conv2d = InducedNormConv2d
    if domain == 1:
        if codomain in [1, 2, float('inf')]:
            _conv2d = LopConv2d
    elif codomain == float('inf'):
        if domain in [2, float('inf')]:
            _conv2d = LopConv2d
    return _conv2d(in_channels, out_channels, kernel_size, stride, padding, bias, coeff, domain, codomain, **kwargs)