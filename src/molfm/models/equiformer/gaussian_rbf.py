# -*- coding: utf-8 -*-
import torch


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


# From Graphormer
class GaussianRadialBasisLayer(torch.nn.Module):
    def __init__(self, num_basis, cutoff):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff + 0.0
        self.mean = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.std = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.weight = torch.nn.Parameter(torch.ones(1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))

        self.std_init_max = 1.0
        self.std_init_min = 1.0 / self.num_basis
        self.mean_init_max = 1.0
        self.mean_init_min = 0
        torch.nn.init.uniform_(self.mean, self.mean_init_min, self.mean_init_max)
        torch.nn.init.uniform_(self.std, self.std_init_min, self.std_init_max)
        torch.nn.init.constant_(self.weight, 1)
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, dist, node_atom=None, edge_src=None, edge_dst=None):
        x = dist / self.cutoff
        x = x.unsqueeze(-1)
        x = self.weight * x + self.bias
        x = x.expand(-1, self.num_basis)
        mean = self.mean
        std = self.std.abs() + 1e-5
        x = gaussian(x, mean, std)
        return x

    def extra_repr(self):
        return "mean_init_max={}, mean_init_min={}, std_init_max={}, std_init_min={}".format(
            self.mean_init_max, self.mean_init_min, self.std_init_max, self.std_init_min
        )


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        num_basis,
        cutoff: float = 5.0,
        basis_width_scalar: float = 2.0,
    ) -> None:
        super().__init__()
        offset = torch.linspace(0, cutoff, num_basis)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        shape = dist.shape
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        dist = torch.exp(self.coeff * torch.pow(dist, 2))
        return dist.reshape(*shape, -1)
