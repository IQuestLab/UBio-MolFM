import torch

from torch import nn
from .module_utils import SO3_Grid,SeparableS2Activation,S2Activation
from fairchem.core.models.uma.nn.activation import GateActivation
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear

class SpectralAtomwise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid

        self.scalar_mlp = nn.Sequential(
            nn.Linear(
                self.sphere_channels,
                self.lmax * self.hidden_channels,
                bias=True,
            ),
            nn.SiLU(),
        )

        self.so3_linear_1 = SO3_Linear(
            self.sphere_channels, self.hidden_channels, lmax=self.lmax
        )
        self.act = GateActivation(
            lmax=self.lmax, mmax=self.lmax, num_channels=self.hidden_channels
        )
        self.so3_linear_2 = SO3_Linear(
            self.hidden_channels, self.sphere_channels, lmax=self.lmax
        )

    def forward(self, x):
        gating_scalars = self.scalar_mlp(x.narrow(1, 0, 1))
        x = self.so3_linear_1(x)
        x = self.act(gating_scalars, x)
        x = self.so3_linear_2(x)
        return x


class FeedForwardNetwork_s2(torch.nn.Module):
    """
    FeedForwardNetwork: Perform feedforward network with S2 activation or gate activation

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        activation (str):           Type of activation function
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs.
        use_sep_s2_act (bool):      If `True`, use separable grid MLP when `use_grid_mlp` is True.
    """

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        output_channels,
        lmax,
        mmax=2,
        grid_resolution=18,
        use_gate_act=False,  # [True, False] Switch between gate activation and S2 activation
        use_grid_mlp=True,  # [False, True] If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act=True,  # Separable S2 activation. Used for ablation study.
    ):
        super(FeedForwardNetwork_s2, self).__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.sphere_channels_all = self.sphere_channels
        self.so3_grid = torch.nn.ModuleList()
        self.lmax = lmax
        self.max_lmax = self.lmax
        self.lmax_list = [lmax]
        self.so3_grid = SO3_Grid(
                        lmax, lmax, resolution=grid_resolution  # , normalization="component"
                    )

        self.use_gate_act = use_gate_act  # [True, False] Switch between gate activation and S2 activation
        self.use_grid_mlp = use_grid_mlp  # [False, True] If `True`, use projecting to grids and performing MLPs for FFNs.
        self.use_sep_s2_act = (
            use_sep_s2_act  # Separable S2 activation. Used for ablation study.
        )

        self.so3_linear_1 = SO3_Linear(
            self.sphere_channels_all, self.hidden_channels, lmax=self.lmax
        )
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                self.scalar_mlp = nn.Sequential(
                    nn.Linear(
                        self.sphere_channels_all,
                        self.hidden_channels,
                        bias=True,
                    ),
                    nn.SiLU(),
                )
            else:
                self.scalar_mlp = None
            self.grid_mlp = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            )
        else:
            if self.use_gate_act:
                self.gating_linear = torch.nn.Linear(
                    self.sphere_channels_all,
                    self.lmax * self.hidden_channels,
                )
                self.gate_act = GateActivation(
                    self.lmax, self.lmax, self.hidden_channels
                )
            else:
                if self.use_sep_s2_act:
                    self.gating_linear = torch.nn.Linear(
                        self.sphere_channels_all, self.hidden_channels
                    )
                    self.s2_act = SeparableS2Activation(self.lmax, self.lmax)
                else:
                    self.gating_linear = None
                    self.s2_act = S2Activation(self.lmax, self.lmax)
        self.so3_linear_2 = SO3_Linear(
            self.hidden_channels, self.output_channels, lmax=self.lmax
        )

    def forward(self, input_embedding):
        out_shape = input_embedding.shape[:-2]

        input_embedding = input_embedding.reshape(
            out_shape.numel(), (self.lmax + 1) ** 2, self.sphere_channels
        )

        input_embedding = self._forward(input_embedding)

        return input_embedding.reshape(out_shape + (-1, self.output_channels))

    def _forward(self, input_embedding):
        gating_scalars = None
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                gating_scalars = self.scalar_mlp(
                    input_embedding.narrow(1, 0, 1)
                )
        else:
            if self.gating_linear is not None:
                gating_scalars = self.gating_linear(
                    input_embedding.narrow(1, 0, 1)
                )

        input_embedding = self.so3_linear_1(input_embedding)

        if self.use_grid_mlp:
            # Project to grid
            input_embedding_grid = self.so3_grid.to_grid(input_embedding,self.max_lmax,self.max_lmax)
            input_embedding_grid = self.grid_mlp(input_embedding_grid)

            input_embedding = self.so3_grid.from_grid(input_embedding_grid,self.max_lmax,self.max_lmax)

            if self.use_sep_s2_act:
                input_embedding = torch.cat(
                    (
                        gating_scalars,
                        input_embedding.narrow(
                            1, 1, input_embedding.shape[1] - 1
                        ),
                    ),
                    dim=1,
                )
        else:
            if self.use_gate_act:
                input_embedding = self.gate_act(
                    gating_scalars, input_embedding
                )
            else:
                if self.use_sep_s2_act:
                    input_embedding = self.s2_act(
                        gating_scalars,
                        input_embedding,
                        self.so3_grid,
                    )
                else:
                    input_embedding = self.s2_act(
                        input_embedding, self.so3_grid
                    )

        return self.so3_linear_2(input_embedding)
