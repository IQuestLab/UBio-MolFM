# -*- coding: utf-8 -*-
###########################################################################################
# Implementation of the symmetric contraction algorithm presented in the MACE paper
# (Batatia et al, MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields , Eq.10 and 11)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import collections
import dataclasses

from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union, Any 

import numpy as np
import opt_einsum_fx
import torch
import torch.fx
import torch.nn.functional
from e3nn import nn, o3
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
from torch_cluster import radius_graph

###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
###########################################################################################
# Utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from scipy.constants import c, e



from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
###########################################################################################
# Higher Order Real Clebsch Gordan (based on e3nn by Mario Geiger)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

###########################################################################################
# Elementary Block for Building O(3) Equivariant Higher Order Message Passing Neural Network
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################


_TP = collections.namedtuple("_TP", "op, args")
_INPUT = collections.namedtuple("_INPUT", "tensor, start, stop")

CUET_AVAILABLE=False




@dataclasses.dataclass
class CuEquivarianceConfig:
    """Configuration for cuequivariance acceleration"""

    enabled: bool = False
    layout: str = "mul_ir"  # One of: mul_ir, ir_mul
    layout_str: str = "mul_ir"
    group: str = "O3"
    optimize_all: bool = False  # Set to True to enable all optimizations
    optimize_linear: bool = False
    optimize_channelwise: bool = False
    optimize_symmetric: bool = False
    optimize_fctp: bool = False

    # def __post_init__(self):
    #     if self.enabled and CUET_AVAILABLE:
    #         self.layout_str = self.layout
    #         self.layout = getattr(cue, self.layout)
    #         self.group = (
    #             O3_e3nn if self.group == "O3_e3nn" else getattr(cue, self.group)
    #         )
    #     if not CUET_AVAILABLE:
    #         self.enabled = False

class Linear:
    """Returns either a cuet.Linear or o3.Linear based on config"""

    def __new__(
        cls,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        cueq_config=None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_linear)
        ):
            return cuet.Linear(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                use_fallback=True,
            )

        return o3.Linear(
            irreps_in,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )


def mask_head(x: torch.Tensor, head: torch.Tensor, num_heads: int) -> torch.Tensor:
    mask = torch.zeros(x.shape[0], x.shape[1] // num_heads, num_heads, device=x.device)
    idx = torch.arange(mask.shape[0], device=x.device)
    mask[idx, :, head] = 1
    mask = mask.permute(0, 2, 1).reshape(x.shape)
    return x * mask


@compile_mode("script")
class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        num_heads: int = 1,
        cueq_config=None,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads
        self.linear_1 = Linear(
            irreps_in=irreps_in, irreps_out=self.hidden_irreps, cueq_config=cueq_config
        )
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps, irreps_out=irrep_out, cueq_config=cueq_config
        )

    def forward(
        self, x: torch.Tensor, heads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        if hasattr(self, "num_heads"):
            if self.num_heads > 1 and heads is not None:
                x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)  # [n_nodes, len(heads)]


@compile_mode("script")
class BesselBasis(torch.nn.Module):
    """
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


@compile_mode("script")
class PolynomialCutoff(torch.nn.Module):
    """Polynomial cutoff function that goes from 1 to 0 as x goes from 0 to r_max.
    Equation (8) -- TODO: from where?
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.calculate_envelope(x, self.r_max, self.p.to(torch.int))

    @staticmethod
    def calculate_envelope(
        x: torch.Tensor, r_max: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        r_over_r_max = x / r_max
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(r_over_r_max, p)
            + p * (p + 2.0) * torch.pow(r_over_r_max, p + 1)
            - (p * (p + 1.0) / 2) * torch.pow(r_over_r_max, p + 2)
        )
        return envelope * (x < r_max)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"


@compile_mode("script")
class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = "bessel",
        distance_transform: str = "None",
    ):
        super().__init__()
        if radial_type == "bessel":
            self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        else:
            raise ValueError("not support rbf except bessle")
        # elif radial_type == "gaussian":
        #     self.bessel_fn = GaussianBasis(r_max=r_max, num_basis=num_bessel)
        # elif radial_type == "chebyshev":
        #     self.bessel_fn = ChebychevBasis(r_max=r_max, num_basis=num_bessel)
        # if distance_transform == "Agnesi":
        #     self.distance_transform = AgnesiTransform()
        # elif distance_transform == "Soft":
        #     self.distance_transform = SoftTransform()
        # else:
        #     pass
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ):
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        if hasattr(self, "distance_transform"):
            edge_lengths = self.distance_transform(
                edge_lengths, node_attrs, edge_index, atomic_numbers
            )
        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        return radial * cutoff  # [n_edges, n_basis]


def _wigner_nj(
    irrepss: List[o3.Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = torch.eye(irreps.dim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _wigner_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                C = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    C *= ir_out.dim**0.5
                if normalization == "norm":
                    C *= ir_left.dim**0.5 * ir.dim**0.5

                C = torch.einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C = C.reshape(
                    ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim
                )
                for u in range(mul):
                    E = torch.zeros(
                        ir_out.dim,
                        *(irreps.dim for irreps in irrepss_left),
                        irreps_right.dim,
                        dtype=dtype,
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    E[..., sl] = C
                    ret += [
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out),
                                args=(
                                    path_left,
                                    _INPUT(len(irrepss_left), sl.start, sl.stop),
                                ),
                            ),
                            E,
                        )
                    ]
            i += mul * ir.dim
    return sorted(ret, key=lambda x: x[0])


def U_matrix_real(
    irreps_in: Union[str, o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irreps_out = o3.Irreps(irreps_out)
    irrepss = [o3.Irreps(irreps_in)] * correlation
    if correlation == 4:
        filter_ir_mid = [
            (0, 1),
            (1, -1),
            (2, 1),
            (3, -1),
            (4, 1),
            (5, -1),
            (6, 1),
            (7, -1),
            (8, 1),
            (9, -1),
            (10, 1),
            (11, -1),
        ]
    wigners = _wigner_nj(irrepss, normalization, filter_ir_mid, dtype)
    current_ir = wigners[0][0]
    out = []
    stack = torch.tensor([])

    for ir, _, base_o3 in wigners:
        if ir in irreps_out and ir == current_ir:
            stack = torch.cat((stack, base_o3.squeeze().unsqueeze(-1)), dim=-1)
            last_ir = current_ir
        elif ir in irreps_out and ir != current_ir:
            if len(stack) != 0:
                out += [last_ir, stack]
            stack = base_o3.squeeze().unsqueeze(-1)
            current_ir, last_ir = ir, ir
        else:
            current_ir = ir
    out += [last_ir, stack]
    return out


BATCH_EXAMPLE = 10
ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]


@compile_mode("script")
class SymmetricContraction(CodeGenMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: Union[int, Dict[str, int]],
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()

        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        del irreps_in, irreps_out

        if not isinstance(correlation, tuple):
            corr = correlation
            correlation = {}
            for irrep_out in self.irreps_out:
                correlation[irrep_out] = corr

        assert shared_weights or not internal_weights

        if internal_weights is None:
            internal_weights = True

        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        del internal_weights, shared_weights

        self.contractions = torch.nn.ModuleList()
        for irrep_out in self.irreps_out:
            self.contractions.append(
                Contraction(
                    irreps_in=self.irreps_in,
                    irrep_out=o3.Irreps(str(irrep_out.ir)),
                    correlation=correlation[irrep_out],
                    internal_weights=self.internal_weights,
                    num_elements=num_elements,
                    weights=self.shared_weights,
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        outs = [contraction(x, y) for contraction in self.contractions]
        return torch.cat(outs, dim=-1)


@compile_mode("script")
class Contraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps,
        correlation: int,
        internal_weights: bool = True,
        num_elements: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.num_features = irreps_in.count((0, 1))
        self.coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        self.correlation = correlation
        dtype = torch.get_default_dtype()
        for nu in range(1, correlation + 1):
            U_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=irrep_out,
                correlation=nu,
                dtype=dtype,
            )[-1]
            self.register_buffer(f"U_matrix_{nu}", U_matrix)

        # Tensor contraction equations
        self.contractions_weighting = torch.nn.ModuleList()
        self.contractions_features = torch.nn.ModuleList()

        # Create weight for product basis
        self.weights = torch.nn.ParameterList([])

        for i in range(correlation, 0, -1):
            # Shapes definying
            num_params = self.U_tensors(i).size()[-1]
            num_equivariance = 2 * irrep_out.lmax + 1
            num_ell = self.U_tensors(i).size()[-2]

            if i == correlation:
                parse_subscript_main = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                    + ["ik,ekc,bci,be -> bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                )
                graph_module_main = torch.fx.symbolic_trace(
                    lambda x, y, w, z: torch.einsum(
                        "".join(parse_subscript_main), x, y, w, z
                    )
                )

                # Optimizing the contractions
                self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_main,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights_max = w
            else:
                # Generate optimized contractions equations
                parse_subscript_weighting = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                    + ["k,ekc,be->bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                )
                parse_subscript_features = (
                    ["bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                    + ["i,bci->bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                )

                # Symbolic tracing of contractions
                graph_module_weighting = torch.fx.symbolic_trace(
                    lambda x, y, z: torch.einsum(
                        "".join(parse_subscript_weighting), x, y, z
                    )
                )
                graph_module_features = torch.fx.symbolic_trace(
                    lambda x, y: torch.einsum("".join(parse_subscript_features), x, y)
                )

                # Optimizing the contractions
                graph_opt_weighting = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_weighting,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                graph_opt_features = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_features,
                    example_inputs=(
                        torch.randn(
                            [BATCH_EXAMPLE, self.num_features, num_equivariance]
                            + [num_ell] * i
                        ).squeeze(2),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                    ),
                )
                self.contractions_weighting.append(graph_opt_weighting)
                self.contractions_features.append(graph_opt_features)
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights.append(w)
        if not internal_weights:
            self.weights = weights[:-1]
            self.weights_max = weights[-1]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        out = self.graph_opt_main(
            self.U_tensors(self.correlation),
            self.weights_max,
            x,
            y,
        )
        for i, (weight, contract_weights, contract_features) in enumerate(
            zip(self.weights, self.contractions_weighting, self.contractions_features)
        ):
            c_tensor = contract_weights(
                self.U_tensors(self.correlation - i - 1),
                weight,
                y,
            )
            c_tensor = c_tensor + out
            out = contract_features(c_tensor, x)

        return out.view(out.shape[0], -1)

    def U_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_{nu}"]


class reshape_irrepstoe3nn(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            d = ir.dim
            self.dims.append(d)
            self.muls.append(mul)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        ix = 0
        out = []
        batch, _, _ = tensor.shape
        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, ix : ix + d]  # [batch, sample, mul * repr]
            ix += d
            field = field.reshape(batch, d, mul).permute(0, 2, 1)
            out.append(field)
        return torch.cat(out, dim=-1)

    def back2orderTmul(self, tensor: torch.Tensor) -> torch.Tensor:
        ix = 0
        out = []
        batch, _ = tensor.shape
        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, ix : ix + d * mul]  # [batch, sample, mul * repr]
            ix += d * mul
            out.append(field.reshape((batch, mul, d)).permute(0, 2, 1))

        return torch.cat(out, dim=-2)



class reshape_irreps(torch.nn.Module):
    def __init__(
        self, irreps: o3.Irreps, cueq_config: Optional[CuEquivarianceConfig] = None
    ) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.cueq_config = cueq_config
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            d = ir.dim
            self.dims.append(d)
            self.muls.append(mul)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        ix = 0
        out = []
        batch, _ = tensor.shape
        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d
            if hasattr(self, "cueq_config"):
                if self.cueq_config is not None:
                    if self.cueq_config.layout_str == "mul_ir":
                        field = field.reshape(batch, mul, d)
                    else:
                        field = field.reshape(batch, d, mul)
                else:
                    field = field.reshape(batch, mul, d)
            else:
                field = field.reshape(batch, mul, d)
            out.append(field)

        if hasattr(self, "cueq_config"):
            if self.cueq_config is not None:  # pylint: disable=no-else-return
                if self.cueq_config.layout_str == "mul_ir":
                    return torch.cat(out, dim=-1)
                return torch.cat(out, dim=-2)
            else:
                return torch.cat(out, dim=-1)
        return torch.cat(out, dim=-1)


_SIMPLIFY_REGISTRY = set()


def simplify_if_compile(module: torch.nn.Module) -> torch.nn.Module:
    """Decorator to register a module for symbolic simplification

    The decorated module will be simplifed using `torch.fx.symbolic_trace`.
    This constrains the module to not have any dynamic control flow, see:

    https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing

    Args:
        module (nn.Module): the module to register

    Returns:
        nn.Module: registered module
    """
    _SIMPLIFY_REGISTRY.add(module)
    return module



@compile_mode("script")
class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=irreps_out, cueq_config=cueq_config
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]
        return self.linear(node_attrs)



@compile_mode("script")
class LinearReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=irrep_out, cueq_config=cueq_config
        )

    def forward(
        self,
        x: torch.Tensor,
        heads: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]



@compile_mode("script")
class LinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, dipole_only: bool = False):
        super().__init__()
        if dipole_only:
            self.irreps_out = o3.Irreps("1x1o")
        else:
            self.irreps_out = o3.Irreps("1x0e + 1x1o")
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=self.irreps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


@compile_mode("script")
class NonLinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Callable,
        dipole_only: bool = False,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        if dipole_only:
            self.irreps_out = o3.Irreps("1x1o")
        else:
            self.irreps_out = o3.Irreps("1x0e + 1x1o")
        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l == 0 and ir in self.irreps_out]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l > 0 and ir in self.irreps_out]
        )
        irreps_gates = o3.Irreps([mul, "0e"] for mul, _ in irreps_gated)
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate for _, ir in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.irreps_nonlin)
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=self.irreps_out
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


@compile_mode("script")
class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        # assert len(atomic_energies.shape) == 1

        self.register_buffer(
            "atomic_energies",
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype()),
        )  # [n_elements, n_heads]

    def forward(
        self, x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, torch.atleast_2d(self.atomic_energies).T)

    def __repr__(self):
        formatted_energies = ", ".join(
            [
                "[" + ", ".join([f"{x:.4f}" for x in group]) + "]"
                for group in torch.atleast_2d(self.atomic_energies)
            ]
        )
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"



@compile_mode("script")
class EquivariantProductBasisBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: int,
        use_sc: bool = True,
        num_elements: Optional[int] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.symmetric_contractions = SymmetricContractionWrapper(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
            cueq_config=cueq_config,
        )
        # Update linear
        self.linear = Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        )
        self.cueq_config = cueq_config

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: Optional[torch.Tensor],
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        use_cueq = False
        use_cueq_mul_ir = False
        if hasattr(self, "cueq_config"):
            if self.cueq_config is not None:
                if self.cueq_config.enabled and (
                    self.cueq_config.optimize_all or self.cueq_config.optimize_symmetric
                ):
                    use_cueq = True
                if self.cueq_config.layout_str == "mul_ir":
                    use_cueq_mul_ir = True
        if use_cueq:
            if use_cueq_mul_ir:
                node_feats = torch.transpose(node_feats, 1, 2)
            index_attrs = torch.nonzero(node_attrs)[:, 1].int()
            node_feats = self.symmetric_contractions(
                node_feats.flatten(1),
                index_attrs,
            )
        else:
            node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)


class SymmetricContractionWrapper:
    """Wrapper around SymmetricContraction/cuet.SymmetricContraction"""

    def __new__(
        cls,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        num_elements: Optional[int] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_symmetric)
        ):
            return cuet.SymmetricContraction(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out),
                layout_in=cue.ir_mul,
                layout_out=cueq_config.layout,
                contraction_degree=correlation,
                num_elements=num_elements,
                original_mace=True,
                dtype=torch.get_default_dtype(),
                math_dtype=torch.get_default_dtype(),
            )

        return SymmetricContraction(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=correlation,
            num_elements=num_elements,
        )

    
class LAMMPS_MP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        feats, data = args  # unpack
        ctx.vec_len = feats.shape[-1]
        ctx.data = data
        out = torch.empty_like(feats)
        data.forward_exchange(feats, out, ctx.vec_len)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs  # unpack
        gout = torch.empty_like(grad)
        ctx.data.reverse_exchange(grad, gout, ctx.vec_len)
        return gout, None


@compile_mode("script")
class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        radial_MLP: Optional[List[int]] = None,
        cueq_config = None,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = radial_MLP
        self.cueq_config = cueq_config
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    def handle_lammps(
        self,
        node_feats: torch.Tensor,
        lammps_class: Optional[Any],
        lammps_natoms: Tuple[int, int],
        first_layer: bool,
    ) -> torch.Tensor:  # noqa: D401 – internal helper
        if lammps_class is None or first_layer or torch.jit.is_scripting():
            return node_feats
        _, n_total = lammps_natoms
        pad = torch.zeros(
            (n_total, node_feats.shape[1]),
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        node_feats = torch.cat((node_feats, pad), dim=0)
        node_feats = LAMMPS_MP.apply(node_feats, lammps_class)
        return node_feats

    def truncate_ghosts(
        self, tensor: torch.Tensor, n_real: Optional[int] = None
    ) -> torch.Tensor:
        """Truncate the tensor to only keep the real atoms in case of presence of ghost atoms during multi-GPU MD simulations."""
        return tensor[:n_real] if n_real is not None else tensor

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


nonlinearities = {1: torch.nn.functional.silu, -1: torch.tanh}


@compile_mode("script")
class TensorProductWeightsBlock(torch.nn.Module):
    def __init__(self, num_elements: int, num_edge_feats: int, num_feats_out: int):
        super().__init__()

        weights = torch.empty(
            (num_elements, num_edge_feats, num_feats_out),
            dtype=torch.get_default_dtype(),
        )
        torch.nn.init.xavier_uniform_(weights)
        self.weights = torch.nn.Parameter(weights)

    def forward(
        self,
        sender_or_receiver_node_attrs: torch.Tensor,  # assumes that the node attributes are one-hot encoded
        edge_feats: torch.Tensor,
    ):
        return torch.einsum(
            "be, ba, aek -> bk", edge_feats, sender_or_receiver_node_attrs, self.weights
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(shape=({", ".join(str(s) for s in self.weights.shape)}), '
            f"weights={np.prod(self.weights.shape)})"
        )





# pylint: disable=C0302
def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src

def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

# Based on mir-group/nequip
def tp_out_irreps_with_instructions(
    irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
) -> Tuple[o3.Irreps, List]:
    trainable = True

    # Collect possible irreps and their instructions
    irreps_out_list: List[Tuple[int, o3.Irreps]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions

class FullyConnectedTensorProduct:
    """Wrapper around o3.FullyConnectedTensorProduct/cuet.FullyConnectedTensorProduct"""

    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        cueq_config = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_fctp)
        ):
            return cuet.FullyConnectedTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                use_fallback=True,
            )

        return o3.FullyConnectedTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )


class TensorProduct:
    """Wrapper around o3.TensorProduct/cuet.ChannelwiseTensorProduct"""

    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: Optional[List] = None,
        shared_weights: bool = False,
        internal_weights: bool = False,
        cueq_config = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_channelwise)
        ):
            return cuet.ChannelWiseTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                dtype=torch.get_default_dtype(),
                math_dtype=torch.get_default_dtype(),
            )

        return o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )
        
        

@compile_mode("script")
class RealAgnosticResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,  # gate
        )
        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        sc = self.truncate_ghosts(sc, n_real)
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]



@compile_mode("script")
class RealAgnosticInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        lammps_natoms: Tuple[int, int] = (0, 0),
        lammps_class: Optional[Any] = None,
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        return (
            self.reshape(message),
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        # assert len(atomic_energies.shape) == 1

        self.register_buffer(
            "atomic_energies",
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype()),
        )  # [n_elements, n_heads]

    def forward(
        self, x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, torch.atleast_2d(self.atomic_energies).T)

    def __repr__(self):
        formatted_energies = ", ".join(
            [
                "[" + ", ".join([f"{x:.4f}" for x in group]) + "]"
                for group in torch.atleast_2d(self.atomic_energies)
            ]
        )
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"


class InteractionKwargs(NamedTuple):
    lammps_class: Optional[torch.Tensor]
    lammps_natoms: Tuple[int, int] = (0, 0)


class GraphContext(NamedTuple):
    is_lammps: bool
    num_graphs: int
    num_atoms_arange: torch.Tensor
    displacement: Optional[torch.Tensor]
    positions: torch.Tensor
    vectors: torch.Tensor
    lengths: torch.Tensor
    cell: torch.Tensor
    node_heads: torch.Tensor
    interaction_kwargs: InteractionKwargs


def prepare_graph(
    data: Dict[str, torch.Tensor],
    compute_virials: bool = False,
    compute_stress: bool = False,
    compute_displacement: bool = False,
    lammps_mliap: bool = False,
) -> GraphContext:
    if torch.jit.is_scripting():
        lammps_mliap = False

    node_heads = (
        data["head"][data["batch"]]
        if "head" in data
        else torch.zeros_like(data["batch"])
    )

    if lammps_mliap:
        n_real, n_total = data["natoms"][0], data["natoms"][1]
        num_graphs = 2
        num_atoms_arange = torch.arange(n_real, device=data["node_attrs"].device)
        displacement = None
        positions = torch.zeros(
            (int(n_real), 3),
            dtype=data["vectors"].dtype,
            device=data["vectors"].device,
        )
        cell = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["vectors"].dtype,
            device=data["vectors"].device,
        )
        vectors = data["vectors"].requires_grad_(True)
        lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        ikw = InteractionKwargs(data["lammps_class"], (n_real, n_total))
    else:
        data["positions"].requires_grad_(True)
        positions = data["positions"]
        cell = data["cell"]
        num_atoms_arange = torch.arange(positions.shape[0], device=positions.device)
        num_graphs = int(data["ptr"].numel() - 1)
        displacement = torch.zeros(
            (num_graphs, 3, 3), dtype=positions.dtype, device=positions.device
        )
        if compute_virials or compute_stress or compute_displacement:
            raise ValueError("not support")
            # p, s, displacement = get_symmetric_displacement(
            #     positions=positions,
            #     unit_shifts=data["unit_shifts"],
            #     cell=cell,
            #     edge_index=data["edge_index"],
            #     num_graphs=num_graphs,
            #     batch=data["batch"],
            # )
            # data["positions"], data["shifts"] = p, s
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        ikw = InteractionKwargs(None, (0, 0))

    return GraphContext(
        is_lammps=lammps_mliap,
        num_graphs=num_graphs,
        num_atoms_arange=num_atoms_arange,
        displacement=displacement,
        positions=positions,
        vectors=vectors,
        lengths=lengths,
        cell=cell,
        node_heads=node_heads,
        interaction_kwargs=ikw,
    )
def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
):
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths

@compile_mode("script")
class MACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,
        lammps_mliap: Optional[bool] = False,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if heads is None:
            heads = ["Default"]
        self.heads = heads
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        self.lammps_mliap = lammps_mliap
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(p=num_polynomial_cutoff)
            self.pair_repulsion = True

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(
            LinearReadoutBlock(
                hidden_irreps, o3.Irreps(f"{len(heads)}x0e"), cueq_config
            )
        )

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
                cueq_config=cueq_config,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{len(heads)}x0e"),
                        len(heads),
                        cueq_config,
                    )
                )
            else:
                self.readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps, o3.Irreps(f"{len(heads)}x0e"), cueq_config
                    )
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )
        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        )  # [n_graphs, n_heads]
        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
            pair_energy = scatter_sum(
                src=pair_node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
        else:
            pair_node_energy = torch.zeros_like(node_e0)
            pair_energy = torch.zeros_like(e0)

        # Interactions
        energies = [e0, pair_energy]
        node_energies_list = [node_e0, pair_node_energy]
        node_feats_concat: List[torch.Tensor] = []

        for i, (interaction, product, readout) in enumerate(
            zip(self.interactions, self.products, self.readouts)
        ):
            node_attrs_slice = data["node_attrs"]
            if is_lammps and i > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                first_layer=(i == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and i == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_concat.append(node_feats)
            node_es = readout(node_feats, node_heads)[num_atoms_arange, node_heads]
            energy = scatter_sum(node_es, data["batch"], dim=0, dim_size=num_graphs)
            energies.append(energy)
            node_energies_list.append(node_es)

        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)
        node_energy = torch.sum(torch.stack(node_energies_list, dim=-1), dim=-1)
        node_feats_out = torch.cat(node_feats_concat, dim=-1)
        node_energy = node_e0.double() + pair_node_energy.double()

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=total_energy,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
        )

        atomic_virials: Optional[torch.Tensor] = None
        atomic_stresses: Optional[torch.Tensor] = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )
        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
        }




@compile_mode("script")
class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.register_buffer(
            "scale",
            torch.tensor(scale, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "shift",
            torch.tensor(shift, dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        return (
            torch.atleast_1d(self.scale)[head] * x + torch.atleast_1d(self.shift)[head]
        )

    def __repr__(self):
        formatted_scale = (
            ", ".join([f"{x:.4f}" for x in self.scale])
            if self.scale.numel() > 1
            else f"{self.scale.item():.4f}"
        )
        formatted_shift = (
            ", ".join([f"{x:.4f}" for x in self.shift])
            if self.shift.numel() > 1
            else f"{self.shift.item():.4f}"
        )
        return f"{self.__class__.__name__}(scale={formatted_scale}, shift={formatted_shift})"


@compile_mode("script")
class ScaleShiftMACE(MACE):
    def __init__(
        self,
        atomic_inter_scale: float = 1.0885,  # follow MACE
        atomic_inter_shift: float = 0, # follow MACE
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        
        # # Atomic energies
        # node_e0 = self.atomic_energies_fn(data["node_attrs"])[
        #     num_atoms_arange, node_heads
        # ]
        # e0 = scatter_sum(
        #     src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        # )  # [n_graphs, num_heads]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # if hasattr(self, "pair_repulsion"):
        #     pair_node_energy = self.pair_repulsion_fn(
        #         lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        #     )
        #     if is_lammps:
        #         pair_node_energy = pair_node_energy[: lammps_natoms[0]]
        # else:
        #     pair_node_energy = torch.zeros_like(node_e0)
        node_heads = torch.zeros_like(data["batch"])
    
        # Interactions
        node_es_list = [] #pair_node_energy
        node_feats_list: List[torch.Tensor] = []

        for i, (interaction, product, readout) in enumerate(
            zip(self.interactions, self.products, self.readouts)
        ):
            node_attrs_slice = data["node_attrs"]
            # if is_lammps and i > 0:
            #     node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                first_layer=(i == 0),
                # lammps_class=lammps_class,
                # lammps_natoms=lammps_natoms,
            )
            # if is_lammps and i == 0:
            #     node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_list.append(node_feats)
            node_es_list.append(
                readout(node_feats, node_heads)[num_atoms_arange, node_heads]
            )

        node_feats_out = torch.cat(node_feats_list, dim=-1)
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)

        total_energy = inter_e
        # node_energy = node_e0.clone().double() + node_inter_es.clone().double()

        # forces, virials, stress, hessian, edge_forces = get_outputs(
        #     energy=inter_e,
        #     positions=positions,
        #     displacement=displacement,
        #     vectors=vectors,
        #     cell=cell,
        #     training=training,
        #     compute_force=compute_force,
        #     compute_virials=compute_virials,
        #     compute_stress=compute_stress,
        #     compute_hessian=compute_hessian,
        #     compute_edge_forces=compute_edge_forces or compute_atomic_stresses,
        # )

        # atomic_virials: Optional[torch.Tensor] = None
        # atomic_stresses: Optional[torch.Tensor] = None
        # if compute_atomic_stresses and edge_forces is not None:
        #     atomic_virials, atomic_stresses = get_atomic_virials_stresses(
        #         edge_forces=edge_forces,
        #         edge_index=data["edge_index"],
        #         vectors=vectors,
        #         num_atoms=positions.shape[0],
        #         batch=data["batch"],
        #         cell=cell,
        #     )
        return total_energy
        #     ,{
        #     "energy": total_energy,
        #     "node_energy": node_energy,
        #     "interaction_energy": inter_e,
        #     "forces": forces,
        #     "edge_forces": edge_forces,
        #     "virials": virials,
        #     "stress": stress,
        #     "atomic_virials": atomic_virials,
        #     "atomic_stresses": atomic_stresses,
        #     "hessian": hessian,
        #     "displacement": displacement,
        #     "node_feats": node_feats_out,
        # }

@compile_mode("script")
class EnergyDipolesMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[np.ndarray],
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[:2]
                )  # Select scalars and l=1 vectors for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out, MLP_irreps, gate, dipole_only=False
                    )
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False)
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_edge_forces: bool = False, # pylint: disable=W0613
        compute_atomic_stresses: bool = False, # pylint: disable=W0613
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        # if compute_virials or compute_stress or compute_displacement:
        #     (
        #         data["positions"],
        #         data["shifts"],
        #         displacement,
        #     ) = get_symmetric_displacement(
        #         positions=data["positions"],
        #         unit_shifts=data["unit_shifts"],
        #         cell=data["cell"],
        #         edge_index=data["edge_index"],
        #         num_graphs=num_graphs,
        #         batch=data["batch"],
        #     )

        # # Atomic energies
        # node_e0 = self.atomic_energies_fn(data["node_attrs"])[
        #     num_atoms_arange, data["head"][data["batch"]]
        # ]
        # e0 = scatter_sum(
        #     src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        # )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        energies = [] # [e0]
        node_energies_list = [] # [node_e0]
        dipoles = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_out = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            # node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            node_energies = node_out[:, 0]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            # print(node_energies)
            energies.append(energy)
            node_dipoles = node_out[:, 1:]
            dipoles.append(node_dipoles)

        # Compute the energies and dipoles
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        # node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        # node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"].unsqueeze(-1),
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        # baseline = compute_fixed_charge_dipole(
        #     charges=data["charges"],
        #     positions=data["positions"],
        #     batch=data["batch"],
        #     num_graphs=num_graphs,
        # )  # [n_graphs,3]
        # total_dipole = total_dipole + baseline

        # forces, virials, stress, _, _ = get_outputs(
        #     energy=total_energy,
        #     positions=data["positions"],
        #     displacement=displacement,
        #     cell=data["cell"],
        #     training=training,
        #     compute_force=compute_force,
        #     compute_virials=compute_virials,
        #     compute_stress=compute_stress,
        # )

        # output = {
        #     "energy": total_energy,
        #     "node_energy": node_energy,
        #     "contributions": contributions,
        #     "forces": forces,
        #     "virials": virials,
        #     "stress": stress,
        #     "displacement": displacement,
        #     "dipole": total_dipole,
        #     "atomic_dipoles": atomic_dipoles,
        # }
        return total_energy

def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>
    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=indices.device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


class MACE_Wrapper(torch.nn.Module):
    def __init__(
        self,
        
    ):
        
        super().__init__()
        model_config = dict(
            r_max=5,
            num_bessel=8, #number of radial basis functions
            num_polynomial_cutoff=5, # this is exponent coefficient
            max_ell=3,
            interaction_cls=RealAgnosticResidualInteractionBlock,# modules.interaction_classes[args.interaction],
            num_interactions= 2, #means Layers
            num_elements=100,
            hidden_irreps=o3.Irreps("224x0e+224x1o+224x2e"), # this is Large. medium is 128x0e+128x1o
            atomic_energies=list([0]*100),
            avg_num_neighbors=1, #args.avg_num_neighbors,
            atomic_numbers=list(range(100)),
        )


        self.model = ScaleShiftMACE(
                    **model_config,
                    correlation=3,
                    gate=torch.nn.functional.silu,
                    interaction_cls_first=RealAgnosticInteractionBlock,
                MLP_irreps=o3.Irreps("16x0e"))


        # model_config = dict(
        #     r_max=5,
        #     num_bessel=8, #number of radial basis functions
        #     num_polynomial_cutoff=5, # this is exponent coefficient
        #     max_ell=3,
        #     interaction_cls=RealAgnosticInteractionBlock,# modules.interaction_classes[args.interaction],
        #     num_interactions= 4, #args.num_interactions, means layers!!!
        #     num_elements=100,
        #     hidden_irreps=o3.Irreps("128x0e+128x1o+128x2e"),
        #     atomic_energies=list([0]*100),
        #     avg_num_neighbors=1, #args.avg_num_neighbors,
        #     atomic_numbers=list(range(100)),
        # )


        # self.model = EnergyDipolesMACE(
        #             **model_config,
        #             correlation=3,
        #             gate=torch.nn.functional.silu,
        #             interaction_cls_first=RealAgnosticInteractionBlock,
        #         MLP_irreps=o3.Irreps("16x0e"))
        
        
    def forward(self,
                batched_data,
                    ):          
        device = batched_data["pos"].device
        B,L = batched_data["pos"].shape[:2]
        batched_data["pos"].requires_grad_(True)
        if "padding_mask" not in batched_data:
            padding_mask = batched_data["token_id"].eq(0)
        else:
            padding_mask = batched_data["padding_mask"]
        node_mask = ~padding_mask
        pos = batched_data["pos"][node_mask].float()
        node_attrs = batched_data["token_id"][node_mask].long()
        ptr = torch.cat(
                    [
                        torch.Tensor(
                            [
                                0,
                            ]
                        )
                        .int()
                        .to(device),
                        torch.cumsum(torch.sum(node_mask, dim=-1), dim=-1),
                    ],
                    dim=0,
                )
        
        batch = torch.arange(B).reshape(B, 1).repeat(1, L).to(device)[node_mask]

        data = {'positions':pos,
                "ptr":ptr,
                "node_attrs":to_one_hot(
                    node_attrs.unsqueeze(-1),
                    num_classes=100,
                ),
                "batch" : batch
                }

        data["edge_index"] = radius_graph(
            data["positions"],
            r=5,
            batch=data["batch"],
            max_num_neighbors=320,
        ).long()
        data["shifts"] = torch.zeros(data["edge_index"].shape[1],3).to(device)

        energy = self.model(data)        
        forces = torch.zeros_like(batched_data["pos"])
        forces[node_mask] = -torch.autograd.grad(
                outputs=energy,
                inputs=pos,
                grad_outputs= [torch.ones_like(energy)],
                create_graph=self.training,
                retain_graph=True,
            )[0]
        num_atoms = torch.sum(node_mask,dim = -1)
        result = {"pred_energy":energy,
                  "pred_energy_per_atom":energy/num_atoms,
                  "pred_forces":forces,
                   "node_irrepsBxN": None,
                    "node_featuresBxN": None,
                    "node_vec_featuresBxN": None,

                  }
        return result