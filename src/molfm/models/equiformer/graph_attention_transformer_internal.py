# -*- coding: utf-8 -*-
import math
import warnings
from typing import Dict, Optional

import e3nn
import torch
import torch_geometric
from e3nn import o3
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists
from e3nn.util.jit import compile_mode

from fairchem.core.models.gemnet.layers.radial_basis import RadialBasis
from torch import logical_not, nn
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_scatter import scatter

from ..utils import construct_o3irrps, construct_o3irrps_base, generate_graph
from .drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath
from .fast_activation import Activation, Gate
from .fast_layer_norm import EquivariantLayerNormFast
from .gaussian_rbf import GaussianRadialBasisLayer
from .graph_norm import EquivariantGraphNorm
from .instance_norm import EquivariantInstanceNorm
from .layer_norm import EquivariantLayerNormV2
from .radial_func import RadialProfile
from .registry import register_model
from .tensor_product_rescale import (
    FullyConnectedTensorProductRescale,
    LinearRS,
    TensorProductRescale,
    irreps2gate,
    sort_irreps_even_first,
)

_RESCALE = True
_USE_BIAS = True

# QM9
_MAX_ATOM_TYPE = 20
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666


def get_norm_layer(norm_type):
    if norm_type == "graph":
        return EquivariantGraphNorm
    elif norm_type == "instance":
        return EquivariantInstanceNorm
    elif norm_type == "layer":
        return EquivariantLayerNormV2
    elif norm_type == "fast_layer":
        return EquivariantLayerNormFast
    elif norm_type is None:
        return None
    else:
        raise ValueError("Norm type {} not supported.".format(norm_type))


class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2

    def extra_repr(self):
        return "negative_slope={}".format(self.alpha)


def get_mul_0(irreps):
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0


class FullyConnectedTensorProductRescaleNorm(FullyConnectedTensorProductRescale):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
        norm_layer="graph",
    ):
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
        )
        self.norm = get_norm_layer(norm_layer)(self.irreps_out)

    def forward(self, x, y, batch, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.norm(out, batch=batch)
        return out


class FullyConnectedTensorProductRescaleNormSwishGate(
    FullyConnectedTensorProductRescaleNorm
):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
        norm_layer="graph",
    ):
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars,
                [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
        super().__init__(
            irreps_in1,
            irreps_in2,
            gate.irreps_in,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
            norm_layer=norm_layer,
        )
        self.gate = gate

    def forward(self, x, y, batch, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.norm(out, batch=batch)
        out = self.gate(out)
        return out


class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
    ):
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars,
                [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
        super().__init__(
            irreps_in1,
            irreps_in2,
            gate.irreps_in,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
        )
        self.gate = gate

    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.gate(out)
        return out


def DepthwiseTensorProduct(
    irreps_node_input,
    irreps_edge_attr,
    irreps_node_output,
    internal_weights=False,
    bias=True,
):
    """
    The irreps of output is pre-determined.
    `irreps_node_output` is used to get certain types of vectors.
    """
    irreps_output = []
    instructions = []

    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", True))

    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output)  # irreps_output.sort()
    instructions = [
        (i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions
    ]
    tp = TensorProductRescale(
        irreps_node_input,
        irreps_edge_attr,
        irreps_output,
        instructions,
        internal_weights=internal_weights,
        shared_weights=internal_weights,
        bias=bias,
        rescale=_RESCALE,
    )
    return tp


class SeparableFCTP(torch.nn.Module):
    """
    Use separable FCTP for spatial convolution.
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        use_activation=False,
        norm_layer="graph",
        internal_weights=False,
    ):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        norm = get_norm_layer(norm_layer)

        self.dtp = DepthwiseTensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            self.irreps_node_output,
            bias=False,
            internal_weights=internal_weights,
        )

        self.dtp_rad = None
        self.fc_neurons = fc_neurons
        if fc_neurons is not None:
            self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
            for slice, slice_sqrt_k in self.dtp.slices_sqrt_k.values():
                self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
                self.dtp_rad.offset.data[slice] *= slice_sqrt_k

        irreps_lin_output = self.irreps_node_output
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(
            self.irreps_node_output
        )
        if use_activation:
            irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
            irreps_lin_output = irreps_lin_output.simplify()
        self.lin = LinearRS(self.dtp.irreps_out.simplify(), irreps_lin_output)

        self.norm = None
        if norm_layer is not None:
            self.norm = norm(self.lin.irreps_out)

        self.gate = None
        if use_activation:
            if irreps_gated.num_irreps == 0:
                gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
            else:
                gate = Gate(
                    irreps_scalars,
                    [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                    irreps_gates,
                    [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                    irreps_gated,  # gated tensors
                )
            self.gate = gate

    def forward(self, node_input, edge_attr, edge_scalars, batch=None, **kwargs):
        """
        Depthwise TP: `node_input` TP `edge_attr`, with TP parametrized by
        self.dtp_rad(`edge_scalars`).
        """
        weight = None
        if self.dtp_rad is not None and edge_scalars is not None:
            weight = self.dtp_rad(edge_scalars)
        out = self.dtp(node_input, edge_attr, weight)
        out = self.lin(out)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        if self.gate is not None:
            out = self.gate(out)
        return out


@compile_mode("script")
class Vec2AttnHeads(torch.nn.Module):
    """
    Reshape vectors of shape [N, irreps_mid] to vectors of shape
    [N, num_heads, irreps_head].
    """

    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out

    def __repr__(self):
        return "{}(irreps_head={}, num_heads={})".format(
            self.__class__.__name__, self.irreps_head, self.num_heads
        )


@compile_mode("script")
class AttnHeads2Vec(torch.nn.Module):
    """
    Convert vectors of shape [N, num_heads, irreps_head] into
    vectors of shape [N, irreps_head * num_heads].
    """

    def __init__(self, irreps_head):
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out

    def __repr__(self):
        return "{}(irreps_head={})".format(self.__class__.__name__, self.irreps_head)


class ConcatIrrepsTensor(torch.nn.Module):
    def __init__(self, irreps_1, irreps_2):
        super().__init__()
        assert irreps_1 == irreps_1.simplify()
        self.check_sorted(irreps_1)
        assert irreps_2 == irreps_2.simplify()
        self.check_sorted(irreps_2)

        self.irreps_1 = irreps_1
        self.irreps_2 = irreps_2
        self.irreps_out = irreps_1 + irreps_2
        self.irreps_out, _, _ = sort_irreps_even_first(
            self.irreps_out
        )  # self.irreps_out.sort()
        self.irreps_out = self.irreps_out.simplify()

        self.ir_mul_list = []
        lmax = max(irreps_1.lmax, irreps_2.lmax)
        irreps_max = []
        for i in range(lmax + 1):
            irreps_max.append((1, (i, -1)))
            irreps_max.append((1, (i, 1)))
        irreps_max = o3.Irreps(irreps_max)

        start_idx_1, start_idx_2 = 0, 0
        dim_1_list, dim_2_list = self.get_irreps_dim(irreps_1), self.get_irreps_dim(
            irreps_2
        )
        for _, ir in irreps_max:
            dim_1, dim_2 = None, None
            index_1 = self.get_ir_index(ir, irreps_1)
            index_2 = self.get_ir_index(ir, irreps_2)
            if index_1 != -1:
                dim_1 = dim_1_list[index_1]
            if index_2 != -1:
                dim_2 = dim_2_list[index_2]
            self.ir_mul_list.append((start_idx_1, dim_1, start_idx_2, dim_2))
            start_idx_1 = start_idx_1 + dim_1 if dim_1 is not None else start_idx_1
            start_idx_2 = start_idx_2 + dim_2 if dim_2 is not None else start_idx_2

    def get_irreps_dim(self, irreps):
        muls = []
        for mul, ir in irreps:
            muls.append(mul * ir.dim)
        return muls

    def check_sorted(self, irreps):
        lmax = None
        p = None
        for _, ir in irreps:
            if p is None and lmax is None:
                p = ir.p
                lmax = ir.l
                continue
            if ir.l == lmax:
                assert p < ir.p, "Parity order error: {}".format(irreps)
            assert lmax <= ir.l

    def get_ir_index(self, ir, irreps):
        for index, (_, irrep) in enumerate(irreps):
            if irrep == ir:
                return index
        return -1

    def forward(self, feature_1, feature_2):
        output = []
        for i in range(len(self.ir_mul_list)):
            start_idx_1, mul_1, start_idx_2, mul_2 = self.ir_mul_list[i]
            if mul_1 is not None:
                output.append(feature_1.narrow(-1, start_idx_1, mul_1))
            if mul_2 is not None:
                output.append(feature_2.narrow(-1, start_idx_2, mul_2))
        output = torch.cat(output, dim=-1)
        return output

    def __repr__(self):
        return "{}(irreps_1={}, irreps_2={})".format(
            self.__class__.__name__, self.irreps_1, self.irreps_2
        )


@compile_mode("script")
class GraphAttention(torch.nn.Module):
    """
    1. Message = Alpha * Value
    2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
    3. 0e -> Activation -> Inner Product -> (Alpha)
    4. (0e+1e+...) -> (Value)
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        irreps_head,
        num_heads,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0.1,
    ):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = (
            self.irreps_node_input
            if irreps_pre_attn is None
            else o3.Irreps(irreps_pre_attn)
        )
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        # Merge src and dst
        self.merge_src = LinearRS(
            self.irreps_node_input, self.irreps_pre_attn, bias=True
        )
        self.merge_dst = LinearRS(
            self.irreps_node_input, self.irreps_pre_attn, bias=False
        )

        irreps_attn_heads = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(
            irreps_attn_heads
        )  # irreps_attn_heads.sort()
        irreps_attn_heads = irreps_attn_heads.simplify()
        mul_alpha = get_mul_0(irreps_attn_heads)
        mul_alpha_head = mul_alpha // num_heads
        irreps_alpha = o3.Irreps("{}x0e".format(mul_alpha))  # for attention score
        irreps_attn_all = (irreps_alpha + irreps_attn_heads).simplify()

        self.sep_act = None
        if self.nonlinear_message:
            self.sep_act = SeparableFCTP(
                self.irreps_pre_attn,
                self.irreps_edge_attr,
                self.irreps_pre_attn,
                fc_neurons,
                use_activation=True,
                norm_layer=None,
                internal_weights=False,
            )
            self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
            self.sep_value = SeparableFCTP(
                self.irreps_pre_attn,
                self.irreps_edge_attr,
                irreps_attn_heads,
                fc_neurons=None,
                use_activation=False,
                norm_layer=None,
                internal_weights=True,
            )
            self.vec2heads_alpha = Vec2AttnHeads(
                o3.Irreps("{}x0e".format(mul_alpha_head)), num_heads
            )
            self.vec2heads_value = Vec2AttnHeads(self.irreps_head, num_heads)
        else:
            self.sep = SeparableFCTP(
                self.irreps_pre_attn,
                self.irreps_edge_attr,
                irreps_attn_all,
                fc_neurons,
                use_activation=False,
                norm_layer=None,
            )
            self.vec2heads = Vec2AttnHeads(
                (o3.Irreps("{}x0e".format(mul_alpha_head)) + irreps_head).simplify(),
                num_heads,
            )

        self.alpha_act = Activation(
            o3.Irreps("{}x0e".format(mul_alpha_head)), [SmoothLeakyReLU(0.2)]
        )
        self.heads2vec = AttnHeads2Vec(irreps_head)

        self.mul_alpha_head = mul_alpha_head
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch_geometric.nn.inits.glorot(self.alpha_dot)  # Following GATv2

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        self.proj = LinearRS(irreps_attn_heads, self.irreps_node_output)
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(
                self.irreps_node_input, drop_prob=proj_drop
            )

    def forward(
        self,
        node_input,
        node_attr,
        edge_src,
        edge_dst,
        edge_attr,
        edge_scalars,
        batch,
        **kwargs,
    ):
        message_src = self.merge_src(node_input)
        message_dst = self.merge_dst(node_input)
        message = message_src[edge_src] + message_dst[edge_dst]

        if self.nonlinear_message:
            weight = self.sep_act.dtp_rad(edge_scalars)
            message = self.sep_act.dtp(message, edge_attr, weight)
            alpha = self.sep_alpha(message)
            alpha = self.vec2heads_alpha(alpha)
            value = self.sep_act.lin(message)
            value = self.sep_act.gate(value)
            value = self.sep_value(
                value, edge_attr=edge_attr, edge_scalars=edge_scalars
            )
            value = self.vec2heads_value(value)
        else:
            message = self.sep(message, edge_attr=edge_attr, edge_scalars=edge_scalars)
            message = self.vec2heads(message)
            head_dim_size = message.shape[-1]
            alpha = message.narrow(2, 0, self.mul_alpha_head)
            value = message.narrow(
                2, self.mul_alpha_head, (head_dim_size - self.mul_alpha_head)
            )

        # inner product
        alpha = self.alpha_act(alpha)
        alpha = torch.einsum("bik, aik -> bi", alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_dst)
        alpha = alpha.unsqueeze(-1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        attn = value * alpha
        attn = scatter(attn, index=edge_dst, dim=0, dim_size=node_input.shape[0])
        attn = self.heads2vec(attn)

        if self.rescale_degree:
            degree = torch_geometric.utils.degree(
                edge_dst, num_nodes=node_input.shape[0], dtype=node_input.dtype
            )
            degree = degree.view(-1, 1)
            attn = attn * degree

        node_output = self.proj(attn)

        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)

        return node_output

    def extra_repr(self):
        output_str = super(GraphAttention, self).extra_repr()
        output_str = output_str + "rescale_degree={}, ".format(self.rescale_degree)
        return output_str


@compile_mode("script")
class FeedForwardNetwork(torch.nn.Module):
    """
    Use two (FCTP + Gate)
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_node_output,
        irreps_mlp_mid=None,
        proj_drop=0.1,
    ):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_mlp_mid = (
            o3.Irreps(irreps_mlp_mid)
            if irreps_mlp_mid is not None
            else self.irreps_node_input
        )
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        self.fctp_1 = FullyConnectedTensorProductRescaleSwishGate(
            self.irreps_node_input,
            self.irreps_node_attr,
            self.irreps_mlp_mid,
            bias=True,
            rescale=_RESCALE,
        )
        self.fctp_2 = FullyConnectedTensorProductRescale(
            self.irreps_mlp_mid,
            self.irreps_node_attr,
            self.irreps_node_output,
            bias=True,
            rescale=_RESCALE,
        )

        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(
                self.irreps_node_output, drop_prob=proj_drop
            )

    def forward(self, node_input, node_attr, **kwargs):
        node_output = self.fctp_1(node_input, node_attr)
        node_output = self.fctp_2(node_output, node_attr)
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        return node_output


@compile_mode("script")
class TransBlock(torch.nn.Module):
    """
    1. Layer Norm 1 -> GraphAttention -> Layer Norm 2 -> FeedForwardNetwork
    2. Use pre-norm architecture
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        irreps_head,
        num_heads,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0.1,
        drop_path_rate=0.0,
        irreps_mlp_mid=None,
        norm_layer="layer",
    ):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = (
            self.irreps_node_input
            if irreps_pre_attn is None
            else o3.Irreps(irreps_pre_attn)
        )
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = (
            o3.Irreps(irreps_mlp_mid)
            if irreps_mlp_mid is not None
            else self.irreps_node_input
        )

        self.norm_1 = get_norm_layer(norm_layer)(self.irreps_node_input)
        self.ga = GraphAttention(
            irreps_node_input=self.irreps_node_input,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            irreps_node_output=self.irreps_node_input,
            fc_neurons=fc_neurons,
            irreps_head=self.irreps_head,
            num_heads=self.num_heads,
            irreps_pre_attn=self.irreps_pre_attn,
            rescale_degree=self.rescale_degree,
            nonlinear_message=self.nonlinear_message,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
        )

        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        self.norm_2 = get_norm_layer(norm_layer)(self.irreps_node_input)
        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input,  # self.concat_norm_output.irreps_out,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output,
            irreps_mlp_mid=self.irreps_mlp_mid,
            proj_drop=proj_drop,
        )
        self.ffn_shortcut = None
        if self.irreps_node_input != self.irreps_node_output:
            self.ffn_shortcut = FullyConnectedTensorProductRescale(
                self.irreps_node_input,
                self.irreps_node_attr,
                self.irreps_node_output,
                bias=True,
                rescale=_RESCALE,
            )

    def forward(
        self,
        node_input,
        node_attr,
        edge_src,
        edge_dst,
        edge_attr,
        edge_scalars,
        batch,
        **kwargs,
    ):
        node_output = node_input
        node_features = node_input
        node_features = self.norm_1(node_features, batch=batch)
        node_features = self.ga(
            node_input=node_features,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_attr,
            edge_scalars=edge_scalars,
            batch=batch,
        )

        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features

        node_features = node_output
        node_features = self.norm_2(node_features, batch=batch)
        node_features = self.ffn(node_features, node_attr)
        if self.ffn_shortcut is not None:
            node_output = self.ffn_shortcut(node_output, node_attr)

        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features

        return node_output


class NodeEmbeddingNetwork(torch.nn.Module):
    def __init__(self, irreps_node_embedding, max_atom_type=_MAX_ATOM_TYPE, bias=True):
        super().__init__()
        self.max_atom_type = max_atom_type
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.atom_type_lin = LinearRS(
            o3.Irreps("{}x0e".format(self.max_atom_type)),
            self.irreps_node_embedding,
            bias=bias,
        )
        self.atom_type_lin.tp.weight.data.mul_(self.max_atom_type**0.5)

    def forward(self, node_atom):
        """
        `node_atom` is a LongTensor.
        """
        node_atom_onehot = torch.nn.functional.one_hot(
            node_atom.long(), self.max_atom_type
        ).float()
        node_attr = node_atom_onehot
        node_embedding = self.atom_type_lin(node_atom_onehot)

        return node_embedding, node_attr, node_atom_onehot


class ScaledScatter(torch.nn.Module):
    def __init__(self, avg_aggregate_num):
        super().__init__()
        self.avg_aggregate_num = avg_aggregate_num + 0.0

    def forward(self, x, index, **kwargs):
        out = scatter(x, index, **kwargs)
        out = out.div(self.avg_aggregate_num**0.5)
        return out

    def extra_repr(self):
        return "avg_aggregate_num={}".format(self.avg_aggregate_num)


class EdgeDegreeEmbeddingNetwork(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding,
        irreps_edge_attr,
        fc_neurons,
        avg_aggregate_num,
        internal_weights=False,
    ):
        super().__init__()
        self.exp = LinearRS(
            o3.Irreps("1x0e"), irreps_node_embedding, bias=_USE_BIAS, rescale=_RESCALE
        )
        self.dw = DepthwiseTensorProduct(
            irreps_node_embedding,
            irreps_edge_attr,
            irreps_node_embedding,
            internal_weights=internal_weights,
            bias=False,
        )
        self.rad = RadialProfile(fc_neurons + [self.dw.tp.weight_numel])
        for slice, slice_sqrt_k in self.dw.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
            self.rad.offset.data[slice] *= slice_sqrt_k
        self.proj = LinearRS(self.dw.irreps_out.simplify(), irreps_node_embedding)
        self.scale_scatter = ScaledScatter(avg_aggregate_num)

    def forward(self, node_input, edge_attr, edge_scalars, edge_src, edge_dst, batch):
        node_features = torch.ones_like(node_input.narrow(1, 0, 1))
        node_features = self.exp(node_features)
        weight = self.rad(edge_scalars)
        edge_features = self.dw(node_features[edge_src], edge_attr, weight)
        edge_features = self.proj(edge_features)
        node_features = self.scale_scatter(
            edge_features, edge_dst, dim=0, dim_size=node_features.shape[0]
        )
        return node_features


class Equiformer(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=6,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=5.0,
        number_of_basis=128,
        basis_type="gaussian",
        fc_neurons=[64, 64],
        irreps_head="32x0e+16x1o+8x2e",
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        norm_layer="layer",
        max_neighbors=128,
        alpha_drop=0.2,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        interal_weight=False,
        mean=None,
        std=None,
        scale=None,
        atomref=None,
        **kwargs,
    ):
        super().__init__()
        irreps_mlp_mid = irreps_node_embedding
        self.interal_weight = interal_weight
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.max_neighbors = max_neighbors
        self.register_buffer("atomref", atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.hs = self.irreps_node_embedding[0][0]
        self._node_vec_dim = (
            self.irreps_node_embedding.dim - self.irreps_node_embedding[0][0]
        )
        self.lmax = self.irreps_node_embedding.lmax
        self.num_layers = num_layers
        self.irreps_edge_attr = (
            o3.Irreps(irreps_sh)
            if irreps_sh is not None
            else o3.Irreps.spherical_harmonics(self.lmax)
        )
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)

        self.atom_embed = NodeEmbeddingNetwork(
            self.irreps_node_embedding, _MAX_ATOM_TYPE
        )
        self.basis_type = basis_type
        if self.basis_type == "gaussian":
            self.rbf = GaussianRadialBasisLayer(
                self.number_of_basis, cutoff=self.max_radius
            )
        elif self.basis_type == "bessel":
            self.rbf = RadialBasis(
                self.number_of_basis,
                cutoff=self.max_radius,
                rbf={"name": "spherical_bessel"},
            )
        else:
            raise ValueError
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(
            self.irreps_node_embedding,
            self.irreps_edge_attr,
            self.fc_neurons,
            _AVG_DEGREE,
        )

        self.blocks = torch.nn.ModuleList()
        self.build_blocks()

        self.output_proj = LinearRS(
            self.irreps_node_embedding, f"{self.irreps_node_embedding[0][0]}x1e"
        )
        self.apply(self._init_weights)

    def build_blocks(self):
        irreps_block_output = self.irreps_node_embedding
        for i in range(self.num_layers):
            blk = TransBlock(
                irreps_node_input=self.irreps_node_embedding,
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons,
                irreps_head=self.irreps_head,
                num_heads=self.num_heads,
                irreps_pre_attn=self.irreps_pre_attn,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop,
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                norm_layer=self.norm_layer,
            )
            self.blocks.append(blk)

    def reset_parameters(self):
        warnings.warn("sorry, output model not implement reset parameters")

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (
                isinstance(module, torch.nn.Linear)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, EquivariantInstanceNorm)
                or isinstance(module, EquivariantGraphNorm)
                or isinstance(module, GaussianRadialBasisLayer)
                or isinstance(module, RadialBasis)
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, torch.nn.Linear)
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)

    def forward(
        self,
        batched_data: Dict,
        token_embedding: torch.Tensor,
        padding_mask: torch.Tensor,
        pbc_expand_batched: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor, [L, B, H].
            padding_mask (torch.Tensor): Padding mask, [B, L].
            batched_data (Dict): Input data for the forward pass.
        Returns:
            torch.Tensor: Encoded tensor, [B, L, H].
        example:
        batch: attn_bias torch.Size([4, 65, 65])
        batch: attn_edge_type torch.Size([4, 64, 64, 1])
        batch: spatial_pos torch.Size([4, 64, 64])
        batch: in_degree torch.Size([4, 64])
        batch: out_degree torch.Size([4, 64])
        batch: token_id torch.Size([4, 64])
        batch: node_attr torch.Size([4, 64, 1])
        batch: edge_input torch.Size([4, 64, 64, 5, 1])
        batch: energy torch.Size([4])
        batch: forces torch.Size([4, 64, 3])
        batch: pos torch.Size([4, 64, 3])
        batch: node_type_edge torch.Size([4, 64, 64, 2])
        batch: pbc torch.Size([4, 3])
        batch: cell torch.Size([4, 3, 3])
        batch: num_atoms torch.Size([4])
        batch: is_periodic torch.Size([4])
        batch: is_molecule torch.Size([4])
        batch: is_protein torch.Size([4])
        batch: protein_masked_pos torch.Size([4, 64, 3])
        batch: protein_masked_aa torch.Size([4, 64])
        batch: protein_mask torch.Size([4, 64, 3])
        batch: init_pos torch.Size([4, 64, 3])
        """
        tensortype = self.atom_embed.atom_type_lin.tp.weight.dtype
        device = token_embedding.device

        if (
            "pbc" in batched_data
            and batched_data["pbc"] is not None
            and torch.any(batched_data["pbc"])
        ):
            #
            bs, length = padding_mask.shape
            token_mask = logical_not(padding_mask).reshape(-1)
            new_batched_data = Data()
            new_batched_data.atomic_numbers = batched_data["token_id"].reshape(-1)[
                token_mask
            ]
            new_batched_data.batch = (
                torch.arange(bs)
                .reshape(-1, 1)
                .repeat(1, length)
                .reshape(-1)
                .to(device)[token_mask]
                .to(device)
            )

            new_batched_data.pos = (
                batched_data["pos"].reshape(bs * length, 3).float()[token_mask]
            )

            new_batched_data.natoms = torch.sum(token_mask.reshape(bs, length), dim=1)
            new_batched_data.ptr = torch.cumsum(new_batched_data.natoms, dim=0)
            new_batched_data.cell = batched_data["cell"].float()
            (
                edge_index,
                edge_distance,
                edge_distance_vec,
                cell_offsets,
                cell_offset_distances,
                neighbors,
            ) = generate_graph(
                new_batched_data,
                self.max_radius,
                max_neighbors=self.max_neighbors,
                use_pbc=True,
                otf_graph=True,  # oc 20 , material data, these kind of data is saved or otf mode?
                enforce_max_neighbors_strictly=True,
            )

            new_batched_data.edge_index = edge_index
            new_batched_data.pos = new_batched_data.pos.to(tensortype)
            new_batched_data.cell = new_batched_data.cell.to(tensortype)
            new_batched_data.edge_distance = edge_distance.to(tensortype)
            new_batched_data.edge_distance_vec = edge_distance_vec.to(tensortype)
            # remove padding number
            new_batched_data.token_embedding = token_embedding.reshape(bs * length, -1)[
                token_mask
            ]
            _node_attr, _node_vec = self.forward_model(new_batched_data)

            node_attr = torch.zeros((bs * length, self.hs), device=device)
            node_vec = torch.zeros((bs * length, 3 * self.hs), device=device)
            node_attr[token_mask] = _node_attr
            node_vec[token_mask] = _node_vec
            return node_attr.reshape(bs, length, -1), node_vec.reshape(
                bs, length, -1, 3
            ).permute(0, 1, 3, 2)

        else:
            #

            bs, length = padding_mask.shape
            token_mask = logical_not(padding_mask).reshape(-1)
            new_batched_data = Data()
            new_batched_data.atomic_numbers = batched_data["token_id"].reshape(-1)[
                token_mask
            ]
            new_batched_data.batch = (
                torch.arange(bs)
                .reshape(-1, 1)
                .repeat(1, length)
                .reshape(-1)
                .to(device)[token_mask]
                .to(device)
            )

            new_batched_data.pos = (
                batched_data["pos"].reshape(bs * length, 3).float()[token_mask]
            )

            new_batched_data.natoms = torch.sum(token_mask.reshape(bs, length), dim=1)
            new_batched_data.ptr = torch.cumsum(new_batched_data.natoms, dim=0)

            (
                edge_index,
                edge_distance,
                edge_distance_vec,
                cell_offsets,
                cell_offset_distances,
                neighbors,
            ) = generate_graph(
                new_batched_data,
                self.max_radius,
                max_neighbors=self.max_neighbors,
                use_pbc=False,
                otf_graph=True,  # oc 20 , material data, these kind of data is saved or otf mode?
                enforce_max_neighbors_strictly=True,
            )
            new_batched_data.edge_index = edge_index
            new_batched_data.pos = new_batched_data.pos.to(tensortype)
            new_batched_data.edge_distance = edge_distance.to(tensortype)
            new_batched_data.edge_distance_vec = edge_distance_vec.to(tensortype)
            new_batched_data.edge_distance = edge_distance
            new_batched_data.edge_distance_vec = edge_distance_vec

            # remove padding number
            if token_embedding is not None:
                token_embedding = token_embedding.transpose(0, 1).reshape(
                    bs * length, -1
                )[token_mask]
                token_embedding = torch.cat(
                    [
                        token_embedding,
                        torch.zeros(
                            token_embedding.shape[0],
                            self._node_vec_dim,
                            dtype=tensortype,
                            device=device,
                        ),
                    ],
                    dim=-1,
                )
                new_batched_data.token_embedding = token_embedding
            else:
                node_atom = batched_data["atomic_numbers"]
                node_atom = node_atom.reshape(-1)
                atom_embedding, atom_attr, atom_onehot = self.atom_embed(node_atom)
                new_batched_data.token_embedding = atom_embedding
            _node_attr, _node_vec = self.forward_model(new_batched_data)

            node_attr = torch.zeros((bs * length, self.hs), device=device)
            node_vec = torch.zeros((bs * length, 3 * self.hs), device=device)
            node_attr[token_mask] = _node_attr
            node_vec[token_mask] = _node_vec
            return node_attr.reshape(bs, length, -1), node_vec.reshape(
                bs, length, -1, 3
            ).permute(0, 1, 3, 2)

    def forward_model(self, data, **kwargs) -> torch.Tensor:
        pos = data["pos"]
        batch = data["batch"]

        edge_src, edge_dst = radius_graph(
            pos, r=self.max_radius, batch=batch, max_num_neighbors=1000
        )
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr,
            x=edge_vec,  # [:, [1, 2, 0]],
            normalize=True,
            normalization="component",
        )

        atom_embedding = data.token_embedding

        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)
        edge_degree_embedding = self.edge_deg_embed(
            atom_embedding, edge_sh, edge_length_embedding, edge_src, edge_dst, batch
        )
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(
            node_features.narrow(1, 0, 1)
        )  # node_attr is ones array, which is used for ffn_tp: x tp all_ones
        for i, blk in enumerate(self.blocks):
            node_features = blk(
                node_input=node_features,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                batch=batch,
            )

        node_embedding = node_features[
            ..., : self.irreps_node_embedding[0][0]
        ]  # the part of order 0
        node_vec = self.output_proj(node_features)
        return node_embedding, node_vec
