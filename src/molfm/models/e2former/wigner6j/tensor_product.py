# -*- coding: utf-8 -*-
import math
import warnings
from collections import OrderedDict
from math import sqrt
from typing import Any, Callable, Iterator, List, Optional, Union

import e3nn
import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import spherical_harmonics
from e3nn.o3._tensor_product._codegen import (
    codegen_tensor_product_left_right,
    codegen_tensor_product_right,
)
from e3nn.o3._tensor_product._instruction import Instruction
from e3nn.util import prod
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
from opt_einsum_fx import optimize_einsums_full
from sympy.physics.wigner import wigner_6j
from torch import fx

from ..module_utils import SO3_Linear_e2former
from .base_tensor_product import Simple_TensorProduct_oTchannel, _sum_tensors, slices_basis


class DepthWiseTensorProduct_reducesameorder(Simple_TensorProduct_oTchannel):
    r"""Fully-connected weighted tensor product

    All the possible path allowed by :math:`|l_1 - l_2| \leq l_{out} \leq l_1 + l_2` are made.
    The output is a sum on different paths:

    .. math::

        z_w = \sum_{u,v} w_{uvw} x_u \otimes y_v + \cdots \text{other paths}

    where :math:`u,v,w` are the indices of the multiplicities.

    Parameters
    ----------
    irreps_in1 : `e3nn.o3.Irreps`
        representation of the first input

    irreps_in2 : `e3nn.o3.Irreps`
        representation of the second input

    irreps_out : `e3nn.o3.Irreps`
        representation of the output

    irrep_normalization : {'component', 'norm'}
        see `e3nn.o3.TensorProduct`

    path_normalization : {'element', 'path'}
        see `e3nn.o3.TensorProduct`

    internal_weights : bool
        see `e3nn.o3.TensorProduct`

    shared_weights : bool
        see `e3nn.o3.TensorProduct`
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        max_ir=None,
        irrep_normalization="none",
        path_normalization="none",
        connection_mode="uvu",
        learnable_weight=True,
        **kwargs,
    ):
        irreps_in1 = (
            o3.Irreps(irreps_in1) if isinstance(irreps_in1, str) else irreps_in1
        )
        irreps_in2 = (
            o3.Irreps(irreps_in2) if isinstance(irreps_in2, str) else irreps_in2
        )
        if max_ir is None:
            irreps_out = (
                o3.Irreps(irreps_out) if isinstance(irreps_out, str) else irreps_out
            )

            instr = []
            out_source = []
            # instead we do this separately
            for i_1, (mul_1, ir_1) in enumerate(irreps_in1):
                for i_2, (mul_2, ir_2) in enumerate(irreps_in2):
                    for i_out, (_, ir_out) in enumerate(irreps_out):
                        #     continue
                        if ir_out not in ir_1 * ir_2:
                            continue
                        ## this is the out index
                        instr += [(i_1, i_2, i_out, connection_mode, learnable_weight)]
                        out_source.append((ir_1.l, ir_2.l, ir_out.l))  # that is a,b,d
        else:
            instr = []
            out_source = []
            # instead we do this separately
            for i_1, (mul_1, ir_1) in enumerate(irreps_in1):
                for i_2, (mul_2, ir_2) in enumerate(irreps_in2):
                    for ir_out in ir_1 * ir_2:
                        #     continue
                        if ir_out.l > max_ir + max(irreps_in1.ls) - ir_2.l:
                            continue
                        ## this is the out index
                        instr += [
                            (i_1, i_2, ir_out.l, connection_mode, learnable_weight)
                        ]
                        out_source.append((ir_1.l, ir_2.l, ir_out.l))  # that is a,b,d
            max_out_order = max([i[2] for i in instr])
            irreps_out = "+".join(
                [
                    "{c}x0e",
                    "{c}x1e",
                    "{c}x2e",
                    "{c}x3e",
                    "{c}x4e",
                    "{c}x5e",
                    "{c}x6e",
                    "{c}x7e",
                    "{c}x8e",
                ][: max_out_order + 1]
            )
            irreps_out = irreps_out.format(c=mul_1 * mul_2)
            irreps_out = o3.Irreps(irreps_out)
        self.out_source = out_source

        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instr,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            **kwargs,
        )

        flat_weight_index = 0
        self.weights_dict = {}
        for ins in self.instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]
            if ins.has_weight:
                self.weights_dict[
                    (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
                ] = slice(flat_weight_index, flat_weight_index + prod(ins.path_shape))
                flat_weight_index += prod(ins.path_shape)

    def get_weight_byL1L2L3(self, L1, L2, L3):
        # L1, irreps_in1
        # L2, irreps_in2
        # L3, irreps_out

        return self.weights_dict[(L1, L2, L3)]


class DepthwiseTensorProduct_wosort(Simple_TensorProduct_oTchannel):
    r"""Full tensor product between two irreps.

    .. math::

        z_{uv} = x_u \otimes y_v

    where :math:`u` and :math:`v` run over the irreps. Note that there are no weights.
    The output representation is determined by the two input representations.

    Parameters
    ----------
    irreps_in1 : `e3nn.o3.Irreps`
        representation of the first input

    irreps_in2 : `e3nn.o3.Irreps`
        representation of the second input

    filter_ir_out : iterator of `e3nn.o3.Irrep`, optional
        filter to select only specific `e3nn.o3.Irrep` of the output

    irrep_normalization : {'component', 'norm'}
        see `e3nn.o3.TensorProduct`
    """

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        filter_ir_out: Iterator[o3.Irrep] = None,
        max_ir=1000,
        irrep_normalization=None,
        path_normalization=None,
        learnable_weight=False,
        connection_mode="uvu",
        **kwargs,
    ) -> None:
        irreps_in1 = o3.Irreps(irreps_in1).simplify()
        irreps_in2 = o3.Irreps(irreps_in2).simplify()
        if filter_ir_out is not None:
            try:
                filter_ir_out = [o3.Irrep(ir) for ir in filter_ir_out]
            except ValueError:
                raise ValueError(
                    f"filter_ir_out (={filter_ir_out}) must be an iterable of e3nn.o3.Irrep"
                )

        out = []
        instr = []
        out_source = []
        # instead we do this separately
        for i_1, (mul_1, ir_1) in enumerate(irreps_in1):
            for i_2, (mul_2, ir_2) in enumerate(irreps_in2):
                for ir_out in ir_1 * ir_2:
                    #     continue
                    if ir_out.l > max_ir + max(irreps_in1.ls) - ir_2.l:
                        continue
                    ## this is the out index
                    i_out = len(out)
                    out.append((mul_1 * mul_2, ir_out))
                    instr += [(i_1, i_2, i_out, connection_mode, learnable_weight)]
                    out_source.append((ir_1.l, ir_2.l, ir_out.l))  # that is a,b,d

        out = o3.Irreps(out)
        self.out_source = out_source
        super().__init__(
            irreps_in1,
            irreps_in2,
            out,
            instr,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            **kwargs,
        )


def CODEGEN_MAIN_LEFT_RIGHT(
    self__irreps_in1,
    self__irreps_in2,
    self__irreps_out,
    self__instructions,
    self__simulate_tp,
    self__info,
) -> fx.GraphModule:
    
    graph = fx.Graph()

    tracer = fx.proxy.GraphAppendingTracer(graph)
    constants = OrderedDict()

    x1s = fx.Proxy(graph.placeholder("x1", torch.Tensor), tracer=tracer)
    x2s = fx.Proxy(graph.placeholder("x2", torch.Tensor), tracer=tracer)
    weights = fx.Proxy(graph.placeholder("w", torch.Tensor), tracer=tracer)

    empty = fx.Proxy(
        graph.call_function(torch.empty, ((),), dict(device="cpu")), tracer=tracer
    )
    output_shape = x1s.shape[:-2]
    del empty

    
    x1s = x1s.reshape(
        -1, self__irreps_in1.dim // self__irreps_in1[0].mul, self__irreps_in1[0].mul
    )
    x2s = x2s.reshape(
        -1, self__irreps_in2.dim // self__irreps_in2[0].mul, self__irreps_in2[0].mul
    )
    batch_numel = x1s.shape[0]

    
    x1_list = [
        x1s[:, i].reshape(batch_numel, mul_ir.ir.dim, mul_ir.mul)
        for i, mul_ir in zip(slices_basis(self__irreps_in1), self__irreps_in1)
    ]
    x2_list = [
        x2s[:, i].reshape(batch_numel, mul_ir.ir.dim, mul_ir.mul)
        for i, mul_ir in zip(slices_basis(self__irreps_in2), self__irreps_in2)
    ]

    outputs = []
    flat_weight_index = 0

    
    for idx, ins in enumerate(self__instructions):
        mul_ir_in1 = self__irreps_in1[ins.i_in1]
        mul_ir_in2 = self__irreps_in2[ins.i_in2]
        mul_ir_out = self__irreps_out[ins.i_out]

        
        #     continue

        x1 = x1_list[ins.i_in1]
        x2 = x2_list[ins.i_in2]

        w3j_name = f"_w3j_{mul_ir_in1.ir.l}_{mul_ir_in2.ir.l}_{mul_ir_out.ir.l}"
        w3j = fx.Proxy(graph.get_attr(w3j_name), tracer=tracer)

        if ins.has_weight:
            if self__simulate_tp is not None:
                w = weights[
                    self__simulate_tp.get_weight_byL1L2L3(
                        self__info[idx][0], self__info[idx][1], self__info[idx][2]
                    )
                ].reshape(tuple(ins.path_shape))

            else:
                w = weights[
                    flat_weight_index : flat_weight_index + prod(ins.path_shape)
                ].reshape(tuple(ins.path_shape))

            flat_weight_index += prod(ins.path_shape)
        

        if ins.connection_mode == "uvw":
            xx = torch.einsum("ziu,zjv,ijk->zku", x1, x2, w3j)
            assert ins.has_weight  # and x2.shape[-1]==1
            w = w.squeeze()
            result = torch.matmul(xx, w)
        elif ins.connection_mode == "uvu":
            assert mul_ir_in1.mul == mul_ir_out.mul
            # not so useful operation because v is summed
            if ins.has_weight:
                xx = torch.einsum("ziu,zjv,ijk->zkuv", x1, x2, w3j)
                result = torch.einsum("uv,zkuv->zku", w, xx)
            else:
                result = torch.einsum("ziu,zjv,ijk->zku", x1, x2, w3j)

        

        result = ins.path_weight * result
        outputs.append(
            result.reshape(batch_numel, mul_ir_out.ir.l * 2 + 1, mul_ir_out.mul)
        )

        w3j_name = f"_w3j_{mul_ir_in1.ir.l}_{mul_ir_in2.ir.l}_{mul_ir_out.ir.l}"
        # Remove unused w3js:
        if len(w3j.node.users) == 0:
            graph.erase_node(w3j.node)
        else:
            if w3j_name not in constants:
                constants[w3j_name] = o3.wigner_3j(
                    mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l
                )

    
    outputs = [
        _sum_tensors(
            [
                out
                for ins, out in zip(self__instructions, outputs)
                if ins.i_out == i_out
            ],
            shape=(batch_numel, mul_ir_out.dim),
            like=x1s,
        )
        for i_out, mul_ir_out in enumerate(self__irreps_out)
        if mul_ir_out.mul > 0
    ]
    outputs = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
    outputs = outputs.reshape(output_shape + (outputs.shape[-2], outputs.shape[-1]))

    graph.output(outputs.node, torch.Tensor)

    # check graphs
    graph.lint()

    # Make GraphModules

    # By putting the constants in a Module rather than a dict,
    #
    constants_root = torch.nn.Module()
    for key, value in constants.items():
        constants_root.register_buffer(key, value)
    graphmod = fx.GraphModule(constants_root, graph, class_name="tp_forward")

    batchdim = 4
    example_inputs = (
        torch.zeros(
            (
                batchdim,
                self__irreps_in1.dim // self__irreps_in1[0].mul,
                self__irreps_in1[0].mul,
            )
        ),
        torch.zeros(
            (
                batchdim,
                self__irreps_in2.dim // self__irreps_in2[0].mul,
                self__irreps_in2[0].mul,
            )
        ),
        torch.zeros(
            flat_weight_index,
        ),
    )
    graphmod = optimize_einsums_full(graphmod, example_inputs)

    return graphmod


class FullyConnectedTensorProductWigner6j(Simple_TensorProduct_oTchannel):
    r"""Fully-connected weighted tensor product

    All the possible path allowed by :math:`|l_1 - l_2| \leq l_{out} \leq l_1 + l_2` are made.
    The output is a sum on different paths:

    .. math::

        z_w = \sum_{u,v} w_{uvw} x_u \otimes y_v + \cdots \text{other paths}

    where :math:`u,v,w` are the indices of the multiplicities.

    Parameters
    ----------
    irreps_in1 : `e3nn.o3.Irreps`
        representation of the first input

    irreps_in2 : `e3nn.o3.Irreps`
        representation of the second input

    irreps_out : `e3nn.o3.Irreps`
        representation of the output

    irrep_normalization : {'component', 'norm'}
        see `e3nn.o3.TensorProduct`

    path_normalization : {'element', 'path'}
        see `e3nn.o3.TensorProduct`

    internal_weights : bool
        see `e3nn.o3.TensorProduct`

    shared_weights : bool
        see `e3nn.o3.TensorProduct`
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        rij_order,
        irrep_normalization="none",
        path_normalization="none",
        previous_out_source=None,
        learnable_weight=False,
        connection_mode="uvu",
        simulate_tp=None,
        **kwargs,
    ):
        irreps_in1 = o3.Irreps(irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        irreps_out = o3.Irreps(irreps_out)

        # calculate the normalization factor
        self.ins = []
        self.info = []
        for i_1, (_, ir_1) in enumerate(irreps_in1):
            for i_2, (_, ir_2) in enumerate(irreps_in2):
                for i_out, (_, ir_out) in enumerate(irreps_out):
                    if ir_out in ir_1 * ir_2:
                        a, b, d = previous_out_source[i_1]
                        c = ir_2.l
                        abc = ir_out.l
                        if b + c != rij_order:
                            continue

                        bc = b + c

                        coefficient = math.comb(rij_order, b) * (-1) ** b
                        path_weight = coefficient * float(
                            wigner_6j(a, b, d, c, abc, bc)
                            * ((-1) ** (a + b + c + abc))
                            * math.sqrt((2 * d + 1) * (2 * bc + 1))
                        )
                        if path_weight != 0:
                            self.ins.append(
                                (
                                    i_1,
                                    i_2,
                                    i_out,
                                    connection_mode,
                                    learnable_weight,
                                    path_weight,
                                )
                            )

                            #                  a                        b+c   abc
                            self.info.append((a, bc, abc))

        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            self.ins,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            path_weight_sqrt=False,
            **kwargs,
        )
        self.simulate_tp = simulate_tp

        graphmod_left_right = CODEGEN_MAIN_LEFT_RIGHT(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            self.instructions,
            self.simulate_tp,
            self.info,
        )

        assert graphmod_left_right is not None
        self.weight = nn.Parameter(torch.ones(1))
        self._codegen_register({"_compiled_main_left_right": graphmod_left_right})

    def _main_left_right(self, x1s, x2s, weights):
        
        empty = torch.empty((), device="cpu")
        output_shape = torch.broadcast_tensors(
            empty.expand(x1s.shape[:-2]), empty.expand(x2s.shape[:-2])
        )[0].shape
        del empty

        
        x1s, x2s = x1s.broadcast_to(output_shape + (-1, -1)), x2s.broadcast_to(
            output_shape + (-1, -1)
        )

        x1s = x1s.reshape(output_shape.numel(), -1, self.irreps_in1[0].mul)
        x2s = x2s.reshape(output_shape.numel(), -1, self.irreps_in2[0].mul)
        batch_numel = x1s.shape[0]

        
        x1_list = [
            x1s[:, i].reshape(batch_numel, mul_ir.ir.dim, mul_ir.mul)
            for i, mul_ir in zip(slices_basis(self.irreps_in1), self.irreps_in1)
        ]
        x2_list = [
            x2s[:, i].reshape(batch_numel, mul_ir.ir.dim, mul_ir.mul)
            for i, mul_ir in zip(slices_basis(self.irreps_in2), self.irreps_in2)
        ]

        outputs = []
        flat_weight_index = 0

        
        for idx, ins in enumerate(self.instructions):
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]

            
            #     continue

            x1 = x1_list[ins.i_in1]
            x2 = x2_list[ins.i_in2]

            
            w3j = o3.wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l).to(
                x1s.device
            )

            if ins.has_weight:
                if self.simulate_tp is not None:
                    w = weights[
                        self.simulate_tp.get_weight_byL1L2L3(
                            self.info[idx][0], self.info[idx][1], self.info[idx][2]
                        )
                    ].reshape((1,) + tuple(ins.path_shape))

                else:
                    w = weights[
                        flat_weight_index : flat_weight_index + prod(ins.path_shape)
                    ].reshape((1,) + tuple(ins.path_shape))

                flat_weight_index += prod(ins.path_shape)
            

            if ins.connection_mode == "uvw":
                xx = torch.einsum("ziu,zjv,ijk->zku", x1, x2, w3j)
                assert ins.has_weight and x2.shape[-1] == 1
                w = w.squeeze()
                result = torch.matmul(xx, w)
            if ins.connection_mode == "uvu":
                xx = torch.einsum("ziu,zjv,ijk->zkuv", x1, x2, w3j)
                assert mul_ir_in1.mul == mul_ir_out.mul
                # not so useful operation because v is summed
                if ins.has_weight:
                    result = torch.einsum("zuv,zkuv->zku", w, xx)
                else:
                    result = torch.sum(xx, dim=-1)

            

            result = ins.path_weight * result
            outputs.append(
                result.reshape(batch_numel, mul_ir_out.ir.l * 2 + 1, mul_ir_out.mul)
            )

        
        outputs = [
            _sum_tensors(
                [
                    out
                    for ins, out in zip(self.instructions, outputs)
                    if ins.i_out == i_out
                ],
                shape=(batch_numel, mul_ir_out.dim),
                like=x1s,
            )
            for i_out, mul_ir_out in enumerate(self.irreps_out)
            if mul_ir_out.mul > 0
        ]

        outputs = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
        return outputs.reshape(output_shape + (outputs.shape[-2], outputs.shape[-1]))

    def forward(self, x, y, weight: Optional[torch.Tensor] = None):
        assert x.shape[-2:].numel() == self._in1_dim, "Incorrect last dimension for x"
        assert y.shape[-2:].numel() == self._in2_dim, "Incorrect last dimension for y"

        weight = self.simulate_tp.weight
        return self._compiled_main_left_right(x, y, weight)


def get_path_norm(
    irreps_in1="16x0e+16x1e+16x2e", irreps_in2="1x2e", irreps_out="16x0e+16x1e+16x2e"
):
    irreps_in1 = e3nn.o3.Irreps(irreps_in1)
    irreps_in2 = e3nn.o3.Irreps(irreps_in2)
    irreps_out = e3nn.o3.Irreps(irreps_out)
    counter = {}
    for i_1, (_, ir_1) in enumerate(irreps_in1):
        for i_2, (_, ir_2) in enumerate(irreps_in2):
            for i_out, (_, ir_out) in enumerate(irreps_out):
                if ir_out in ir_1 * ir_2:
                    counter[ir_out[0]] = counter.get(ir_out[0], 0) + 1
    buffer = []
    for mul, ir in irreps_out:
        buffer.append(torch.ones(2 * ir[0] + 1) * counter.get(ir[0], 0))
    return torch.cat(buffer, dim=0)


def get_tp_e3nnsh_diff():
    Y = torch.randn(1, 4, 3)

    self__square_tp = DepthWiseTensorProduct_reducesameorder(
        "1x1e",
        "1x1e",
        "1x2e",
        irrep_normalization="component",
        path_normalization="none",
    )
    self__square_tp.weight = torch.nn.Parameter(
        torch.ones(self__square_tp.weight.size()), requires_grad=False
    )
    Y_sq = self__square_tp(Y, Y)  # batch_size \times m time 5

    self__tri_tp = DepthWiseTensorProduct_reducesameorder(
        "1x2e",
        "1x1e",
        "1x3e",
        irrep_normalization="component",
        path_normalization="none",
    )
    self__tri_tp.weight = torch.nn.Parameter(
        torch.ones(self__tri_tp.weight.size()), requires_grad=False
    )
    Y_tr = self__tri_tp(Y_sq, Y)  # batch_size \times m time 5

    self__4_tp = DepthWiseTensorProduct_reducesameorder(
        "1x3e",
        "1x1e",
        "1x4e",
        irrep_normalization="component",
        path_normalization="none",
    )
    self__4_tp.weight = torch.nn.Parameter(
        torch.ones(self__4_tp.weight.size()), requires_grad=False
    )
    Y_4 = self__4_tp(Y_tr, Y)  # batch_size \times m time 5

    self__5_tp = DepthWiseTensorProduct_reducesameorder(
        "1x4e",
        "1x1e",
        "1x5e",
        irrep_normalization="component",
        path_normalization="none",
    )
    self__5_tp.weight = torch.nn.Parameter(
        torch.ones(self__5_tp.weight.size()), requires_grad=False
    )
    Y_5 = self__5_tp(Y_4, Y)  # batch_size \times m time 5

    self__6_tp = DepthWiseTensorProduct_reducesameorder(
        "1x5e",
        "1x1e",
        "1x6e",
        irrep_normalization="component",
        path_normalization="none",
    )
    self__6_tp.weight = torch.nn.Parameter(
        torch.ones(self__6_tp.weight.size()), requires_grad=False
    )
    Y_6 = self__6_tp(Y_5, Y)  # batch_size \times m time 5

    print(
        Y_sq
        / (e3nn.o3.spherical_harmonics(2, Y, normalize=False, normalization="integral"))
    )  # 1.29441716
    print(
        Y_tr
        / (e3nn.o3.spherical_harmonics(3, Y, normalize=False, normalization="integral"))
    )  # 0.84739512
    print(
        Y_4
        / (e3nn.o3.spherical_harmonics(4, Y, normalize=False, normalization="integral"))
    )  # 0.56493002
    print(
        Y_5
        / (e3nn.o3.spherical_harmonics(5, Y, normalize=False, normalization="integral"))
    )  # 0.38087577
    print(
        Y_6
        / (e3nn.o3.spherical_harmonics(6, Y, normalize=False, normalization="integral"))
    )  # 0.25875416


class E2TensorProductArbitraryOrder_woequal(torch.nn.Module):
    """Equivariant tensor product layer that can handle arbitrary order spherical harmonics.

    This module implements a generalized version of the E2TensorProduct that can work with
    any order of spherical harmonics. It combines multiple tensor products and Wigner 6-j
    symbols to maintain equivariance.

    Args:
        irreps_in (str): Input irreducible representations specification
        irreps_out (str): Output irreducible representations specification
        head (int): Number of attention heads
        order (int): Order of spherical harmonics to use
        learnable_weight (bool, optional): Whether weights should be learnable. Defaults to True.
        connection_mode (str, optional): Connection mode - 'uvw' or 'uvu'. Defaults to 'uvw'.
        path_normalization (str, optional): Type of path normalization. Defaults to 'element'.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        head,
        order,
        learnable_weight=True,
        connection_mode="uvw",
        path_normalization="element",
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.order = order
        self.in_c = o3.Irreps(self.irreps_in)[0][0]
        self.out_c = o3.Irreps(self.irreps_out)[0][0]
        self.lmax = e3nn.o3.Irreps(irreps_in)[-1][1][0]
        assert connection_mode in [
            "uvw",
            "uvu",
        ], "connection_mode must be either 'uvw' or 'uvu'"
        if not learnable_weight:
            connection_mode = "uvu"

        self.tensor_product_tp_component_1 = DepthWiseTensorProduct_reducesameorder(
            irreps_in,
            f"1x{order}e",
            irreps_out,
            irrep_normalization="component",
            path_normalization="none",
            learnable_weight=learnable_weight,
            connection_mode=connection_mode,
        )

        self.head = head

        tmp_ri_irreps = "+".join(["1x0e", "1x1e", "1x2e", "1x3e", "1x4e"][: order + 1])

        if learnable_weight is False:
            self.tp_without_sort = DepthWiseTensorProduct_reducesameorder(
                irreps_in,
                o3.Irreps(tmp_ri_irreps),
                None,
                max_ir=self.order,  # self.order,
                irrep_normalization="component",
                path_normalization="none",
                learnable_weight=False,
            )
        else:
            self.tp_without_sort = DepthwiseTensorProduct_wosort(
                irreps_in,
                o3.Irreps(tmp_ri_irreps),
                max_ir=self.order,  # self.order,
                irrep_normalization="component",
                path_normalization="none",
                learnable_weight=False,
            )
        # Create Wigner 6j tensor product
        self.wigner_6j_tp = FullyConnectedTensorProductWigner6j(
            self.tp_without_sort.irreps_out,
            o3.Irreps(tmp_ri_irreps),
            irreps_out,
            rij_order=self.order,  # self.order,
            previous_out_source=self.tp_without_sort.out_source,
            irrep_normalization="component",
            path_normalization="none",
            learnable_weight=learnable_weight,
            connection_mode=connection_mode,
            simulate_tp=self.tensor_product_tp_component_1,
        )

        self.coeffs = self.get_coeffs()
        if order > 6:
            raise ValueError("Coeffs for order > 6 are not implemented")

        # Setup path normalization
        if path_normalization == "element" or path_normalization is None:
            path_norm = 1 / torch.sqrt(
                get_path_norm(irreps_in, f"1x{order}e", irreps_in).reshape(1, -1, 1)
            )
            self.register_buffer("path_norm", path_norm)
        else:
            self.register_buffer("path_norm", torch.ones(1))

    @staticmethod
    def get_coeffs():
        #       0,   1               2           3           4           5           6
        return [
            1,
            2.046653509140,
            1.29441716,
            0.84739512,
            0.56493002,
            0.38087577,
            0.25875416,
        ]

    def forward(
        self,
        pos,
        exp_pos,
        h,
        exp_h,
        alpha_ij,
        f_sparse_idx_expnode=None,
        batched_data={},
    ):  # f_sparse_idx_expnode
        """Forward pass of the layer.

        Args:
            pos (torch.Tensor): B*L2
            h (torch.Tensor): B*L2
            alpha_ij (torch.Tensor): B*L1*L2/topK, when topK

        Returns:
            torch.Tensor: Transformed features
        """
        f_N1, topK = alpha_ij.shape[:2]
        f_N2 = exp_pos.shape[0]
        if "Y_powers" in batched_data:
            Y_powers = batched_data["Y_powers"]
            exp_Y_powers = batched_data["exp_Y_powers"]

        else:
            Y_powers = []
            for i in range(self.order + 1):
                if i == 0:
                    Y_powers.append(
                        self.coeffs[i]
                        * torch.ones_like(pos.narrow(-1, 0, 1).unsqueeze(dim=-1))
                    )
                else:
                    Y_powers.append(
                        self.coeffs[i]
                        * e3nn.o3.spherical_harmonics(
                            i, pos, normalize=False, normalization="integral"
                        ).unsqueeze(-1)
                    )

            exp_Y_powers = []
            for i in range(self.order + 1):
                if i == 0:
                    exp_Y_powers.append(
                        self.coeffs[i]
                        * torch.ones_like(exp_pos.narrow(-1, 0, 1).unsqueeze(dim=-1))
                    )
                else:
                    exp_Y_powers.append(
                        self.coeffs[i]
                        * e3nn.o3.spherical_harmonics(
                            i, exp_pos, normalize=False, normalization="integral"
                        ).unsqueeze(-1)
                    )

            Y_powers = torch.cat(Y_powers, dim=1)
            exp_Y_powers = torch.cat(exp_Y_powers, dim=1)

        component_1 = self.tp_without_sort(
            exp_h, exp_Y_powers[:, : (self.order + 1) ** 2]
        )
        component_1 = component_1.reshape(f_N2, -1, self.head, self.in_c // self.head)

        if f_sparse_idx_expnode is not None:
            component_1 = torch.sum(
                alpha_ij.unsqueeze(dim=2).unsqueeze(dim=-1)
                * component_1[f_sparse_idx_expnode],
                dim=1,
            )
        else:
            component_1 = torch.einsum("bjh,johk -> bohk", alpha_ij, component_1)
        component_1 = component_1.reshape(f_N1, -1, self.in_c)

        out = self.wigner_6j_tp(component_1, Y_powers[:, : (self.order + 1) ** 2])

        return out * self.path_norm

    def vanilla_forward(
        self, pos, exp_pos, h, exp_h, alpha_ij, f_sparse_idx_expnode=None
    ):
        """Simple forward pass without component decomposition.

        This method provides a simpler implementation for comparison and testing.

        Args:
            pos (torch.Tensor): Position coordinates
            h (torch.Tensor): Input features
            alpha_ij (torch.Tensor): Attention weights

        Returns:
            torch.Tensor: Transformed features
        """
        f_N1, topK = alpha_ij.shape[:2]
        f_N2 = exp_pos.shape[0]

        delta_pos = pos.unsqueeze(dim=1) - exp_pos.unsqueeze(dim=0)
        delta_pos_order_l = (
            e3nn.o3.spherical_harmonics(
                self.order, delta_pos, normalize=False, normalization="integral"
            )
            * self.coeffs[self.order]
        )
        delta_pos_order_l = delta_pos_order_l.unsqueeze(dim=-1)

        h_new = exp_h.reshape(f_N2, -1, self.head, self.in_c // self.head)
        h_new = torch.einsum("bjh, johk -> bjohk", alpha_ij, h_new)
        h_new = h_new.reshape(f_N1, f_N2, -1, self.in_c)
        out_new = self.tensor_product_tp_component_1(h_new, delta_pos_order_l)
        out_new = torch.sum(out_new, dim=1)
        return out_new * self.path_norm


class E2TensorProduct_FirstOrder(torch.nn.Module):
    """Equivariant tensor product layer that can handle arbitrary order spherical harmonics.

    This module implements a generalized version of the E2TensorProduct that can work with
    any order of spherical harmonics. It combines multiple tensor products and Wigner 6-j
    symbols to maintain equivariance.

    Args:
        irreps_in (str): Input irreducible representations specification
        irreps_out (str): Output irreducible representations specification
        head (int): Number of attention heads
        order (int): Order of spherical harmonics to use
        learnable_weight (bool, optional): Whether weights should be learnable. Defaults to True.
        connection_mode (str, optional): Connection mode - 'uvw' or 'uvu'. Defaults to 'uvw'.
        path_normalization (str, optional): Type of path normalization. Defaults to 'element'.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        head,
        order,
        learnable_weight=True,
        connection_mode="uvw",
        path_normalization="element",
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.order = order
        self.in_c = o3.Irreps(self.irreps_in)[0][0]
        self.out_c = o3.Irreps(self.irreps_out)[0][0]
        self.lmax = e3nn.o3.Irreps(irreps_in)[-1][1][0]
        self.head = head
        assert connection_mode in [
            "uvw",
            "uvu",
        ], "connection_mode must be either 'uvw' or 'uvu'"
        if not learnable_weight:
            connection_mode = "uvu"

        if connection_mode == "uvw":
            warnings.warn(
                " sorry current this first order functio use w1*w2 to simulate uvw mode"
            )
            connection_mode = "uvu"
            self.w1 = SO3_Linear_e2former(self.in_c, self.in_c, lmax=self.lmax)
            self.w2 = SO3_Linear_e2former(self.in_c, self.out_c, lmax=self.lmax)
        else:
            self.w1 = nn.Identity()
            self.w2 = nn.Identity()

        if order != 1:
            raise ValueError(
                "sorry, E2TensorProduct_FirstOrder function only  support order rij with order 1"
            )
        self.tensor_product_tp_component_1 = DepthWiseTensorProduct_reducesameorder(
            irreps_in,
            f"1x{1}e",
            irreps_out,
            irrep_normalization="component",
            path_normalization="none",
            learnable_weight=learnable_weight,
            connection_mode=connection_mode,
        )

        self.coeffs = self.get_coeffs()
        # Setup path normalization
        if path_normalization == "element" or path_normalization is None:
            path_norm = 1 / torch.sqrt(
                get_path_norm(irreps_in, f"1x{order}e", irreps_in).reshape(1, -1, 1)
            )
            self.register_buffer("path_norm", path_norm)
        else:
            self.register_buffer("path_norm", torch.ones(1))

    @staticmethod
    def get_coeffs():
        #       0,   1               2           3           4           5           6
        return [
            1,
            2.046653509140,
            1.29441716,
            0.84739512,
            0.56493002,
            0.38087577,
            0.25875416,
        ]

    def forward(
        self,
        pos,
        exp_pos,
        h,
        exp_h,
        alpha_ij,
        f_sparse_idx_expnode=None,
        batched_data={},
    ):  # f_sparse_idx_expnode
        """Forward pass of the layer.

        Args:
            pos (torch.Tensor): B*L2
            h (torch.Tensor): B*L2
            alpha_ij (torch.Tensor): B*L1*L2/topK, when topK

        Returns:
            torch.Tensor: Transformed features
        """
        exp_h = self.w1(exp_h)
        f_N1, topK = alpha_ij.shape[:2]
        f_N2 = exp_pos.shape[0]
        if "Y_powers" in batched_data:
            Y_powers = batched_data["Y_powers"]
            exp_Y_powers = batched_data["exp_Y_powers"]

        else:
            Y_powers = []
            for i in range(self.order + 1):
                if i == 0:
                    Y_powers.append(
                        self.coeffs[i]
                        * torch.ones_like(pos.narrow(-1, 0, 1).unsqueeze(dim=-1))
                    )
                else:
                    Y_powers.append(
                        self.coeffs[i]
                        * e3nn.o3.spherical_harmonics(
                            i, pos, normalize=False, normalization="integral"
                        ).unsqueeze(-1)
                    )

            exp_Y_powers = []
            for i in range(self.order + 1):
                if i == 0:
                    exp_Y_powers.append(
                        self.coeffs[i]
                        * torch.ones_like(exp_pos.narrow(-1, 0, 1).unsqueeze(dim=-1))
                    )
                else:
                    exp_Y_powers.append(
                        self.coeffs[i]
                        * e3nn.o3.spherical_harmonics(
                            i, exp_pos, normalize=False, normalization="integral"
                        ).unsqueeze(-1)
                    )

        # # Compute main component
        component_1 = exp_h.reshape(
            f_N2, (self.lmax + 1) ** 2, self.head, self.in_c // self.head
        )
        if f_sparse_idx_expnode is not None:
            component_1 = torch.sum(
                alpha_ij.unsqueeze(dim=2).unsqueeze(dim=-1)
                * component_1[f_sparse_idx_expnode],
                dim=1,
            )
        else:
            component_1 = torch.einsum("bjh,johk -> bohk", alpha_ij, component_1)

        component_1 = component_1.reshape(f_N1, (self.lmax + 1) ** 2, self.in_c)
        component_1 = self.tensor_product_tp_component_1(
            component_1, Y_powers[self.order]
        )

        component_2 = self.tensor_product_tp_component_1(
            exp_h, exp_Y_powers[self.order]
        )
        component_2 = component_2.reshape(
            f_N2, -1, self.head, component_2.shape[-1] // self.head
        )
        if f_sparse_idx_expnode is not None:
            component_2 = torch.sum(
                alpha_ij.unsqueeze(dim=2).unsqueeze(dim=-1)
                * component_2[f_sparse_idx_expnode],
                dim=1,
            )
        else:
            component_2 = torch.einsum("bjh,johk -> bohk", alpha_ij, component_2)
        component_2 = component_2.reshape(f_N1, -1, component_2.shape[-2:].numel())

        return self.w2(component_1 - component_2) * self.path_norm

    def vanilla_forward(
        self, pos, exp_pos, h, exp_h, alpha_ij, f_sparse_idx_expnode=None
    ):
        """Simple forward pass without component decomposition.

        This method provides a simpler implementation for comparison and testing.

        Args:
            pos (torch.Tensor): Position coordinates
            h (torch.Tensor): Input features
            alpha_ij (torch.Tensor): Attention weights

        Returns:
            torch.Tensor: Transformed features
        """
        exp_h = self.w1(exp_h)
        f_N1, topK = alpha_ij.shape[:2]
        f_N2 = exp_pos.shape[0]

        delta_pos = pos.unsqueeze(dim=1) - exp_pos.unsqueeze(dim=0)
        delta_pos_order_l = (
            e3nn.o3.spherical_harmonics(
                self.order, delta_pos, normalize=False, normalization="integral"
            )
            * self.coeffs[self.order]
        )
        delta_pos_order_l = delta_pos_order_l.unsqueeze(dim=-1)

        h_new = exp_h.reshape(f_N2, -1, self.head, self.in_c // self.head)
        h_new = torch.einsum("bjh, johk -> bjohk", alpha_ij, h_new)
        h_new = h_new.reshape(f_N1, f_N2, -1, self.in_c)
        out_new = self.tensor_product_tp_component_1(h_new, delta_pos_order_l)
        out_new = torch.sum(out_new, dim=1)
        return self.w2(out_new) * self.path_norm
      


class E2TensorProductArbitraryOrder(torch.nn.Module):
    """Equivariant tensor product layer that can handle arbitrary order spherical harmonics.

    This module implements a generalized version of the E2TensorProduct that can work with
    any order of spherical harmonics. It combines multiple tensor products and Wigner 6-j
    symbols to maintain equivariance.

    Args:
        irreps_in (str): Input irreducible representations specification
        irreps_out (str): Output irreducible representations specification
        head (int): Number of attention heads
        order (int): Order of spherical harmonics to use
        learnable_weight (bool, optional): Whether weights should be learnable. Defaults to True.
        connection_mode (str, optional): Connection mode - 'uvw' or 'uvu'. Defaults to 'uvw'.
        path_normalization (str, optional): Type of path normalization. Defaults to 'element'.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        head,
        order,
        learnable_weight=True,
        connection_mode="uvw",
        path_normalization="element",
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.order = order
        self.in_c = o3.Irreps(self.irreps_in)[0][0]
        self.out_c = o3.Irreps(self.irreps_out)[0][0]
        self.lmax = e3nn.o3.Irreps(irreps_in)[-1][1][0]
        assert connection_mode in [
            "uvw",
            "uvu",
        ], "connection_mode must be either 'uvw' or 'uvu'"
        if not learnable_weight:
            connection_mode = "uvu"

        self.tensor_product_tp_component_1 = DepthWiseTensorProduct_reducesameorder(
            irreps_in,
            f"1x{order}e",
            irreps_out,
            irrep_normalization="component",
            path_normalization="none",
            learnable_weight=learnable_weight,
            connection_mode=connection_mode,
        )

        # Setup attention mechanism
        e3nn.o3.Irreps(
            [(mul // head, (ir, p)) for mul, (ir, p) in e3nn.o3.Irreps(irreps_in)]
        )
        self.head = head

        self.components = nn.ModuleList(
            [
                self._create_component(i, learnable_weight, connection_mode)
                for i in range(1, order + 1)
            ]
        )

        self.coeffs = self.get_coeffs()
        if order > 6:
            raise ValueError("Coeffs for order > 6 are not implemented")

        # Setup path normalization
        if path_normalization == "element" or path_normalization is None:
            path_norm = 1 / torch.sqrt(
                get_path_norm(irreps_in, f"1x{order}e", irreps_in).reshape(1, -1, 1)
            )
            self.register_buffer("path_norm", path_norm)
        else:
            self.register_buffer("path_norm", torch.ones(1))

    def _create_component(self, i, learnable_weight, connection_mode):
        """Creates a component for processing spherical harmonics of order i.

        Args:
            i (int): Order of spherical harmonics
            learnable_weight (bool): Whether weights should be learnable
            connection_mode (str): Connection mode for tensor products

        Returns:
            ModuleDict: Dictionary containing the component's neural network modules
        """
        # Create tensor product without sorting
        tp_without_sort = DepthwiseTensorProduct_wosort(
            self.irreps_in,
            o3.Irreps(f"1x{i}e"),
            max_ir=e3nn.o3.Irreps(self.irreps_in)[-1][1].l + (self.order - i),
            irrep_normalization="component",
            path_normalization="none",
            learnable_weight=False,
        )

        e3nn.o3.Irreps(
            [(mul // self.head, (ir, p)) for mul, (ir, p) in tp_without_sort.irreps_out]
        )

        # Create Wigner 6j tensor product
        wigner_6j_tp = FullyConnectedTensorProductWigner6j(
            tp_without_sort.irreps_out,
            o3.Irreps(f"1x{self.order-i}e"),
            self.irreps_out,
            rij_order=self.order,  # self.order,
            previous_out_source=tp_without_sort.out_source,
            irrep_normalization="component",
            path_normalization="none",
            learnable_weight=learnable_weight,
            connection_mode=connection_mode,
            simulate_tp=self.tensor_product_tp_component_1,
        )

        return nn.ModuleDict(
            {"tp_without_sort": tp_without_sort, "wigner_6j_tp": wigner_6j_tp}
        )

    @staticmethod
    def get_coeffs():
        #       0,   1               2           3           4           5           6
        return [
            1,
            2.046653509140,
            1.29441716,
            0.84739512,
            0.56493002,
            0.38087577,
            0.25875416,
        ]

    def forward(
        self,
        pos,
        exp_pos,
        h,
        exp_h,
        alpha_ij,
        f_sparse_idx_expnode=None,
        batched_data={},
    ):  # f_sparse_idx_expnode
        """Forward pass of the layer.

        Args:
            pos (torch.Tensor): B*L2
            h (torch.Tensor): B*L2
            alpha_ij (torch.Tensor): B*L1*L2/topK, when topK

        Returns:
            torch.Tensor: Transformed features
        """
        f_N1, topK = alpha_ij.shape[:2]
        f_N2 = exp_pos.shape[0]
        if "Y_powers" in batched_data:
            Y_powers = batched_data["Y_powers"]
            exp_Y_powers = batched_data["exp_Y_powers"]

        else:
            Y_powers = []
            for i in range(self.order + 1):
                if i == 0:
                    Y_powers.append(
                        self.coeffs[i]
                        * torch.ones_like(pos.narrow(-1, 0, 1).unsqueeze(dim=-1))
                    )
                else:
                    Y_powers.append(
                        self.coeffs[i]
                        * e3nn.o3.spherical_harmonics(
                            i, pos, normalize=False, normalization="integral"
                        ).unsqueeze(-1)
                    )

            exp_Y_powers = []
            for i in range(self.order + 1):
                if i == 0:
                    exp_Y_powers.append(
                        self.coeffs[i]
                        * torch.ones_like(exp_pos.narrow(-1, 0, 1).unsqueeze(dim=-1))
                    )
                else:
                    exp_Y_powers.append(
                        self.coeffs[i]
                        * e3nn.o3.spherical_harmonics(
                            i, exp_pos, normalize=False, normalization="integral"
                        ).unsqueeze(-1)
                    )

        # # Compute main component
        component_1 = exp_h.reshape(
            f_N2, (self.lmax + 1) ** 2, self.head, self.in_c // self.head
        )
        if f_sparse_idx_expnode is not None:
            component_1 = torch.sum(
                alpha_ij.unsqueeze(dim=2).unsqueeze(dim=-1)
                * component_1[f_sparse_idx_expnode],
                dim=1,
            )
        else:
            component_1 = torch.einsum("bjh,johk -> bohk", alpha_ij, component_1)

        component_1 = component_1.reshape(f_N1, (self.lmax + 1) ** 2, self.in_c)
        component_1 = self.tensor_product_tp_component_1(
            component_1, Y_powers[self.order]
        )

        # Compute additional components
        out = component_1
        for i, component in enumerate(self.components):
            k = i + 1
            c = component["tp_without_sort"](exp_h, exp_Y_powers[k])
            c = c.reshape(f_N2, -1, self.head, c.shape[-1] // self.head)
            if f_sparse_idx_expnode is not None:
                c = torch.sum(
                    alpha_ij.unsqueeze(dim=2).unsqueeze(dim=-1)
                    * c[f_sparse_idx_expnode],
                    dim=1,
                )
            else:
                c = torch.einsum("bjh,johk -> bohk", alpha_ij, c)

            c = c.reshape(f_N1, -1, c.shape[-2:].numel())
            c = component["wigner_6j_tp"](c, Y_powers[self.order - k])

            out = out + c

        return out * self.path_norm

    def vanilla_forward(
        self, pos, exp_pos, h, exp_h, alpha_ij, f_sparse_idx_expnode=None
    ):
        """Simple forward pass without component decomposition.

        This method provides a simpler implementation for comparison and testing.

        Args:
            pos (torch.Tensor): Position coordinates
            h (torch.Tensor): Input features
            alpha_ij (torch.Tensor): Attention weights

        Returns:
            torch.Tensor: Transformed features
        """
        f_N1, topK = alpha_ij.shape[:2]
        f_N2 = exp_pos.shape[0]

        delta_pos = pos.unsqueeze(dim=1) - exp_pos.unsqueeze(dim=0)
        delta_pos_order_l = (
            e3nn.o3.spherical_harmonics(
                self.order, delta_pos, normalize=False, normalization="integral"
            )
            * self.coeffs[self.order]
        )
        delta_pos_order_l = delta_pos_order_l.unsqueeze(dim=-1)

        h_new = exp_h.reshape(f_N2, -1, self.head, self.in_c // self.head)
        h_new = torch.einsum("bjh, johk -> bjohk", alpha_ij, h_new)
        h_new = h_new.reshape(f_N1, f_N2, -1, self.in_c)
        out_new = self.tensor_product_tp_component_1(h_new, delta_pos_order_l)
        out_new = torch.sum(out_new, dim=1)
        return out_new * self.path_norm

if __name__ == '__main__':
    import torch


    for o in range(1, 4):
        head, hidden = 8, 16
        f_N1, f_N2 = 7, 8
        alpha_ij = torch.randn(f_N1, f_N2, head)
        h = torch.randn(f_N1, (o + 1) ** 2, head * hidden)
        exp_h = torch.randn(f_N2, (o + 1) ** 2, head * hidden)
        pos = torch.randn(f_N1, 3)
        exp_pos = torch.randn(f_N2, 3)

        irreps_in = "+".join(
            [
                f"{head*hidden}x0e",
                f"{head*hidden}x1e",
                f"{head*hidden}x2e",
                f"{head*hidden}x3e",
                f"{head*hidden}x4e",
            ][: o + 1]
        )
        irreps_out = irreps_in

        learnable_weight = True
        connection_mode = "uvw"

        model_arbitrary = E2TensorProductArbitraryOrder(
            irreps_in,
            irreps_out,
            head,
            order=o,
            learnable_weight=learnable_weight,
            connection_mode=connection_mode,
            path_normalization="element",
        )
        out_arbitrary = model_arbitrary(pos, exp_pos, h, exp_h, alpha_ij)

        out_second = model_arbitrary.vanilla_forward(pos, exp_pos, h, exp_h, alpha_ij)
        # Print comparison metrics
        diff = out_arbitrary / out_second
        print(f"\nComparing Arbitrary Order (n={o}) vs Second Order:")
        print(f"Max difference: {torch.max(diff):.8f}")
        print(f"Mean difference: {torch.mean(diff):.8f}")
        print(f"Min difference: {torch.min(diff):.8f}")
