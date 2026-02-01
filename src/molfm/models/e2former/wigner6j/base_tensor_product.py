# -*- coding: utf-8 -*-
from collections import OrderedDict
from math import sqrt
from typing import List, Optional, Union

import torch
from e3nn import o3
from e3nn.o3._tensor_product._instruction import Instruction
from e3nn.util import prod
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
from opt_einsum_fx import optimize_einsums_full
from torch import fx, nn


def _sum_tensors(xs: List[torch.Tensor], shape: torch.Size, like: torch.Tensor):
    if len(xs) > 0:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out
    return like.new_zeros(shape)


def slices_basis(irreps):
    r"""List of slices corresponding to indices for each irrep.

    Examples
    --------

    >>> Irreps('2x0e + 1e').slices()
    [slice(0, 2, None), slice(2, 5, None)]
    """
    s = []
    i = 0
    for mul_ir in irreps:
        s.append(slice(i, i + mul_ir[1][0] * 2 + 1))
        i += mul_ir[1][0] * 2 + 1
    return s


def CODEGEN_MAIN_LEFT_RIGHT(
    self__irreps_in1,
    self__irreps_in2,
    self__irreps_out,
    self__instructions,
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
        elif ins.connection_mode == "uuu":
            result = torch.einsum("ziu,zju,ijk->zku", x1, x2, w3j)

        

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


class Simple_TensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: List[tuple],
        in1_var: Optional[Union[List[float], torch.Tensor]] = None,
        in2_var: Optional[Union[List[float], torch.Tensor]] = None,
        out_var: Optional[Union[List[float], torch.Tensor]] = None,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights=True,
        path_weight_sqrt=True,
        rescale=True,
        use_bias=False,
    ):
        super().__init__()
        self.rescale = (rescale,)

        self.use_bias = use_bias
        if self.use_bias:
            raise ValueError(
                "Not implemented yet, the bias is for order 0 irreps only, it  only works for TensorProductRescale"
            )
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)

        instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]
        instructions = [
            Instruction(
                i_in1=i_in1,
                i_in2=i_in2,
                i_out=i_out,
                connection_mode=connection_mode,
                has_weight=has_weight,
                path_weight=path_weight,
                path_shape={
                    "uvw": (
                        self.irreps_in1[i_in1].mul,
                        self.irreps_in2[i_in2].mul,
                        self.irreps_out[i_out].mul,
                    ),
                    "uvu": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uvv": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uuw": (self.irreps_in1[i_in1].mul, self.irreps_out[i_out].mul),
                    "uuu": (self.irreps_in1[i_in1].mul,),
                    "uvuv": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uvu<v": (
                        self.irreps_in1[i_in1].mul
                        * (self.irreps_in2[i_in2].mul - 1)
                        // 2,
                    ),
                    "u<vw": (
                        self.irreps_in1[i_in1].mul
                        * (self.irreps_in2[i_in2].mul - 1)
                        // 2,
                        self.irreps_out[i_out].mul,
                    ),
                }[connection_mode],
            )
            for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in instructions
        ]

        if in1_var is None:
            in1_var = [1.0 for _ in range(len(self.irreps_in1))]

        if in2_var is None:
            in2_var = [1.0 for _ in range(len(self.irreps_in2))]

        if out_var is None:
            out_var = [1.0 for _ in range(len(self.irreps_out))]

        def num_elements(ins):
            return {
                "uvw": (
                    self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul
                ),
                "uvu": self.irreps_in2[ins.i_in2].mul,
                "uuu": 1,
            }[ins.connection_mode]

        normalization_coefficients = []
        for ins in instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]

            if irrep_normalization == "component":
                alpha = mul_ir_out.ir.dim
            if irrep_normalization == "norm":
                alpha = mul_ir_in1.ir.dim * mul_ir_in2.ir.dim
            if irrep_normalization == "none":
                alpha = 1

            if path_normalization == "element":
                x = sum(
                    in1_var[i.i_in1] * in2_var[i.i_in2] * num_elements(i)
                    for i in instructions
                    if i.i_out == ins.i_out
                )
            if path_normalization == "path":
                x = in1_var[ins.i_in1] * in2_var[ins.i_in2] * num_elements(ins)
                x *= len([i for i in instructions if i.i_out == ins.i_out])
            if path_normalization == "none":
                x = 1

            alpha /= x
            alpha *= out_var[ins.i_out]
            alpha = sqrt(alpha)
            alpha *= sqrt(ins.path_weight) if path_weight_sqrt else ins.path_weight
            normalization_coefficients += [alpha]

        self.instructions = [
            Instruction(
                ins.i_in1,
                ins.i_in2,
                ins.i_out,
                ins.connection_mode,
                ins.has_weight,
                alpha,
                ins.path_shape,
            )
            for ins, alpha in zip(instructions, normalization_coefficients)
        ]

        self._in1_dim = self.irreps_in1.dim
        self._in2_dim = self.irreps_in2.dim

        self.weight_numel = sum(
            prod(ins.path_shape) for ins in self.instructions if ins.has_weight
        )

        self.internal_weights = internal_weights
        self.shared_weights = internal_weights
        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            self.register_buffer("weight", torch.Tensor())

        self.init_rescale_bias()

    def calculate_fan_in(self, ins):
        return {
            "uvw": (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
            "uvu": self.irreps_in2[ins.i_in2].mul,
            "uvv": self.irreps_in1[ins.i_in1].mul,
            "uuw": self.irreps_in1[ins.i_in1].mul,
            "uuu": 1,
            "uvuv": 1,
            "uvu<v": 1,
            "u<vw": self.irreps_in1[ins.i_in1].mul
            * (self.irreps_in2[ins.i_in2].mul - 1)
            // 2,
        }[ins.connection_mode]

    def init_rescale_bias(self) -> None:
        irreps_out = self.irreps_out
        # For each zeroth order output irrep we need a bias
        self.irreps_out_orders = [
            int(irrep_str[-2]) for irrep_str in str(irreps_out).split("+")
        ]
        self.irreps_out_dims = [
            int(irrep_str.split("x")[0]) for irrep_str in str(irreps_out).split("+")
        ]
        self.irreps_out_slices = irreps_out.slices()

        # Store tuples of slices and corresponding biases in a list
        self.bias = None
        self.bias_slices = []
        self.bias_slice_idx = []
        self.irreps_bias = self.irreps_out.simplify()
        self.irreps_bias_orders = [
            int(irrep_str[-2]) for irrep_str in str(self.irreps_bias).split("+")
        ]
        self.irreps_bias_parity = [
            irrep_str[-1] for irrep_str in str(self.irreps_bias).split("+")
        ]
        self.irreps_bias_dims = [
            int(irrep_str.split("x")[0])
            for irrep_str in str(self.irreps_bias).split("+")
        ]
        if self.use_bias:
            self.bias = []
            for slice_idx in range(len(self.irreps_bias_orders)):
                if (
                    self.irreps_bias_orders[slice_idx] == 0
                    and self.irreps_bias_parity[slice_idx] == "e"
                ):
                    out_slice = self.irreps_bias.slices()[slice_idx]
                    out_bias = torch.nn.Parameter(
                        torch.zeros(
                            self.irreps_bias_dims[slice_idx], dtype=self.weight.dtype
                        )
                    )
                    self.bias += [out_bias]
                    self.bias_slices += [out_slice]
                    self.bias_slice_idx += [slice_idx]
        self.bias = torch.nn.ParameterList(self.bias)

        self.slices_sqrt_k = {}
        with torch.no_grad():
            slices_fan_in = {}  # fan_in per slice
            for instr in self.instructions:
                slice_idx = instr[2]
                fan_in = self.calculate_fan_in(instr)
                slices_fan_in[slice_idx] = (
                    slices_fan_in[slice_idx] + fan_in
                    if slice_idx in slices_fan_in.keys()
                    else fan_in
                )
            for instr in self.instructions:
                slice_idx = instr[2]
                if self.rescale:
                    sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                else:
                    sqrt_k = 1.0
                self.slices_sqrt_k[slice_idx] = (
                    self.irreps_out_slices[slice_idx],
                    sqrt_k,
                )

            # Re-initialize weights in each instruction
            if self.internal_weights:
                for weight, instr in zip(self.weight_views(), self.instructions):
                    slice_idx = instr[2]
                    if self.rescale:
                        sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                        weight.data.mul_(sqrt_k)
                    #

            # Initialize the biases

    @torch.jit.unused
    def _prep_weights_python(
        self, weight: Optional[Union[torch.Tensor, List[torch.Tensor]]]
    ) -> Optional[torch.Tensor]:
        if isinstance(weight, list):
            weight_shapes = [
                ins.path_shape for ins in self.instructions if ins.has_weight
            ]
            if not self.shared_weights:
                weight = [
                    w.reshape(-1, prod(shape))
                    for w, shape in zip(weight, weight_shapes)
                ]
            else:
                weight = [
                    w.reshape(prod(shape)) for w, shape in zip(weight, weight_shapes)
                ]
            return torch.cat(weight, dim=-1)
        else:
            return weight

    def _get_weights(self, weight: Optional[torch.Tensor]) -> torch.Tensor:
        if not torch.jit.is_scripting():
            weight = self._prep_weights_python(weight)
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError(
                    "Weights must be provided when the TensorProduct does not have `internal_weights`"
                )
            return self.weight
        else:
            if self.shared_weights:
                assert weight.shape == (self.weight_numel,), "Invalid weight shape"
            else:
                assert weight.shape[-1] == self.weight_numel, "Invalid weight shape"
                assert (
                    weight.ndim > 1
                ), "When shared weights is false, weights must have batch dimension"
            return weight

    def weight_views(
        self, weight: Optional[torch.Tensor] = None, yield_instruction: bool = False
    ):
        r"""Iterator over weight views for each weighted instruction.

        Parameters
        ----------
        weight : `torch.Tensor`, optional
            like ``weight`` argument to ``forward()``

        yield_instruction : `bool`, default False
            Whether to also yield the corresponding instruction.

        Yields
        ------
        If ``yield_instruction`` is ``True``, yields ``(instruction_index, instruction, weight_view)``.
        Otherwise, yields ``weight_view``.
        """
        weight = self._get_weights(weight)
        batchshape = weight.shape[:-1]
        offset = 0
        for ins_i, ins in enumerate(self.instructions):
            if ins.has_weight:
                flatsize = prod(ins.path_shape)
                this_weight = weight.narrow(-1, offset, flatsize).view(
                    batchshape + ins.path_shape
                )
                offset += flatsize
                if yield_instruction:
                    yield ins_i, ins, this_weight
                else:
                    yield this_weight

    def _sum_tensors(
        self, xs: List[torch.Tensor], shape: torch.Size, like: torch.Tensor
    ):
        if len(xs) > 0:
            out = xs[0]
            for x in xs[1:]:
                out = out + x
            return out
        return like.new_zeros(shape)

    def _main_left_right(self, x1s, x2s, weights):
        
        empty = torch.empty((), device="cpu")
        output_shape = torch.broadcast_tensors(
            empty.expand(x1s.shape[:-1]), empty.expand(x2s.shape[:-1])
        )[0].shape
        del empty

        
        x1s, x2s = x1s.broadcast_to(output_shape + (-1,)), x2s.broadcast_to(
            output_shape + (-1,)
        )
        output_shape = output_shape + (self.irreps_out.dim,)
        x1s = x1s.reshape(-1, self.irreps_in1.dim)
        x2s = x2s.reshape(-1, self.irreps_in2.dim)
        batch_numel = x1s.shape[0]

        
        if self.weight_numel > 0:
            weights = weights.reshape(-1, self.weight_numel)

        
        x1_list = [
            x1s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(self.irreps_in1.slices(), self.irreps_in1)
        ]
        x2_list = [
            x2s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(self.irreps_in2.slices(), self.irreps_in2)
        ]

        outputs = []
        flat_weight_index = 0

        
        for ins in self.instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]

            
            #     continue

            x1 = x1_list[ins.i_in1]
            x2 = x2_list[ins.i_in2]

            
            xx = torch.einsum("zui,zvj->zuvij", x1, x2)
            
            w3j = o3.wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l).to(
                x1s.device
            )

            l1l2l3 = (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)

            if ins.has_weight:
                w = weights[
                    :, flat_weight_index : flat_weight_index + prod(ins.path_shape)
                ].reshape((-1,) + tuple(ins.path_shape))

                flat_weight_index += prod(ins.path_shape)

            

            if ins.connection_mode == "uvw":
                assert ins.has_weight
                if l1l2l3 == (0, 0, 0):
                    result = torch.einsum(
                        "zuvw,zu,zv->zw",
                        w,
                        x1.reshape(batch_numel, mul_ir_in1.dim),
                        x2.reshape(batch_numel, mul_ir_in2.dim),
                    )
                elif mul_ir_in1.ir.l == 0:
                    result = torch.einsum(
                        "zuvw,zu,zvj->zwj",
                        w,
                        x1.reshape(batch_numel, mul_ir_in1.dim),
                        x2,
                    ) / sqrt(mul_ir_out.ir.dim)
                elif mul_ir_in2.ir.l == 0:
                    result = torch.einsum(
                        "zuvw,zui,zv->zwi",
                        w,
                        x1,
                        x2.reshape(batch_numel, mul_ir_in2.dim),
                    ) / sqrt(mul_ir_out.ir.dim)
                elif mul_ir_out.ir.l == 0:
                    result = torch.einsum("zuvw,zui,zvi->zw", w, x1, x2) / sqrt(
                        mul_ir_in1.ir.dim
                    )
                else:
                    result = torch.einsum("zuvw,ijk,zuvij->zwk", w, w3j, xx)
            if ins.connection_mode == "uvu":
                assert mul_ir_in1.mul == mul_ir_out.mul
                if ins.has_weight:
                    if l1l2l3 == (0, 0, 0):
                        result = torch.einsum(
                            "zuv,zu,zv->zu",
                            w,
                            x1.reshape(batch_numel, mul_ir_in1.dim),
                            x2.reshape(batch_numel, mul_ir_in2.dim),
                        )
                    elif mul_ir_in1.ir.l == 0:
                        result = torch.einsum(
                            "zuv,zu,zvj->zuj",
                            w,
                            x1.reshape(batch_numel, mul_ir_in1.dim),
                            x2,
                        ) / sqrt(mul_ir_out.ir.dim)
                    elif mul_ir_in2.ir.l == 0:
                        result = torch.einsum(
                            "zuv,zui,zv->zui",
                            w,
                            x1,
                            x2.reshape(batch_numel, mul_ir_in2.dim),
                        ) / sqrt(mul_ir_out.ir.dim)
                    elif mul_ir_out.ir.l == 0:
                        result = torch.einsum("zuv,zui,zvi->zu", w, x1, x2) / sqrt(
                            mul_ir_in1.ir.dim
                        )
                    else:
                        result = torch.einsum("zuv,ijk,zuvij->zuk", w, w3j, xx)
                else:
                    # not so useful operation because v is summed
                    result = torch.einsum("ijk,zuvij->zuk", w3j, xx)
            

            result = ins.path_weight * result
            outputs.append(result.reshape(batch_numel, mul_ir_out.dim))

        
        outputs = [
            self._sum_tensors(
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
        return outputs.reshape(output_shape)

    def forward(self, x, y, weight: Optional[torch.Tensor] = None):
        assert x.shape[-1] == self._in1_dim, "Incorrect last dimension for x"
        assert y.shape[-1] == self._in2_dim, "Incorrect last dimension for y"
        weight = self.weight
        return self._main_left_right(x, y, weight)


class Simple_TensorProduct_oTchannel(torch.nn.Module, CodeGenMixin):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: List[tuple] = None,
        learnable_weight=None,
        connection_mode="uvu",
        reduce_same_order=False,
        in1_var: Optional[Union[List[float], torch.Tensor]] = None,
        in2_var: Optional[Union[List[float], torch.Tensor]] = None,
        out_var: Optional[Union[List[float], torch.Tensor]] = None,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights=True,
        path_weight_sqrt=True,
        rescale=True,
        use_bias=False,
    ):
        super().__init__()
        self.rescale = rescale

        self.use_bias = use_bias
        if self.use_bias:
            raise ValueError(
                "Not implemented yet, the bias is for order 0 irreps only, it  only works for TensorProductRescale"
            )
        self.irreps_in1 = (
            o3.Irreps(irreps_in1) if isinstance(irreps_in1, str) else irreps_in1
        )
        self.irreps_in2 = (
            o3.Irreps(irreps_in2) if isinstance(irreps_in2, str) else irreps_in2
        )
        self.irreps_out = (
            o3.Irreps(irreps_out) if isinstance(irreps_out, str) else irreps_out
        )

        for i in range(1, len(self.irreps_in1)):
            if self.irreps_in1[i][0] != self.irreps_in1[i - 1][0]:
                raise ValueError("The input channel must have the same channel")

        if instructions is None and learnable_weight is None:
            raise ValueError("please set instructions or learable weight")
        if instructions is not None and learnable_weight is not None:
            raise ValueError("please set instructions or learable weight")
        if instructions is None:
            instructions, irreps_output = self._get_instruction(
                irreps_in1,
                irreps_in2,
                irreps_out,
                learnable_weight=learnable_weight,
                connection_mode=connection_mode,
            )
            self.irreps_out = irreps_output
        instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]
        instructions = [
            Instruction(
                i_in1=i_in1,
                i_in2=i_in2,
                i_out=i_out,
                connection_mode=connection_mode,
                has_weight=has_weight,
                path_weight=path_weight,
                path_shape={
                    "uvw": (
                        self.irreps_in1[i_in1].mul,
                        self.irreps_in2[i_in2].mul,
                        self.irreps_out[i_out].mul,
                    ),
                    "uvu": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uvv": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uuw": (self.irreps_in1[i_in1].mul, self.irreps_out[i_out].mul),
                    "uuu": (self.irreps_in1[i_in1].mul,),
                    "uvuv": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uvu<v": (
                        self.irreps_in1[i_in1].mul
                        * (self.irreps_in2[i_in2].mul - 1)
                        // 2,
                    ),
                    "u<vw": (
                        self.irreps_in1[i_in1].mul
                        * (self.irreps_in2[i_in2].mul - 1)
                        // 2,
                        self.irreps_out[i_out].mul,
                    ),
                }[connection_mode],
            )
            for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in instructions
        ]

        if in1_var is None:
            in1_var = [1.0 for _ in range(len(self.irreps_in1))]

        if in2_var is None:
            in2_var = [1.0 for _ in range(len(self.irreps_in2))]

        if out_var is None:
            out_var = [1.0 for _ in range(len(self.irreps_out))]

        def num_elements(ins):
            return {
                "uvw": (
                    self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul
                ),
                "uvu": self.irreps_in2[ins.i_in2].mul,
                "uuu": self.irreps_in2[ins.i_in2].mul,
            }[ins.connection_mode]

        normalization_coefficients = []
        for ins in instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]

            if irrep_normalization == "component":
                alpha = mul_ir_out.ir.dim
            if irrep_normalization == "norm":
                alpha = mul_ir_in1.ir.dim * mul_ir_in2.ir.dim
            if irrep_normalization == "none":
                alpha = 1

            if path_normalization == "element":
                x = sum(
                    in1_var[i.i_in1] * in2_var[i.i_in2] * num_elements(i)
                    for i in instructions
                    if i.i_out == ins.i_out
                )
            if path_normalization == "path":
                x = in1_var[ins.i_in1] * in2_var[ins.i_in2] * num_elements(ins)
                x *= len([i for i in instructions if i.i_out == ins.i_out])
            if path_normalization == "none":
                x = 1

            alpha /= x
            alpha *= out_var[ins.i_out]
            alpha = sqrt(alpha)
            alpha *= sqrt(ins.path_weight) if path_weight_sqrt else ins.path_weight
            normalization_coefficients += [alpha]

        self.instructions = [
            Instruction(
                ins.i_in1,
                ins.i_in2,
                ins.i_out,
                ins.connection_mode,
                ins.has_weight,
                alpha,
                ins.path_shape,
            )
            for ins, alpha in zip(instructions, normalization_coefficients)
        ]

        self._in1_dim = self.irreps_in1.dim
        self._in2_dim = self.irreps_in2.dim

        self.weight_numel = sum(
            prod(ins.path_shape) for ins in self.instructions if ins.has_weight
        )

        self.internal_weights = internal_weights
        self.shared_weights = internal_weights
        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            self.register_buffer("weight", torch.Tensor([0]))

        self.init_rescale_bias()

        graphmod_left_right = CODEGEN_MAIN_LEFT_RIGHT(
            self.irreps_in1, self.irreps_in2, self.irreps_out, self.instructions
        )

        assert graphmod_left_right is not None
        self._codegen_register({"_compiled_main_left_right": graphmod_left_right})

    # this is specific tensor product without reduce operation on same order
    def _get_instruction(
        self,
        input1,
        input2,
        output,
        learnable_weight=True,
        connection_mode="uvu",
        reduce_sameorder=True,
    ):
        if learnable_weight is False:
            connection_mode = "uvu"
        if reduce_sameorder:
            irreps_output = []
            instructions = []

            for i, (mul, ir_in) in enumerate(input1):
                for j, (_, ir_edge) in enumerate(input2):
                    for ir_out in ir_in * ir_edge:
                        if ir_out in output:  # or ir_out == o3.Irrep(0, 1):
                            k = len(irreps_output)
                            irreps_output.append((mul, ir_out))
                            instructions.append(
                                (i, j, k, connection_mode, learnable_weight)
                            )
            irreps_output = o3.Irreps(irreps_output)

            return instructions, irreps_output







    def calculate_fan_in(self, ins):
        return {
            "uvw": (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
            "uvu": self.irreps_in2[ins.i_in2].mul,
            "uvv": self.irreps_in1[ins.i_in1].mul,
            "uuw": self.irreps_in1[ins.i_in1].mul,
            "uuu": 1,
            "uvuv": 1,
            "uvu<v": 1,
            "u<vw": self.irreps_in1[ins.i_in1].mul
            * (self.irreps_in2[ins.i_in2].mul - 1)
            // 2,
        }[ins.connection_mode]

    def init_rescale_bias(self) -> None:
        irreps_out = self.irreps_out
        # For each zeroth order output irrep we need a bias
        self.irreps_out_orders = [
            int(irrep_str[-2]) for irrep_str in str(irreps_out).split("+")
        ]
        self.irreps_out_dims = [
            int(irrep_str.split("x")[0]) for irrep_str in str(irreps_out).split("+")
        ]
        self.irreps_out_slices = irreps_out.slices()

        # Store tuples of slices and corresponding biases in a list
        self.bias = None
        self.bias_slices = []
        self.bias_slice_idx = []
        self.irreps_bias = self.irreps_out.simplify()
        self.irreps_bias_orders = [
            int(irrep_str[-2]) for irrep_str in str(self.irreps_bias).split("+")
        ]
        self.irreps_bias_parity = [
            irrep_str[-1] for irrep_str in str(self.irreps_bias).split("+")
        ]
        self.irreps_bias_dims = [
            int(irrep_str.split("x")[0])
            for irrep_str in str(self.irreps_bias).split("+")
        ]
        if self.use_bias:
            self.bias = []
            for slice_idx in range(len(self.irreps_bias_orders)):
                if (
                    self.irreps_bias_orders[slice_idx] == 0
                    and self.irreps_bias_parity[slice_idx] == "e"
                ):
                    out_slice = self.irreps_bias.slices()[slice_idx]
                    out_bias = torch.nn.Parameter(
                        torch.zeros(
                            self.irreps_bias_dims[slice_idx], dtype=self.weight.dtype
                        )
                    )
                    self.bias += [out_bias]
                    self.bias_slices += [out_slice]
                    self.bias_slice_idx += [slice_idx]
        self.bias = torch.nn.ParameterList(self.bias)

        self.slices_sqrt_k = {}
        with torch.no_grad():
            slices_fan_in = {}  # fan_in per slice
            for instr in self.instructions:
                slice_idx = instr[2]
                fan_in = self.calculate_fan_in(instr)
                slices_fan_in[slice_idx] = (
                    slices_fan_in[slice_idx] + fan_in
                    if slice_idx in slices_fan_in.keys()
                    else fan_in
                )
            for instr in self.instructions:
                slice_idx = instr[2]
                if self.rescale:
                    sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                else:
                    sqrt_k = 1.0
                self.slices_sqrt_k[slice_idx] = (
                    self.irreps_out_slices[slice_idx],
                    sqrt_k,
                )

            # Re-initialize weights in each instruction
            if self.internal_weights:
                for weight, instr in zip(self.weight_views(), self.instructions):
                    slice_idx = instr[2]
                    if self.rescale:
                        sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                        weight.data.mul_(sqrt_k)
                    #

            # Initialize the biases

    @torch.jit.unused
    def _prep_weights_python(
        self, weight: Optional[Union[torch.Tensor, List[torch.Tensor]]]
    ) -> Optional[torch.Tensor]:
        if isinstance(weight, list):
            weight_shapes = [
                ins.path_shape for ins in self.instructions if ins.has_weight
            ]
            if not self.shared_weights:
                weight = [
                    w.reshape(-1, prod(shape))
                    for w, shape in zip(weight, weight_shapes)
                ]
            else:
                weight = [
                    w.reshape(prod(shape)) for w, shape in zip(weight, weight_shapes)
                ]
            return torch.cat(weight, dim=-1)
        else:
            return weight

    def _get_weights(self, weight: Optional[torch.Tensor]) -> torch.Tensor:
        if not torch.jit.is_scripting():
            weight = self._prep_weights_python(weight)
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError(
                    "Weights must be provided when the TensorProduct does not have `internal_weights`"
                )
            return self.weight
        else:
            if self.shared_weights:
                assert weight.shape == (self.weight_numel,), "Invalid weight shape"
            else:
                assert weight.shape[-1] == self.weight_numel, "Invalid weight shape"
                assert (
                    weight.ndim > 1
                ), "When shared weights is false, weights must have batch dimension"
            return weight

    def weight_views(
        self, weight: Optional[torch.Tensor] = None, yield_instruction: bool = False
    ):
        r"""Iterator over weight views for each weighted instruction.

        Parameters
        ----------
        weight : `torch.Tensor`, optional
            like ``weight`` argument to ``forward()``

        yield_instruction : `bool`, default False
            Whether to also yield the corresponding instruction.

        Yields
        ------
        If ``yield_instruction`` is ``True``, yields ``(instruction_index, instruction, weight_view)``.
        Otherwise, yields ``weight_view``.
        """
        weight = self._get_weights(weight)
        batchshape = weight.shape[:-1]
        offset = 0
        for ins_i, ins in enumerate(self.instructions):
            if ins.has_weight:
                flatsize = prod(ins.path_shape)
                this_weight = weight.narrow(-1, offset, flatsize).view(
                    batchshape + ins.path_shape
                )
                offset += flatsize
                if yield_instruction:
                    yield ins_i, ins, this_weight
                else:
                    yield this_weight

    def forward(self, x, y, weight: Optional[torch.Tensor] = None):
        assert x.shape[-2:].numel() == self._in1_dim, "Incorrect last dimension for x"
        assert y.shape[-2:].numel() == self._in2_dim, "Incorrect last dimension for y"

        if weight is not None and self.weight_numel > 0:
            weight = self.weight.reshape(1, self.weight_numel) * weight.reshape(
                -1, self.weight_numel
            )
        else:
            weight = self.weight
        return self._compiled_main_left_right(x, y, weight)







#     """
#     """

#         irreps_node_input,
#         irreps_node_output,








#         """
#                 irreps_in,
#                 irreps_in,
#         """


