# -*- coding: utf-8 -*-
import math
import os
import sys
import time

import e3nn
import torch
import torch_geometric
from e3nn import o3
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists
from e3nn.util.jit import compile_mode

from fairchem.core.models.gemnet.layers.radial_basis import RadialBasis
from torch_cluster import radius_graph
from torch_scatter import scatter

from .dp_attention_transformer import DotProductAttention, DPTransBlock, ScaleFactor
from .drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath
from .expnorm_rbf import ExpNormalSmearing
from .fast_activation import Activation, Gate
from .gaussian_rbf import GaussianRadialBasisLayer
from .graph_attention_transformer import (
    AttnHeads2Vec,
    DepthwiseTensorProduct,
    EdgeDegreeEmbeddingNetwork,
    FeedForwardNetwork,
    FullyConnectedTensorProductRescaleNorm,
    FullyConnectedTensorProductRescaleNormSwishGate,
    FullyConnectedTensorProductRescaleSwishGate,
    NodeEmbeddingNetwork,
    ScaledScatter,
    SeparableFCTP,
    Vec2AttnHeads,
    get_norm_layer,
)
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
)

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)
import torch.nn as nn

_RESCALE = True
_USE_BIAS = True

_MAX_ATOM_TYPE = 64
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666


class DotProductAttentionTransformerMD17(torch.nn.Module):
    def __init__(
        self,
        irreps_in="64x0e",
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=6,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=5.0,
        number_of_basis=128,
        basis_type="gaussian",
        fc_neurons=[64, 64],
        irreps_feature="512x0e",
        irreps_head="32x0e+16x1o+8x2e",
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        irreps_mlp_mid="128x0e+64x1e+32x2e",
        norm_layer="layer",
        alpha_drop=0.2,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        mean=None,
        std=None,
        scale=None,
        atomref=None,
    ):
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.scale = scale
        self.register_buffer("atomref", atomref)
        self.register_buffer("task_mean", mean)
        self.register_buffer("task_std", std)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
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
        elif self.basis_type == "exp":
            self.rbf = ExpNormalSmearing(
                cutoff_lower=0.0,
                cutoff_upper=self.max_radius,
                num_rbf=self.number_of_basis,
                trainable=False,
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

        self.norm = get_norm_layer(self.norm_layer)(self.irreps_node_embedding)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)


    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_node_embedding
            blk = DPTransBlock(
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
        self.apply(self._init_weights)

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

    # the gradient of energy is following the implementation here:
    @torch.enable_grad()
    def forward(self, batch_data) -> torch.Tensor:
        node_atom = batch_data["atomic_numbers"]
        pos = batch_data["pos"]
        batch = batch_data["batch"]
        edge_src, edge_dst = radius_graph(
            pos, r=self.max_radius, batch=batch, max_num_neighbors=1000
        )
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr,
            x=edge_vec,
            normalize=True,
            normalization="component",
        )

        atom_embedding, atom_attr, atom_onehot = self.atom_embed(node_atom)
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)
        edge_degree_embedding = self.edge_deg_embed(
            atom_embedding, edge_sh, edge_length_embedding, edge_src, edge_dst, batch
        )
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

        for blk in self.blocks:
            node_features = blk(
                node_input=node_features,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                batch=batch,
            )

        node_features = self.norm(node_features, batch=batch)
        batch_data["node_embedding"] = None
        batch_data["node_vec"] = node_features
        return node_features


@register_model
def dot_product_attention_transformer_exp_l2_md17(
    irreps_in,
    radius,
    num_basis=128,
    atomref=None,
    task_mean=None,
    task_std=None,
    num_layers=6,
    **kwargs,
):
    model = DotProductAttentionTransformerMD17(
        irreps_in="64x0e+32x1e+16x2e+8x3e+4x4e",  # irreps_in,
        irreps_node_embedding="128x0e+128x1e+128x2e+128x3e+128x4e",
        num_layers=num_layers,  #'128x0e+64x1e+32x2e+16x3e+8x4e'
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e+1x3e+1x4e",
        max_radius=radius,
        number_of_basis=num_basis,
        fc_neurons=[64, 64],
        basis_type="exp",
        irreps_feature="512x0e+256x1e+128x2e+64x3e+32x4e",  #
        irreps_head="32x0e+16x1e+8x2e+4x3e+2x4e",
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        irreps_mlp_mid="384x0e+192x1e+96x2e+48x3e+24x4e",
        norm_layer="layer",
        alpha_drop=0.0,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        mean=task_mean,
        std=task_std,
        scale=None,
        atomref=atomref,
    )
    return model


@register_model
def dot_product_attention_transformer_exp_l3_md17(
    irreps_in,
    radius,
    num_basis=128,
    atomref=None,
    task_mean=None,
    task_std=None,
    num_layers=6,
    **kwargs,
):
    model = DotProductAttentionTransformerMD17(
        irreps_in=irreps_in,
        irreps_node_embedding="128x0e+64x1e+64x2e+32x3e",
        num_layers=num_layers,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e+1x3e",
        max_radius=radius,
        number_of_basis=num_basis,
        fc_neurons=[64, 64],
        basis_type="exp",
        irreps_feature="512x0e",
        irreps_head="32x0e+16x1e+16x2e+8x3e",
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        irreps_mlp_mid="384x0e+192x1e+192x2e+96x3e",
        norm_layer="layer",
        alpha_drop=0.0,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        mean=task_mean,
        std=task_std,
        scale=None,
        atomref=atomref,
    )
    return model
