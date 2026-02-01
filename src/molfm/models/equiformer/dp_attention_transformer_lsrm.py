# -*- coding: utf-8 -*-
import math

import e3nn
import torch
import torch_geometric
from e3nn import o3
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists
from e3nn.util.jit import compile_mode

from fairchem.core.models.gemnet.layers.radial_basis import RadialBasis
from torch import nn
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter

from .Concat import IrrepsConcat
from .dp_attention_transformer import DotProductAttention, DPTransBlock, ScaleFactor
from .drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath
from .expnorm_rbf import ExpNormalSmearing, GaussianSmearing
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
from .Scatter import IrrepsScatter
from .tensor_product_rescale import (
    FullyConnectedTensorProductRescale,
    LinearRS,
    TensorProductRescale,
    irreps2gate,
)

_RESCALE = True
_USE_BIAS = True

_MAX_ATOM_TYPE = 64
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666

act_class_mapping = {
    # "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def norm(vec, dim=1, keepdim=False):
    return torch.square(vec + 1e-6).sum(dim=dim, keepdim=keepdim).sqrt() + 1e-6


class LongShortIneractModel_dis_direct_Eqformer(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_gaussians,
        cutoff,
        norm=False,
        act="silu",
        num_heads=8,
        **kwargs,
    ):
        super().__init__(
            aggr="add", node_dim=0
        )  # currently only node embedding is computed and updated, only group message flow to node
        self.act = act_class_mapping[act]()
        self.norm = norm
        self.layernorm_node = nn.LayerNorm(hidden_channels)
        self.layernorm_group = nn.LayerNorm(hidden_channels)
        self.layernorm_node_vec = nn.LayerNorm(hidden_channels)
        self.layernorm_group_vec = nn.LayerNorm(hidden_channels)
        self.model_2 = nn.ModuleDict(
            {
                "q": nn.Linear(hidden_channels, hidden_channels),
                "k": nn.Linear(hidden_channels, hidden_channels),
                "val": nn.Linear(hidden_channels, hidden_channels),
                "mlp_scalar_pos": nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels // 2),
                    self.act,
                    nn.Linear(hidden_channels // 2, hidden_channels),
                ),
                "mlp_scalar_vec": nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels // 2),
                    self.act,
                    nn.Linear(hidden_channels // 2, hidden_channels),
                ),
                "linears": nn.ModuleList(
                    [
                        nn.Linear(hidden_channels, hidden_channels, bias=False)
                        for _ in range(6)
                    ]
                ),
            }
        )
        self.model_1 = None
        self.num_heads = num_heads
        self.attn_channels = hidden_channels // num_heads
        self.reset_parameters()

    def reset_parameters(self):
        """
        Create Xavier Unifrom for all linear modules
        """
        self.layernorm_node.reset_parameters()
        self.layernorm_group.reset_parameters()
        for model in [self.model_1, self.model_2]:
            if model is None:
                continue
            for _, value in model.items():
                if isinstance(value, nn.ModuleList):
                    for m in value.modules():
                        if isinstance(m, nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight)
                elif isinstance(value, nn.Linear):
                    torch.nn.init.xavier_uniform_(value.weight)
                    value.bias.data.fill_(0)
                else:
                    pass

    def forward(
        self,
        edge_index,
        node_embedding,
        node_pos,
        node_vec,
        group_embedding,
        group_pos,
        group_vec,
        edge_attr,
        edge_weight,
        edge_vec,
    ):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            group_embedding = self.layernorm_group(group_embedding)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]

        attn_2, val_2 = self.calculate_attention(
            node_embedding,
            group_embedding,
            edge_index[0],
            edge_index[1],
            edge_attr,
            self.model_2,
            "silu",
        )
        m_s_node, m_v_node = self.propagate(
            edge_index.flip(0),
            size=(num_groups, num_nodes),
            x=(group_embedding, node_embedding),
            v=group_vec[edge_index[1]],
            u_ij=-edge_vec,
            d_ij=edge_weight,
            attn_score=attn_2,
            val=val_2[edge_index[1]],
            mode="group_to_node",
        )

        v_node_1 = self.model_2["linears"][2](node_vec)
        v_node_2 = self.model_2["linears"][3](node_vec)
        dx_node = (v_node_1 * v_node_2).sum(dim=1) * self.model_2["linears"][4](
            m_s_node
        ) + self.model_2["linears"][5](m_s_node)
        dv_node = m_v_node + self.model_2["linears"][0](m_s_node).unsqueeze(
            1
        ) * self.model_2["linears"][1](node_vec)
        return dx_node, dv_node

    def calculate_attention(
        self, x_1, x_2, x1_index, x2_index, expanded_edge_weight, model, attn_type
    ):
        r"""
        Calculate attention value for each edge.
        x_1: embedding for query. target node embedding.
        x_2: embedding for key value. source node embedding.
        edge_index: graph to calculate attention
        expanded_edge_weight: the expanded edge weight
        model: the model for calculating attention, be a dictionary with keys: q, k, val, mlp_edge_attr, mlp_scalar_pos, mlp_scalar_vec
        """
        __supported_attn__ = ["softmax", "silu"]
        q = model["q"](x_1).reshape(
            -1, self.num_heads, self.attn_channels
        )  # num_groups x num_heads x attn_channels
        k = model["k"](x_2).reshape(
            -1, self.num_heads, self.attn_channels
        )  # num_nodes x num_heads x attn_channels
        val = model["val"](x_2).reshape(-1, self.num_heads, self.attn_channels)

        q_i = q[x1_index]
        k_j = k[x2_index]

        expanded_edge_weight = expanded_edge_weight.reshape(
            -1, self.num_heads, self.attn_channels
        )
        attn = q_i * k_j * expanded_edge_weight
        attn = attn.sum(dim=-1) / math.sqrt(self.attn_channels)

        if attn_type == "softmax":
            attn = softmax(attn, x1_index, dim=0)
        elif attn_type == "silu":
            attn = act_class_mapping["silu"]()(attn)
        else:
            raise NotImplementedError(
                f"Attention type {attn_type} is not supported, supported types are {__supported_attn__}"
            )
        return attn, val

    def message(self, x_i, x_j, v, u_ij, d_ij, attn_score, val, mode):
        """
        Calculate the message from node j to node i
        Return: Scalar message from node j to node i, Vector message from node j to node i
        @param x_j: Node feature of node j
        @param x_i: Node feature of node i
        @param smeared_distance: Smearing distance between node i and node j
        @param v: Embedding of the group th
        @param mode: 'node_to_group' or 'group_to_node'
        """

        if mode == "node_to_group":
            model = self.model_1
        else:
            model = self.model_2

        m_s_ij = val * attn_score.unsqueeze(-1)  # scalar message
        m_s_ij = m_s_ij.reshape(-1, self.num_heads * self.attn_channels)
        m_v_ij = model["mlp_scalar_pos"](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(
            -1
        ) + model["mlp_scalar_vec"](m_s_ij).unsqueeze(1) * (
            v
        )  # vector message
        return m_s_ij, m_v_ij

    def aggregate(self, features, index, ptr, dim_size):
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec


class Bipartite_Edge_Feat_Init(nn.Module):
    def __init__(
        self,
        rbf_type="expnorm",
        num_rbf=50,
        trainable_rbf=True,
        hidden_channels=128,
        cutoff_lower=0,
        cutoff_upper=10,
    ):
        super().__init__()
        if rbf_type == "exp":
            rbf = ExpNormalSmearing
        elif rbf_type == "gaussian":
            rbf = GaussianSmearing
        else:
            assert False
        self.distance_encoder = rbf(
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            num_rbf=num_rbf,
            trainable=trainable_rbf,
        )
        self.rbf_linear = nn.Linear(num_rbf, hidden_channels)

    def forward(self, edge_index, node_pos, group_pos, *args, **kwargs):
        edge_vec = node_pos[edge_index[0]] - group_pos[edge_index[1]]
        edge_weight = norm(edge_vec, dim=1)
        edge_vec = edge_vec / edge_weight.unsqueeze(1)
        edge_attr = self.distance_encoder(edge_weight)
        edge_attr = self.rbf_linear(edge_attr)
        return edge_index, edge_weight, edge_attr, edge_vec


class DotProductAttentionTransformerMD17_Serial(torch.nn.Module):
    def __init__(
        self,
        irreps_in="64x0e",
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=6,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=5.0,
        long_cutoff_upper=9.0,
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
        long_num_layers=2,
    ):
        super().__init__()
        self.long_cutoff_upper = long_cutoff_upper
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
        self.long_num_layers = long_num_layers
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.hidden_channels = int(irreps_feature.split("x")[0])

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
        self.IrrepsScatter = IrrepsScatter(self.irreps_node_embedding)
        self.blocks = torch.nn.ModuleList()
        self.long_blocks = torch.nn.ModuleList()
        self.build_blocks()

        self.bipartite_edge_fea_init = Bipartite_Edge_Feat_Init(
            rbf_type=self.basis_type,
            num_rbf=50,
            trainable_rbf=True,
            hidden_channels=self.hidden_channels,
            cutoff_lower=0,
            cutoff_upper=long_cutoff_upper,
        )

        self.norm_short = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.norm_long = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        self.concat = IrrepsConcat(self.irreps_feature)
        self.head = torch.nn.Sequential(
            LinearRS(
                self.irreps_feature + self.irreps_feature,
                self.irreps_feature,
                rescale=_RESCALE,
            ),
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps("1x0e"), rescale=_RESCALE),
        )
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)

        self.apply(self._init_weights)

    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
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

        for i in range(self.long_num_layers):
            irreps_block_output = self.irreps_feature
            blk = LongShortIneractModel_dis_direct_Eqformer(
                self.hidden_channels,
                num_gaussians=50,
                cutoff=self.long_cutoff_upper,
                norm=True,
                act="silu",
                num_heads=8,
            )
            self.long_blocks.append(blk)

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
    def forward(self, data) -> torch.Tensor:
        pos = data.pos
        batch = data.batch

        # TODO: different group normalization
        group_batch = scatter(batch, data.labels, reduce="mean", dim=0).long().squeeze()
        group_batch = group_batch + batch.max() + 1
        node_atom = data.atomic_numbers.long()
        labels = data.labels
        pos = pos.requires_grad_(True)

        group_pos = scatter(pos * node_atom, labels, reduce="sum", dim=0) / scatter(
            node_atom, labels, reduce="sum", dim=0
        )
        node_id, group_id = data.interaction_graph[0], data.interaction_graph[1]
        node_group_dis = torch.sqrt(
            torch.sum((pos[node_id] - group_pos[group_id]) ** 2, dim=1)
        )
        data.interaction_graph = data.interaction_graph[
            :, node_group_dis <= self.long_cutoff_upper
        ]

        # process the short range graph
        edge_src = data.edge_index[0]
        edge_dst = data.edge_index[1]
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr,
            x=edge_vec,
            normalize=True,
            normalization="component",
        )
        atom_embedding, atom_attr, atom_onehot = self.atom_embed(node_atom.squeeze())
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)
        edge_degree_embedding = self.edge_deg_embed(
            atom_embedding, edge_sh, edge_length_embedding, edge_src, edge_dst, batch
        )
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        node_features_short = node_features
        for i, blk in enumerate(self.blocks):
            node_features_short = blk(
                node_input=node_features_short,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                batch=batch,
            )

        node_vec_long = torch.zeros(
            node_features_short.size(0),
            3,
            node_features_short.size(1),
            device=node_features_short.device,
        )

        node_features_long = node_features_short.clone()
        # process the long range graph
        edge_des_int = data.interaction_graph[0]  # node
        edge_src_int = data.interaction_graph[1] + node_features_short.shape[0]  # group
        (
            pos.index_select(0, edge_des_int)
            - group_pos.index_select(0, edge_src_int - node_features.shape[0])
            + 1e-8
        )

        (
            edge_index_bipartite,
            edge_weight_bipartite,
            edge_attr_bipartite,
            edge_vec_bipartite,
        ) = self.bipartite_edge_fea_init(data.interaction_graph, pos, group_pos)

        for idx in range(self.long_num_layers):
            group_embedding = scatter(node_features_long, labels, dim=0, reduce="mean")
            group_vec = scatter(node_vec_long, labels, dim=0, reduce="mean")

            delta_node_features_long, delta_node_vec_long = self.long_blocks[idx](
                edge_index=edge_index_bipartite,
                node_embedding=node_features_long,
                node_pos=pos,
                node_vec=node_vec_long,
                group_embedding=group_embedding,
                group_pos=group_pos,
                group_vec=group_vec,
                edge_attr=edge_attr_bipartite,
                edge_weight=edge_weight_bipartite,
                edge_vec=edge_vec_bipartite,
            )

            node_features_long = node_features_long + delta_node_features_long
            node_vec_long = node_vec_long + delta_node_vec_long

        node_features_short = self.norm_short(node_features_short)
        node_features_long = self.norm_long(node_features_long)
        node_features_out, _ = self.concat([node_features_short, node_features_long])
        if self.out_dropout is not None:
            node_features_out = self.out_dropout(node_features_out)
        outputs = self.head(node_features_out)
        outputs = self.scale_scatter(outputs, batch, dim=0)

        outputs = outputs * self.task_std + self.task_mean
        if self.scale is not None:
            outputs = self.scale * outputs

        energy = outputs

        forces = -1 * (
            torch.autograd.grad(
                energy,
                pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
        )

        return {"energy": energy, "forces": forces}


@register_model
def dot_product_attention_transformer_exp_l2_md17_lsrmserial(
    irreps_in,
    radius,
    num_basis=128,
    atomref=None,
    task_mean=None,
    task_std=None,
    num_layers=6,
    long_num_layers=2,
    **kwargs,
):
    model = DotProductAttentionTransformerMD17_Serial(
        irreps_in=irreps_in,
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=num_layers,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=radius,
        number_of_basis=num_basis,
        fc_neurons=[64, 64],
        basis_type="exp",
        irreps_feature="512x0e",
        irreps_head="32x0e+16x1e+8x2e",
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        irreps_mlp_mid="384x0e+192x1e+96x2e",
        norm_layer="layer",
        alpha_drop=0.0,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        mean=task_mean,
        std=task_std,
        scale=None,
        atomref=atomref,
        long_num_layers=long_num_layers,
    )
    return model


class DotProductAttentionTransformerMD17_tensorserial(torch.nn.Module):
    def __init__(
        self,
        irreps_in="64x0e",
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=6,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=5.0,
        long_cutoff_upper=9.0,
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
        long_num_layers=2,
    ):
        super().__init__()
        self.long_cutoff_upper = long_cutoff_upper
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
        self.long_num_layers = long_num_layers
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
        self.IrrepsScatter = IrrepsScatter(self.irreps_feature)
        self.blocks = torch.nn.ModuleList()
        self.long_blocks = torch.nn.ModuleList()
        self.build_blocks()

        self.norm_short = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.norm_long = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        self.concat = IrrepsConcat(self.irreps_feature)
        self.head = torch.nn.Sequential(
            LinearRS(
                self.irreps_feature + self.irreps_feature,
                self.irreps_feature,
                rescale=_RESCALE,
            ),
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps("1x0e"), rescale=_RESCALE),
        )
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)

        self.apply(self._init_weights)

    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
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

        for i in range(self.long_num_layers):
            irreps_block_output = self.irreps_feature

            blk = DPTransBlock(
                irreps_node_input=self.irreps_feature,
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
            self.long_blocks.append(blk)

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
    def forward(self, data) -> torch.Tensor:
        pos = data.pos
        batch = data.batch

        # TODO: different group normalization
        group_batch = scatter(batch, data.labels, reduce="mean", dim=0).long().squeeze()
        group_batch = group_batch + batch.max() + 1
        node_atom = data.atomic_numbers.long()
        labels = data.labels
        pos = pos.requires_grad_(True)

        group_pos = scatter(pos * node_atom, labels, reduce="sum", dim=0) / scatter(
            node_atom, labels, reduce="sum", dim=0
        )
        node_id, group_id = data.interaction_graph[0], data.interaction_graph[1]
        node_group_dis = torch.sqrt(
            torch.sum((pos[node_id] - group_pos[group_id]) ** 2, dim=1)
        )
        data.interaction_graph = data.interaction_graph[
            :, node_group_dis <= self.long_cutoff_upper
        ]

        # process the short range graph
        edge_src = data.edge_index[0]
        edge_dst = data.edge_index[1]
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr,
            x=edge_vec,
            normalize=True,
            normalization="component",
        )
        atom_embedding, atom_attr, atom_onehot = self.atom_embed(node_atom.squeeze())
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)
        edge_degree_embedding = self.edge_deg_embed(
            atom_embedding, edge_sh, edge_length_embedding, edge_src, edge_dst, batch
        )
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        node_features_short = node_features
        for i, blk in enumerate(self.blocks):
            node_features_short = blk(
                node_input=node_features_short,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                batch=batch,
            )
        node_features_long = node_features_short.clone()
        # process the long range graph
        edge_des_int = data.interaction_graph[0]  # node
        edge_src_int = data.interaction_graph[1] + node_features_short.shape[0]  # group
        edge_vec_int = (
            pos.index_select(0, edge_des_int)
            - group_pos.index_select(0, edge_src_int - node_features.shape[0])
            + 1e-8
        )

        edge_sh_int = o3.spherical_harmonics(
            l=self.irreps_edge_attr,
            x=edge_vec_int,
            normalize=True,
            normalization="component",
        )

        edge_length_int = (edge_vec_int).norm(dim=1)
        edge_length_embedding_int = self.rbf(edge_length_int)
        # TODO: Split long and short embeddings
        for blk in self.long_blocks:
            group_features = self.IrrepsScatter(node_features_long, labels)

            new_node_features = torch.cat([node_features_long, group_features], dim=0)
            node_idx = torch.arange(
                node_features_short.shape[0], device=node_features_short.device
            )
            new_node_attr = torch.ones_like(new_node_features.narrow(1, 0, 1))
            new_batch = torch.cat([batch, group_batch], dim=0)
            updated_node_featuees = blk(
                node_input=new_node_features,
                node_attr=new_node_attr,
                edge_src=edge_src_int,
                edge_dst=edge_des_int,
                edge_attr=edge_sh_int,
                edge_scalars=edge_length_embedding_int,
                batch=new_batch,
            )
            node_features_long = updated_node_featuees[node_idx, :]

        node_features_short = self.norm_short(node_features_short)
        node_features_long = self.norm_long(node_features_long)
        node_features_out, _ = self.concat([node_features_short, node_features_long])
        if self.out_dropout is not None:
            node_features_out = self.out_dropout(node_features_out)
        outputs = self.head(node_features_out)
        outputs = self.scale_scatter(outputs, batch, dim=0)

        outputs = outputs * self.task_std + self.task_mean
        if self.scale is not None:
            outputs = self.scale * outputs

        energy = outputs

        forces = -1 * (
            torch.autograd.grad(
                energy,
                pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
        )

        return {"energy": energy, "forces": forces}


@register_model
def dot_product_attention_transformer_exp_l2_md17_lsrm_tensorserial(
    irreps_in,
    radius,
    num_basis=128,
    atomref=None,
    task_mean=None,
    task_std=None,
    num_layers=6,
    long_num_layers=2,
    **kwargs,
):
    model = DotProductAttentionTransformerMD17_tensorserial(
        irreps_in=irreps_in,
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=num_layers,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=radius,
        number_of_basis=num_basis,
        fc_neurons=[64, 64],
        basis_type="exp",
        irreps_feature="512x0e",
        irreps_head="32x0e+16x1e+8x2e",
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        irreps_mlp_mid="384x0e+192x1e+96x2e",
        norm_layer="layer",
        alpha_drop=0.0,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        mean=task_mean,
        std=task_std,
        scale=None,
        atomref=atomref,
        long_num_layers=long_num_layers,
    )
    return model


class DotProductAttentionTransformerMD17LSRM(torch.nn.Module):
    def __init__(
        self,
        irreps_in="64x0e",
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=6,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=5.0,
        long_cutoff_upper=9.0,
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
        long_num_layers=2,
    ):
        super().__init__()
        self.long_cutoff_upper = long_cutoff_upper
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
        self.long_num_layers = long_num_layers
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
        self.IrrepsScatter = IrrepsScatter(self.irreps_node_embedding)
        self.blocks = torch.nn.ModuleList()
        self.long_blocks = torch.nn.ModuleList()
        self.build_blocks()

        self.norm_short = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.norm_long = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        self.concat = IrrepsConcat(self.irreps_feature)
        self.head = torch.nn.Sequential(
            LinearRS(
                self.irreps_feature + self.irreps_feature,
                self.irreps_feature,
                rescale=_RESCALE,
            ),
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps("1x0e"), rescale=_RESCALE),
        )
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)

        self.apply(self._init_weights)

    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
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

        for i in range(self.long_num_layers):
            if i != (self.long_num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
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
            self.long_blocks.append(blk)

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
    def forward(self, data) -> torch.Tensor:
        pos = data.pos
        batch = data.batch

        # TODO: different group normalization
        group_batch = scatter(batch, data.labels, reduce="mean", dim=0).long().squeeze()
        group_batch = group_batch + batch.max() + 1
        node_atom = data.atomic_numbers.long()
        labels = data.labels
        pos = pos.requires_grad_(True)

        group_pos = scatter(pos * node_atom, labels, reduce="sum", dim=0) / scatter(
            node_atom, labels, reduce="sum", dim=0
        )
        node_id, group_id = data.interaction_graph[0], data.interaction_graph[1]
        node_group_dis = torch.sqrt(
            torch.sum((pos[node_id] - group_pos[group_id]) ** 2, dim=1)
        )
        data.interaction_graph = data.interaction_graph[
            :, node_group_dis <= self.long_cutoff_upper
        ]

        # process the short range graph
        edge_src = data.edge_index[0]
        edge_dst = data.edge_index[1]
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr,
            x=edge_vec,
            normalize=True,
            normalization="component",
        )
        atom_embedding, atom_attr, atom_onehot = self.atom_embed(node_atom.squeeze())
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)
        edge_degree_embedding = self.edge_deg_embed(
            atom_embedding, edge_sh, edge_length_embedding, edge_src, edge_dst, batch
        )
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        node_features_short = node_features
        for i, blk in enumerate(self.blocks):
            node_features_short = blk(
                node_input=node_features_short,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                batch=batch,
            )
            if i == len(self.blocks) - 2:
                node_features_long = node_features_short.clone()
        # process the long range graph
        edge_des_int = data.interaction_graph[0]  # node
        edge_src_int = data.interaction_graph[1] + node_features_short.shape[0]  # group
        edge_vec_int = (
            pos.index_select(0, edge_des_int)
            - group_pos.index_select(0, edge_src_int - node_features.shape[0])
            + 1e-8
        )

        edge_sh_int = o3.spherical_harmonics(
            l=self.irreps_edge_attr,
            x=edge_vec_int,
            normalize=True,
            normalization="component",
        )

        edge_length_int = (edge_vec_int).norm(dim=1)
        edge_length_embedding_int = self.rbf(edge_length_int)
        # TODO: Split long and short embeddings
        for blk in self.long_blocks:
            group_features = self.IrrepsScatter(node_features_long, labels)

            new_node_features = torch.cat([node_features_long, group_features], dim=0)
            node_idx = torch.arange(
                node_features_short.shape[0], device=node_features_short.device
            )
            new_node_attr = torch.ones_like(new_node_features.narrow(1, 0, 1))
            new_batch = torch.cat([batch, group_batch], dim=0)
            updated_node_featuees = blk(
                node_input=new_node_features,
                node_attr=new_node_attr,
                edge_src=edge_src_int,
                edge_dst=edge_des_int,
                edge_attr=edge_sh_int,
                edge_scalars=edge_length_embedding_int,
                batch=new_batch,
            )
            node_features_long = updated_node_featuees[node_idx, :]

        node_features_short = self.norm_short(node_features_short)
        node_features_long = self.norm_long(node_features_long)
        node_features_out, _ = self.concat([node_features_short, node_features_long])
        if self.out_dropout is not None:
            node_features_out = self.out_dropout(node_features_out)
        outputs = self.head(node_features_out)
        outputs = self.scale_scatter(outputs, batch, dim=0)

        outputs = outputs * self.task_std + self.task_mean
        if self.scale is not None:
            outputs = self.scale * outputs

        energy = outputs

        forces = -1 * (
            torch.autograd.grad(
                energy,
                pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
        )

        return {"energy": energy, "forces": forces}


@register_model
def dot_product_attention_transformer_exp_l2_md17_lsrm(
    irreps_in,
    radius,
    num_basis=128,
    atomref=None,
    task_mean=None,
    task_std=None,
    num_layers=6,
    long_num_layers=2,
    **kwargs,
):
    model = DotProductAttentionTransformerMD17LSRM(
        irreps_in=irreps_in,
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=num_layers,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=radius,
        number_of_basis=num_basis,
        fc_neurons=[64, 64],
        basis_type="exp",
        irreps_feature="512x0e",
        irreps_head="32x0e+16x1e+8x2e",
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        irreps_mlp_mid="384x0e+192x1e+96x2e",
        norm_layer="layer",
        alpha_drop=0.0,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        mean=task_mean,
        std=task_std,
        scale=None,
        atomref=atomref,
        long_num_layers=long_num_layers,
    )
    return model
