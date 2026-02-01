# -*- coding: utf-8 -*-
import os
import math
from tabnanny import check
import warnings
from typing import Dict, Optional
from functools import partial

import torch.distributed as dist

import e3nn
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
    
from torch import logical_not, nn
from torch.profiler import record_function
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F



from loguru import logger
from .module_utils import init_edge_rot_euler_angles,eulers_to_wigner # 
from .e2former_cluster import E2formerMixCluster,E2formerCluster
from .triton_dr.triton_sparse_qk_autograd import sparse_qk_triton_n_tiled as triton_sparse_qk
from .triton_dr.sparse_v_agg_lastdim_autograd import sparse_v_agg_triton_n_tiled as sparse_v_agg_lastdim

# # QM9
from .module_utils import (  # ,\; EquivariantInstanceNorm,EquivariantGraphNorm; EquivariantRMSNormArraySphericalHarmonicsV2,; GaussianLayer,; irreps2gate,; sort_irreps_even_first,
    FeedForwardNetwork_s3,
    GaussianSmearing,
    # Learn_PolynomialDistance,
    RadialProfile,
    SmoothLeakyReLU,
    # SO3_Grid,
    SO3_Linear2Scalar_e2former,
    SO3_Linear_e2former,
    get_normalization_layer,
    sph_fromxyz
)
from .wigner6j.tensor_product import (
    # DepthWiseTensorProduct_reducesameorder,
    E2TensorProductArbitraryOrder,
    # Simple_TensorProduct_oTchannel,
)
from .wigner6j.so2_tensor_product import E2TensorProductSO2_FirstOrder






def irreps_times(irreps, factor):
    out = [(int(mul * factor), ir) for mul, ir in irreps if mul > 0]
    return e3nn.o3.Irreps(out)


class E2AttentionArbOrder_sparse(torch.nn.Module):
    """
    Use IrrepsLinear with external weights W(|r_i|)

    """

    def __init__(
        self,
        irreps_node_input="256x0e+256x1e+256x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+32x1e+32x2e",
        alpha_drop=0.1,
        tp_type="QK_alpha",
        attn_type="first-order",  ## second-order
        atom_type_cnt=256,
        norm_layer="identity",
        **kwargs,
    ):
        super().__init__()
        self.atom_type_cnt = atom_type_cnt
        self.irreps_node_input = (
            e3nn.o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.scalar_dim = self.irreps_node_input[0][0]  # scalar_dim x 0e
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        irreps_head = (
            e3nn.o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        )

        self.irreps_head = irreps_head
        # irreps_node_output,  attention will not change the input shape/embeding length
        self.irreps_node_output = self.irreps_node_input
        self.lmax = self.irreps_node_input[-1][1][0]
        # new params
        self.attn_type = attn_type
        self.tp_type = tp_type.split("+")[0]
        self.use_triton = "triton" in tp_type
        if "triton2" in tp_type:
            self.use_triton = 2
        self.node_embed_dim = 128

        self.source_embedding = nn.Embedding(self.atom_type_cnt, self.node_embed_dim)
        self.target_embedding = nn.Embedding(self.atom_type_cnt, self.node_embed_dim)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.alpha_act = SmoothLeakyReLU(0.2)
        # *3 means, rij, src_embedding, tgt_embedding
        self.edge_channel_list = [
            attn_weight_input_dim + self.node_embed_dim * 2,
            min(128, attn_weight_input_dim // 2),
            min(128, attn_weight_input_dim // 2),
        ]
        self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        
        hidden_features = None
        
        if self.tp_type == "QKdot_alpha":
            self.dot_linear = SO3_Linear_e2former(
                    self.irreps_node_input[0][0],
                    attn_weight_input_dim,
                    lmax=self.lmax,
                )
            self.rad_func_m0 = RadialProfile(self.edge_channel_list+
                                            [self.attn_weight_input_dim * (self.lmax+1)])
            self.fc_m0 = nn.Linear(self.attn_weight_input_dim*(self.lmax+1),self.num_attn_heads)

            self.fc_easy = RadialProfile(
                self.edge_channel_list + [self.num_attn_heads]
            )
        elif self.tp_type == "QKdotS_alpha":
            self.sum_vec = torch.zeros([(self.lmax+1)**2,self.lmax+1])
            for l in range(self.lmax+1):
                self.sum_vec[l**2:(l+1)**2,l] = 1
            self.sum_vec = torch.nn.Parameter(self.sum_vec,requires_grad=False)
            self.dot_linear = SO3_Linear_e2former(
                    self.irreps_node_input[0][0],
                    self.num_attn_heads//(self.lmax+1)+1,
                    lmax=self.lmax,
                )
            self.direction_linear = nn.Sequential(nn.Linear(self.num_attn_heads,self.num_attn_heads),
                                                    nn.LayerNorm(self.num_attn_heads),
                                                    nn.SiLU(),
                                                    nn.Linear(self.num_attn_heads,self.num_attn_heads))
            self.fc_easy = RadialProfile(
                self.edge_channel_list + [self.num_attn_heads + self.num_attn_heads] #+ self.num_attn_heads * (1+self.lmax)
            )
        else:

            self.fc_easy = RadialProfile(
                self.edge_channel_list + [2*self.num_attn_heads]
            )
        self.query_linear = SO3_Linear2Scalar_e2former(
            self.irreps_node_input[0][0],
            num_attn_heads * self.attn_scalar_head,
            lmax=self.lmax,
            hidden_features = hidden_features
        )
        self.key_linear = SO3_Linear2Scalar_e2former(
            self.irreps_node_input[0][0],
            num_attn_heads * self.attn_scalar_head,
            lmax=self.lmax,
            hidden_features = hidden_features
        )

    
        self.proj_value = SO3_Linear_e2former(
            self.irreps_node_input[0][0],
            self.irreps_node_input[0][0],
            lmax=self.lmax,
        )
        
        if self.attn_type == "zero-order":
            self.proj_zero = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )

        elif self.attn_type == "so2-first-order":
            self.first_order_tp = E2TensorProductSO2_FirstOrder(self.irreps_node_input, 
                                                                (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
                                                                order = 1,
                                                                head = self.num_attn_heads,
                                                                
                                                            )
            self.proj_first = SO3_Linear_e2former(
                num_attn_heads * self.irreps_head[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )
        
        

    def forward(
        self,
        node_pos,
        node_irreps_input,
        edge_dis,
        edge_vec,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        poly_dist=None,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        **kwargs,
    ):

        f_N1, _, hidden = node_irreps_input.shape
        topK = attn_weight.shape[1]


        value = node_irreps_input 
        f_N2 = batched_data["f_outcell_index"].shape[0]

        tgt_node = self.target_embedding(atomic_numbers)
        src_node = self.source_embedding(atomic_numbers)[batched_data["f_outcell_index"]]
        f_sparse_idx_expnode = batched_data["f_sparse_idx_expnode"]
        with record_function("query key linear function"):

            query = self.query_linear(node_irreps_input).reshape(
                f_N1, self.num_attn_heads, -1
            )
            key = self.key_linear(node_irreps_input)[batched_data["f_outcell_index"]]
            key_lastdim = key.shape[-1]//self.num_attn_heads
        

        with record_function("rbf function-a"):
            x_edge = torch.cat(
                [
                    attn_weight,
                    tgt_node.reshape(f_N1, 1, -1).repeat(1, topK, 1),
                    src_node[f_sparse_idx_expnode],
                ],
                dim=-1,
            )


        with record_function("qk attention"):
            if self.tp_type == "QKdot_alpha":
                x_0_extra = []
                node_irreps_input_dot = self.dot_linear(node_irreps_input)
                for l in range(self.lmax+1):
                    rij_l = e3nn.o3.spherical_harmonics(l,edge_vec,normalize=True).unsqueeze(dim = -1) #B*N*N*2l+1*1
                    node_l = node_irreps_input_dot[:,l**2:(l+1)**2] #B*N*2l+1*hidden
                    x_0_extra.append(torch.sum(rij_l*node_l.unsqueeze(dim=1),dim = -2))
                edge_m0 = self.rad_func_m0(x_edge)
                gate = self.fc_m0(torch.cat(x_0_extra,dim = -1)*edge_m0)
                edge_gate  = self.fc_easy(x_edge)
                inputhead = edge_gate.reshape(
                    (f_N1,topK,self.num_attn_heads)
                )
            elif self.tp_type == "QKdotS_alpha":
                lxs = o3._spherical_harmonics._spherical_harmonics(
                    lmax=self.lmax,
                    x=edge_vec[...,0],
                    y=edge_vec[...,1],
                    z=edge_vec[...,2],
                )
                node_irreps_input_dot = self.dot_linear(node_irreps_input)
                x_0_extra = torch.einsum("nlc,nkl,ls->nksc",node_irreps_input_dot,lxs,self.sum_vec).reshape(f_N1,topK,-1)[:,:,:self.num_attn_heads]
                edge_gate  = self.fc_easy(x_edge)
                edge_m0 = edge_gate[:,:,self.num_attn_heads:]
                gate = self.direction_linear(x_0_extra*edge_m0)
                inputhead = edge_gate[...,:self.num_attn_heads].reshape(
                    (f_N1,topK,self.num_attn_heads)
                )
            else:
                edge_gate  = self.fc_easy(x_edge)
                gate = edge_gate[...,:self.num_attn_heads]

                inputhead = edge_gate[...,self.num_attn_heads:].reshape(
                    (f_N1,topK,self.num_attn_heads)
                )

            
            key = key.reshape(-1, self.num_attn_heads, key_lastdim)

            if self.use_triton:
                gate  = gate.contiguous()                  # [N,K,H]
                scale = 1.0 / math.sqrt(query.shape[-1])
                alpha = triton_sparse_qk(query, key, f_sparse_idx_expnode, gate, scale)  # [f_N1, topK, num_heads]

            else:
                key = key[f_sparse_idx_expnode]

                alpha = gate* torch.sum(query.unsqueeze(dim=1) * key, dim=3)/ math.sqrt(query.shape[-1])
                
        if self.tp_type == 'QK_softplus':
            alpha = F.softplus(alpha) * poly_dist.unsqueeze(-1)
            alpha = alpha.masked_fill(attn_mask, 0)

        else:

            alpha = alpha - alpha.max(dim=1, keepdim=True).values
            alpha = torch.exp(alpha) * poly_dist.unsqueeze(-1)
            alpha = alpha.masked_fill(attn_mask, 0)
            alpha = (alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-3))


        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)


        alpha = alpha.reshape(f_N1, topK, self.num_attn_heads) * inputhead

        triton_kernel = None
        if self.use_triton:
            triton_kernel = partial(sparse_v_agg_lastdim,)
        else:
            triton_kernel = None


        value = self.proj_value(value)[batched_data["f_outcell_index"]]
        if self.attn_type == "zero-order":
            with record_function("zero-order"):
                value = value.reshape(f_N2,-1,self.num_attn_heads)

                if self.use_triton:
                    agg = triton_kernel(value=value,idx = batched_data["f_sparse_idx_expnode"].contiguous(),alpha = alpha)
                else:
                    agg = torch.sum(
                            alpha.unsqueeze(dim=2)
                            * value[batched_data["f_sparse_idx_expnode"]],
                            dim=1,
                            )
                agg = agg.reshape(f_N1,(self.lmax+1)**2,self.irreps_node_input[0][0])
                node_output = self.proj_zero(agg)
 

        elif self.attn_type == "first-order" or self.attn_type == "so2-first-order":
            with record_function("first-order"):
                
                node_output = self.first_order_tp(
                        node_pos,
                        batched_data["f_exp_node_pos"],
                        None,
                        value,
                        alpha / (edge_dis.unsqueeze(dim=-1) + 1e-8),
                        triton_kernel= triton_kernel,
                        f_sparse_idx_expnode = batched_data["f_sparse_idx_expnode"],
                        batched_data=batched_data,
                    )
                node_output = self.proj_first(node_output)


        return node_output, attn_weight



@compile_mode("script")
class TransBlock(torch.nn.Module):
    """
    1. Layer Norm 1 -> E2Attention -> Layer Norm 2 -> FeedForwardNetwork
    2. Use pre-norm architecture
    """

    def __init__(
        self,
        irreps_node_input="256x0e+256x1e+256x2e",
        irreps_node_output="256x0e+256x1e+256x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+32x1e+32x2e",
        alpha_drop=0.1,
        tp_type="QK_alpha",
        attn_type="first-order",  ## second-order
        atom_type_cnt=256,
        norm_layer="identity",
        ffn_type = 's3',
        layer_id = 0
    ):
        super().__init__()
        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.irreps_node_output = (
            o3.Irreps(irreps_node_output)
            if isinstance(irreps_node_output, str)
            else irreps_node_output
        )

        self.lmax = irreps_node_input[-1][1][0]
        self.norm_1 = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=irreps_node_input[0][0]
        )

        func = None

        self.attn_type = attn_type

        if isinstance(attn_type, str) and attn_type.endswith("order"):
            func = E2AttentionArbOrder_sparse
        else:
            raise ValueError(
                f" sorry, the attn type is not support, please check {attn_type}"
            )
        self.attn_weight_input_dim = attn_weight_input_dim
        self.ga = func(
            irreps_node_input,
            attn_weight_input_dim,  # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads,
            attn_scalar_head,
            irreps_head,
            alpha_drop=alpha_drop,
            attn_type=attn_type,
            tp_type=tp_type,
            norm_layer=norm_layer,
            atom_type_cnt = atom_type_cnt,
        )



        self.SO3_grid = None

        self.ffn = None
        self.norm_ffn = get_normalization_layer(
                norm_layer, lmax=self.lmax, num_channels=irreps_node_input[0][0]
            )
        if ("default" in ffn_type) or ("s2" in ffn_type):
            from .uma_utils import FeedForwardNetwork_s2,SpectralAtomwise
            
            self.ffn = FeedForwardNetwork_s2(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                lmax=self.lmax,
                grid_resolution=18,
                use_grid_mlp=False,  # notice in eqv2, default is True
            )
        elif "s3" in ffn_type:
            self.ffn = FeedForwardNetwork_s3(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                lmax=self.lmax,
            )
        elif "spectral" in ffn_type:
            from .uma_utils import FeedForwardNetwork_s2,SpectralAtomwise

            self.ffn = SpectralAtomwise(
                sphere_channels=self.irreps_node_input[0][0],
                hidden_channels=self.irreps_node_input[0][0],
                lmax=self.lmax,
                mmax=None,SO3_grid=None
            )


    def forward(
        self,
        node_pos,
        node_irreps,
        sys_node_embedding,
        edge_dis,
        edge_vec,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        attn_mask,
        poly_dist=None,
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        """

        irreps_in = e3nn.o3.Irreps("256x0e+256x1e+256x2e")
        B,L = 4,100
        dis_embedding_dim = 32
        node_pos = torch.randn(B,L,3)
        edge_dis = torch.sqrt(torch.sum((node_pos.view(B,L,1,3)-node_pos.view(B,1,L,3))**2,dim = -1))
        dis_embedding = torch.randn(B,L,L,dis_embedding_dim)
        attn_mask = torch.randn(B,L,L,1)>0
        atomic_numbers = torch.randint(0,10,(B,L))
        func = TransBlock(
                irreps_in,
                irreps_in,
                attn_weight_input_dim=dis_embedding_dim, # e.g. rbf(|r_ij|) or relative pos in sequence
                num_attn_heads=8,
                attn_scalar_head = 48,
                irreps_head="32x0e+32x1e+32x2e",
                rescale_degree=False,
                nonlinear_message=False,
                alpha_drop=0.1,
                proj_drop=0,
                drop_path_rate=0.1,
                attn_type = 'second-order',
                ffn_type="eqv2ffn",
                norm_layer="rms_norm_sh_BL", # used for norm 1 and norm2
            )

        out = func.forward(
                node_pos,
                torch.randn(B,L,9,256),
                edge_dis,
                dis_embedding, # e.g. rbf(|r_ij|) or relative pos in sequence
                atomic_numbers,
                attn_mask,

                batch=None)
        """
        node_irreps = node_irreps[:,:(self.lmax+1)**2]

        node_irreps_res = node_irreps
        node_irreps = self.norm_1(node_irreps)
        if sys_node_embedding is not None:
            node_irreps[:, 0, :] = node_irreps[:, 0, :] + sys_node_embedding

        node_irreps, attn_weight = self.ga(
            node_pos=node_pos,
            node_irreps_input=node_irreps,
            edge_dis=edge_dis,
            poly_dist=poly_dist,
            edge_vec=edge_vec,
            attn_weight=attn_weight,
            atomic_numbers=atomic_numbers,
            attn_mask=attn_mask,
            batched_data=batched_data,
        )
        node_irreps = node_irreps + node_irreps_res

        with record_function("FeedForwardFunction"):
            ## residual connection
            node_irreps_res = node_irreps
            node_irreps = self.norm_ffn(node_irreps)
            node_irreps = self.ffn(node_irreps)

            node_irreps = node_irreps_res + node_irreps

        return node_irreps, attn_weight


class EdgeDegreeEmbeddingNetwork_higherorder(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding,
        avg_aggregate_num=10,
        number_of_basis=32,
        cutoff=15,
        time_embed=False,
        use_layer_norm=True,
        use_atom_edge=False,
        name="default",
        **kwargs,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.irreps_node_embedding = (
            o3.Irreps(irreps_node_embedding)
            if isinstance(irreps_node_embedding, str)
            else irreps_node_embedding
        )
        if self.irreps_node_embedding[0][1].l != 0:
            raise ValueError("node embedding must have sph order 0 embedding.")
        self.number_of_basis = number_of_basis
        self.gbf_projs = nn.ModuleList()

        self.scalar_dim = self.irreps_node_embedding[0][0]
        if time_embed:
            self.time_embed_proj = nn.Sequential(
                nn.Linear(self.scalar_dim, self.scalar_dim, bias=True),
                nn.SiLU(),
                nn.Linear(self.scalar_dim, number_of_basis, bias=True),
            )
        self.max_num_elements = 300
        self.use_atom_edge = use_atom_edge
        if use_atom_edge:
            self.source_embedding = nn.Embedding(self.max_num_elements, number_of_basis)
            self.target_embedding = nn.Embedding(self.max_num_elements, number_of_basis)
        else:
            self.source_embedding = None
            self.target_embedding = None
        self.weight_list = nn.ParameterList()
        self.lmax = len(self.irreps_node_embedding) - 1
        for idx in range(len(self.irreps_node_embedding)):
            self.gbf_projs.append(
                RadialProfile(
                    [
                        number_of_basis * 3 if use_atom_edge else number_of_basis,
                        min(number_of_basis//2, 128),
                        min(number_of_basis//2, 128),
                        self.irreps_node_embedding[idx][0],
                    ],
                    use_layer_norm=use_layer_norm,
                )
            )

        self.name = name
        
        self.sph = sph_fromxyz(lmax =  self.lmax)
        self.proj = SO3_Linear_e2former(
            self.irreps_node_embedding[idx][0] * 2,
            self.irreps_node_embedding[idx][0],
            lmax=self.lmax,
        )
        self.avg_aggregate_num = avg_aggregate_num

    def forward(
        self,
        node_input,
        node_pos,
        edge_dis,
        atomic_numbers,
        edge_vec,
        batched_data,
        attn_mask,
        poly_dist,
        edge_scalars,
        **kwargs,
    ):
        """
        model =  EdgeDegreeEmbeddingNetwork_higherorder(
                "256x0e+256x1e+256x2e",
                avg_aggregate_num=10,
                number_of_basis=32,
                cutoff=5,
                time_embed=False,
                use_atom_edge=True)

        f_N = 3+9+20
        f_N2 = 70
        topK = 5
        num_basis = 32
        hidden = 256
        node_input = None
        exp_node_pos = torch.randn(f_N2,3)

        node_pos = torch.randn(f_N,3)
        edge_dis = torch.randn(f_N,topK)
        atomic_numbers = torch.randint(0,10,(f_N,))
        edge_vec = torch.randn(f_N,topK,3)

        attn_mask = torch.randn(f_N,topK,1)>0
        edge_scalars = torch.randn(f_N,topK,num_basis)
        f_sparse_idx_node = torch.randint(0,f_N,(f_N,topK))
        f_sparse_idx_expnode = torch.randint(0,f_N2,(f_N2,topK))
        batched_data = {'f_sparse_idx_node':f_sparse_idx_node,'f_sparse_idx_expnode':f_sparse_idx_expnode}

        out = model(node_input,
                node_pos,
                edge_dis,
                atomic_numbers,
                edge_vec,
                batched_data,
                attn_mask,
                edge_scalars,)

        """

        f_sparse_idx_expnode = batched_data["f_sparse_idx_expnode"].clone()
        
        f_N1,topK = edge_vec.shape[:2]
        tgt_atm = (
            self.target_embedding(atomic_numbers).unsqueeze(dim=1).repeat(1, topK, 1)
        )
        src_atm = self.source_embedding(atomic_numbers)[batched_data["f_outcell_index"]]

        edge_dis_embed = torch.cat(
            [edge_scalars, tgt_atm, src_atm[f_sparse_idx_expnode]],
            dim=-1,
        )
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)
        with record_function("e3nn sph"):
            
            lxs = o3._spherical_harmonics._spherical_harmonics(
                    lmax=self.lmax,
                    x=edge_vec[...,0],
                    y=edge_vec[...,1],
                    z=edge_vec[...,2],
                )
            lxs = torch.where(attn_mask, 0, lxs).transpose(1, 2) #f_N1*topK*(L) ->f_N1*(L)*topK 
            lxs = lxs * poly_dist.reshape(f_N1,1,topK)
        

        node_features = []
        for idx in range(len(self.irreps_node_embedding)):
            edge_fea = self.gbf_projs[idx](edge_dis_embed)
            lx_embed = torch.matmul(lxs[:,idx**2:(idx+1)**2], edge_fea)   # (m, d, h)
            node_features.append(lx_embed)

        node_features = torch.cat(node_features, dim=1) / self.avg_aggregate_num

        return node_features  # node_features


def compile_module(module,compile_options = {"triton.cudagraphs": False}):
    try:
        return torch.compile(module, options=compile_options)
    except TypeError:
        return torch.compile(module, mode="max-autotune")

class E2former(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding="128x0e+128x1e+128x2e",
        num_layers=6,
        max_neighbors=20,
        max_radius=4.5,
        basis_type="gaussiansmear",
        number_of_basis=128,
        num_attn_heads=4,
        attn_scalar_head=32,
        irreps_head="32x0e+32x1e+32x2e",
        norm_layer="rms_norm_sh",  # the default is deprecated
        alpha_drop=0.1,
        tp_type="QK_alpha",
        attn_type="first-order",
        atom_type_cnt=256,
        edge_embedtype="default",
        ffn_type="default",
        use_compile = False,
        avg_degree=23.01,
        with_cluster=False,
        long_range_layers = 2,
        long_cutoff_lower  = 2.5,
        long_cutoff_upper = 12,
        **kwargs,
    ):
        super().__init__()
        self.config = locals()           
        del self.config['self']          


        self.tp_type = tp_type
        if "+" in attn_type:
            self.attn_type = attn_type.split("+")
        else:
            self.attn_type = [attn_type,attn_type]

        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop

        self.norm_layer = norm_layer


        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.num_layers = num_layers
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.irreps_head = irreps_head

        if "0e" not in self.irreps_node_embedding:
            raise ValueError("sorry, the irreps node embedding must have 0e embedding")



        self.default_node_embedding = torch.nn.Embedding(
            atom_type_cnt, self.irreps_node_embedding[0][0]
        )

        self._node_scalar_dim = self.irreps_node_embedding[0][0]
        self._node_vec_dim = (
            self.irreps_node_embedding.dim - self.irreps_node_embedding[0][0]
        )

        self.basis_type = basis_type
        self.heads2basis = nn.Linear(
            self.num_attn_heads, self.number_of_basis, bias=True
        )

        if self.basis_type == "gaussiansmear":
            self.rbf = GaussianSmearing(
                self.number_of_basis, cutoff=self.max_radius, basis_width_scalar=2
            )
        else:
            raise ValueError
        Jd_list =  torch.load(os.path.join(os.path.dirname(__file__), 'Jd.pt'))
        self.Jd_list = nn.ParameterList([nn.Parameter(i.float(),requires_grad=False) for i in Jd_list])

        self.edge_deg_embed_dense = EdgeDegreeEmbeddingNetwork_higherorder(
            self.irreps_node_embedding,
            avg_degree,
            cutoff=self.max_radius,
            number_of_basis=self.number_of_basis,
            use_atom_edge=True,
            use_layer_norm="wolayernorm" not in edge_embedtype,
        )
        if use_compile:
            self.edge_deg_embed_dense = compile_module(self.edge_deg_embed_dense)

        self.blocks = torch.nn.ModuleList()
        for i in range(self.num_layers):
            blk = TransBlock(
                irreps_node_input=self.irreps_node_embedding,
                irreps_node_output=self.irreps_node_embedding,
                attn_weight_input_dim=self.number_of_basis,
                num_attn_heads=self.num_attn_heads,
                attn_scalar_head=self.attn_scalar_head,
                irreps_head=self.irreps_head,
                alpha_drop=self.alpha_drop,
                tp_type=self.tp_type,
                attn_type=self.attn_type[0] if i < self.num_layers//2 else self.attn_type[1], #,
                norm_layer=self.norm_layer,
                ffn_type=ffn_type,
                layer_id=i,
                
            )        
            if use_compile:
                self.blocks.append(compile_module(blk))
            else:
                self.blocks.append(blk)

        self.scalar_dim = self.irreps_node_embedding[0][0]

        self.lmax = len(self.irreps_node_embedding) - 1
        if hasattr(self.blocks[0].ga,"first_order_tp"):
            self.l3_sequential = self.blocks[0].ga.first_order_tp.so2_tp.l3_sequential
        
        else:
            self.l3_sequential = [(l,1) for l in range(self.lmax+1)]
        
        self.norm_final = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
        )
        
        self.long_range = None
        self.long_cutoff_lower = long_cutoff_lower
        self.long_cutoff_upper = long_cutoff_upper
        self.with_cluster = str(with_cluster)
        if self.with_cluster.startswith('node'):
            self.pre_norm_node = get_normalization_layer(
                                            norm_layer, lmax=self.lmax, 
                                            num_channels=self.scalar_dim
                                            )
            self.norm_cluster = get_normalization_layer(
                norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
            )
            self.final_linear = SO3_Linear_e2former(
                self.scalar_dim*2, self.scalar_dim, lmax=self.lmax, bias=True
            )

            self.rbf_long = GaussianSmearing(
                self.number_of_basis, cutoff=self.long_cutoff_upper, basis_width_scalar=2
            )
            self.long_blocks = torch.nn.ModuleList()
            for i in range(long_range_layers):
                blk = TransBlock(
                    irreps_node_input=self.irreps_node_embedding,
                    irreps_node_output=self.irreps_node_embedding,
                    attn_weight_input_dim=self.number_of_basis,
                    num_attn_heads=self.num_attn_heads,
                    attn_scalar_head=self.attn_scalar_head,
                    irreps_head=self.irreps_head,
                    alpha_drop=self.alpha_drop,
                    tp_type=self.tp_type,
                    attn_type='zero-order', #,
                    norm_layer=self.norm_layer,
                    ffn_type=ffn_type,
                    layer_id=i,
                )
                if use_compile:
                    self.long_blocks.append(compile_module(blk))
                else:
                    self.long_blocks.append(blk)
            self.norm_final = get_normalization_layer(
                norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
            )
        elif self.with_cluster =="mixcluster":
            self.config.update({"attn_type":"zero-order",
                                "tp_type":"QK_alpha+triton",
                                "short_max_radius":self.max_radius})
            self.long_range = E2formerMixCluster(**self.config)
        elif self.with_cluster == "True":
            self.config.update({"attn_type":"zero-order",
                                "tp_type":"QK_alpha+triton",
                    "short_max_radius":self.max_radius})
            self.long_range = E2formerCluster(**self.config)
            
        
        if len(self.irreps_node_embedding) == 1:
            self.f_linear = nn.Sequential(
                nn.Linear(self.scalar_dim, self.scalar_dim),
                nn.LayerNorm(self.scalar_dim),
                nn.SiLU(),
                nn.Linear(self.scalar_dim, 3 * self.scalar_dim),
            )


    def reset_parameters(self):
        warnings.warn("sorry, output model not implement reset parameters")

    def forward(
        self,
        batched_data: Dict,
        token_embedding: torch.Tensor,
        sys_embedding: torch.Tensor,
        neighbor_info,
        cluster_neighbor_info = None,   
        padding_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor, [L, B, H].
            padding_mask (torch.Tensor): Padding mask, [B, L].
            batched_data (Dict): Input data for the forward pass.
            atomic_numbers (torch.Tensor): The masked token type, [B, L].
        Returns:
            torch.Tensor: Encoded tensor, [B, L, H].
        example:
        batch: attn_bias torch.Size([4, 65, 65])
        batch: attn_edge_type torch.Size([4, 64, 64, 1])
        batch: spatial_pos torch.Size([4, 64, 64]) -> shortest path
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
        batch: time_embed torch.Size([4, 64, node_embedding_scalar])

        example:
        import torch
        func = E2former(
                irreps_node_embedding="24x0e+16x1e+8x2e",
                num_layers=6,
                max_radius=5.0,
                basis_type="gaussian",
                number_of_basis=128,
                num_attn_heads=4,
                attn_scalar_head = 32,
                irreps_head="32x0e+16x1e+8x2e",
                alpha_drop=0,)
        batched_data = {}
        B,L = 4,100

        node_pos = torch.randn(B,L,3)*100
        token_embedding = torch.randn(B,L,24)
        token_id = torch.randint(0,100,(B,L))
        batched_data = {
            "pos":node_pos,
            # "token_embedding":token_embedding,
            "token_id":token_id,
        }
        out = func(batched_data, None, None)


        tr = o3._wigner.wigner_D(1,
                            torch.Tensor([0.8]),
                            torch.Tensor([0.5]),
                            torch.Tensor([0.3]))
        pos = (tr@(node_pos.unsqueeze(-1))).squeeze(-1)
        batched_data = {
            "pos":pos,
            # "token_embedding":token_embedding,
            "token_id":token_id,
        }
        out_tr = func(batched_data, None, None)


        print(torch.max(torch.abs(out_tr[0]-out[0])),
            torch.max(torch.abs(out_tr[1]-(tr@out[1]))))
        """

        tensortype = self.default_node_embedding.weight.dtype
        device = batched_data["pos"].device
        B, L = batched_data["pos"].shape[:2]

        node_pos = batched_data["pos"]
        padding_mask = ~batched_data["atom_masks"]
        node_mask = logical_not(padding_mask)
 
        node_pos = torch.where(
            padding_mask.unsqueeze(dim=-1).repeat(1, 1, 3), 999.0, node_pos
        )


        f_atomic_numbers = batched_data["atomic_numbers"].reshape(B, L)[node_mask]

        f_node_pos = node_pos[node_mask]
        f_N1 = f_node_pos.shape[0]

        f_exp_node_pos = batched_data["f_exp_node_pos"]
        

        f_batch = torch.arange(B).reshape(B, 1).repeat(1, L).to(device)[node_mask]


        f_edge_vec= neighbor_info["f_edge_vec"]
        f_dist= neighbor_info["f_dist"]
        f_poly_dist= neighbor_info["f_poly_dist"]
        f_attn_mask = neighbor_info["f_attn_mask"]
        f_dist_embedding = self.rbf(f_dist)  # flattn_N* topK* self.number_of_basis)

        f_atom_embedding = token_embedding[node_mask]

        
                
        wigner,wigner_inv = eulers_to_wigner(
                init_edge_rot_euler_angles(f_node_pos),
                0,
                self.lmax,
                self.Jd_list,
                self.l3_sequential
        )

        if torch.any(batched_data["pbc"]):
            wigner_exp,wigner_inv_exp = eulers_to_wigner(
                init_edge_rot_euler_angles(f_exp_node_pos),
                0,
                self.lmax,
                self.Jd_list,
                self.l3_sequential
            )
        else:
            wigner_exp = wigner
            wigner_inv_exp = wigner_inv
        neighbor_info.update({
                "wigner":wigner,
                "wigner_inv":wigner_inv,
                "wigner_exp":wigner_exp,
                "wigner_inv_exp":wigner_inv_exp
            })

        batched_data.update(neighbor_info)

        # not use sparse mode
        with record_function("node_initiliation"):

            edge_degree_embedding_dense = self.edge_deg_embed_dense(
                node_input = f_atom_embedding,
                node_pos = f_node_pos,
                edge_dis = f_dist,
                atomic_numbers=f_atomic_numbers,
                edge_vec=f_edge_vec,
                batched_data=batched_data,
                attn_mask=f_attn_mask,
                poly_dist=f_poly_dist,
                edge_scalars=f_dist_embedding
            )

        f_node_irreps = edge_degree_embedding_dense
        for i, blk in enumerate(self.blocks):
            with record_function(f"tranblock {i}"):

                f_node_irreps, f_dist_embedding = blk(
                    node_pos=f_node_pos,
                    node_irreps=f_node_irreps,
                    sys_node_embedding = None if sys_embedding is None else sys_embedding[f_batch],
                    edge_dis=f_dist,
                    edge_vec=f_edge_vec,
                    attn_weight=f_dist_embedding,
                    atomic_numbers=f_atomic_numbers,
                    attn_mask=f_attn_mask,
                    poly_dist=f_poly_dist,
                    batch=f_batch,  #
                    batched_data=batched_data

                )
        if self.with_cluster.startswith('node'):
            f_node_irreps_short = f_node_irreps
            f_node_irreps = self.pre_norm_node(f_node_irreps)
            batched_data.update(cluster_neighbor_info)
            f_dist_embedding = self.rbf_long(cluster_neighbor_info["f_dist"])
            for i, blk in enumerate(self.long_blocks):
                with record_function(f"long-tranblock {i}"):
                    f_node_irreps, f_dist_embedding = blk(
                        node_pos=f_node_pos,
                        node_irreps=f_node_irreps,
                        sys_node_embedding = None if sys_embedding is None else sys_embedding[f_batch],
                        edge_dis=cluster_neighbor_info["f_dist"],
                        edge_vec=cluster_neighbor_info["f_edge_vec"],
                        attn_weight=f_dist_embedding,
                        atomic_numbers=f_atomic_numbers,
                        attn_mask=cluster_neighbor_info["f_attn_mask"],
                        poly_dist=cluster_neighbor_info["f_poly_dist"],
                        batch=f_batch,  #
                        batched_data=batched_data
                    )
            f_node_irreps = self.norm_cluster(f_node_irreps)
            f_node_irreps = self.final_linear(torch.cat([f_node_irreps_short,f_node_irreps],dim=-1))

        elif self.with_cluster!="False":
            f_node_irreps = self.long_range(batched_data,f_node_irreps,cluster_neighbor_info=cluster_neighbor_info)
        else:
            f_node_irreps = self.norm_final(f_node_irreps)


        node_irreps = torch.zeros(
            (B, L, (self.lmax+1)**2, self._node_scalar_dim), device=device
        )
        node_irreps[node_mask] = f_node_irreps  # the part of order 0

        node_attr = node_irreps[:,:,0]
        node_vec = node_irreps[:,:,1:4]
        
        return node_attr, node_vec, node_irreps



# coeffs length is lmax+1, help normalize or refactor of spherical hamonics
def get_powers(vec,coeffs,lmax):
    out_powers = [
        coeffs[0] * torch.ones_like(vec.narrow(-1, 0, 1).unsqueeze(dim=-1))
    ]
    for i in range(1, lmax + 1):
        out_powers.append(
            coeffs[i]
            * e3nn.o3.spherical_harmonics(
                i, vec, normalize=False, normalization="integral"
            ).unsqueeze(-1)
        )

    return out_powers
