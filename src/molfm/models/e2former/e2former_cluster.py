# -*- coding: utf-8 -*-
import os
import math
import warnings
import torch.nn.functional as F
from typing import Dict, Optional
from .triton_dr.triton_sparse_qk_autograd import sparse_qk_triton_n_tiled as triton_sparse_qk
from .triton_dr.sparse_v_agg_lastdim_autograd import sparse_v_agg_triton_n_tiled as sparse_v_agg_lastdim


import e3nn
import torch

from sklearn.cluster import KMeans
from torch_scatter import scatter_mean,scatter_sum
from e3nn import o3

    
from torch import logical_not, nn

from torch.profiler import record_function
from loguru import logger


from .module_utils import ( 
    FeedForwardNetwork_s3,
    GaussianSmearing,
    RadialProfile,
    SmoothLeakyReLU,
    SO3_Linear2Scalar_e2former,
    SO3_Linear_e2former,
    get_normalization_layer,
)
from .so2 import _init_edge_rot_mat
from .wigner6j.tensor_product import (
    DepthWiseTensorProduct_reducesameorder,
    E2TensorProductArbitraryOrder,
    Simple_TensorProduct_oTchannel,
)
from .wigner6j.so2_tensor_product import E2TensorProductSO2_FirstOrder


from .eqv2_block import EdgeDegreeEmbeddingNetwork_eqv2,MessageBlock_eqv2


from functools import partial
def construct_cluster(node_pos,node_mask=None,method = "grid",**kwargs):
    B,N = node_pos.shape[:2]
    device = node_pos.device
    if node_mask is None:
        node_mask = torch.ones(B,N).bool()
    f_node_pos = node_pos[node_mask] 
    f_batch = torch.arange(B).reshape(B, 1).repeat(1, N).to(device)[node_mask] 
    if method == "grid":
        ### bulid cluster embeddings
        grid_size = kwargs.get("grid_size",5)

        cell_indices = torch.floor(f_node_pos / grid_size).to(torch.int64) # [f_N1, 3]
        combined_ids = torch.cat([f_batch.unsqueeze(-1), cell_indices], dim=-1) # [f_N1, 4]
        unique_ids, flat_atom_clusterid = torch.unique(
            combined_ids, dim=0, return_inverse=True
        )

        
        f_cluster_pos = scatter_mean(
            f_node_pos, flat_atom_clusterid, dim=0, dim_size=unique_ids.shape[0]
        ) 

        cluster_per_mol = torch.bincount(unique_ids[:, 0], minlength=B) 
        max_clusters = torch.max(cluster_per_mol)

        cluster_mask = (torch.arange(max_clusters,device=device).reshape(1,-1).repeat(B,1))<(cluster_per_mol.reshape(-1,1))
        cluster_pos = torch.full(
            (B, max_clusters, 3),
            999,
            device=device,
            dtype=node_pos.dtype,
        ) 
        cluster_pos[cluster_mask] = f_cluster_pos
    elif method == 'kmeans':
        min_nodes_foreachGroup = kwargs.get("min_nodes_foreachGroup",12)

        atom_clusterid = torch.zeros((B, N), dtype=torch.int, device=device)
        max_clusters = math.ceil(N / min_nodes_foreachGroup)
        cluster_pos = torch.zeros(B, max_clusters, 3, device=device)+999
        cluster_mask = torch.zeros(B, max_clusters, dtype=torch.bool, device=device)


        for i in range(B):
            valid_pos = node_pos[i][node_mask[i]]  # [N_valid, 3]
            num_atoms = valid_pos.size(0)
            num_clusters = math.ceil(num_atoms / min_nodes_foreachGroup)

            # KMeans 
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            atom_clusterid_np = kmeans.fit_predict(valid_pos.cpu().detach().numpy())  # [N_valid]
            
            atom_clusterid[i][node_mask[i]] = torch.tensor(atom_clusterid_np, device=device)
            cluster_pos[i,:num_clusters] = scatter_mean(valid_pos,torch.tensor(atom_clusterid_np, device=device).long(),dim = 0,dim_size = num_clusters)
            cluster_mask[i,:num_clusters] = 1


        cluster_ptr = torch.cat(
            [
                torch.zeros(1,dtype = torch.int32,device=node_pos.device),
                torch.cumsum(torch.sum(cluster_mask, dim=-1), dim=-1),
            ],
            dim=0,
        )
        flat_atom_clusterid = (atom_clusterid+cluster_ptr[:B].unsqueeze(dim = 1))[node_mask]

    return cluster_pos,cluster_mask,flat_atom_clusterid



class E2AttentionArbOrder_sparse_forcluster(torch.nn.Module):
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
        attn_type="first-order", ## second-order
        neighbor_weight=None,
        atom_type_cnt=256,
        **kwargs,
    ):
        super().__init__()
        self.atom_type_cnt = atom_type_cnt
        self.neighbor_weight = neighbor_weight
        self.irreps_node_input = (
            e3nn.o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.scalar_dim = self.irreps_node_input[0][0]  # scalar_dim x 0e
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        irreps_head = e3nn.o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        
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

        self.source_embedding = nn.Embedding(
                self.atom_type_cnt, self.node_embed_dim
            )
        self.target_embedding = nn.Embedding(
                self.atom_type_cnt, self.node_embed_dim
            )
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
        self.edge_query_linear = SO3_Linear2Scalar_e2former(
            self.irreps_node_input[0][0],
            self.node_embed_dim,
            lmax=self.lmax,
            hidden_features = self.node_embed_dim
        )
        self.alpha_act = SmoothLeakyReLU(0.2)
        # *3 means, rij, src_embedding, tgt_embedding
        self.edge_channel_list = [attn_weight_input_dim + self.node_embed_dim + self.scalar_dim,
                                  attn_weight_input_dim,
                                  attn_weight_input_dim]
        self.alpha_dropout = torch.nn.Dropout(alpha_drop)


        hidden_features = None
        self.query_linear = SO3_Linear2Scalar_e2former(
            self.irreps_node_input[0][0],
            num_attn_heads * self.attn_scalar_head,
            lmax=self.lmax,
            hidden_features = hidden_features
        )
        self.key_linear = nn.Sequential(
                SO3_Linear2Scalar_e2former(
                self.irreps_node_input[0][0],
                num_attn_heads * self.attn_scalar_head,
                lmax=self.lmax,
                hidden_features = hidden_features
            ),
            nn.LayerNorm(num_attn_heads * self.attn_scalar_head),
            nn.SiLU(),
            nn.Linear(num_attn_heads * self.attn_scalar_head,num_attn_heads * self.attn_scalar_head),

        )

        self.fc_easy = RadialProfile(self.edge_channel_list+[self.num_attn_heads])
        
        self.proj_value = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                lmax=self.lmax,
            )

        self.pos_embedding_proj = nn.Linear(self.attn_weight_input_dim, self.scalar_dim*2)
        self.node_scalar_proj = nn.Linear(self.scalar_dim, self.scalar_dim*2)

        if self.attn_type == "zero-order":
            self.rad_func_intputhead = nn.Sequential(RadialProfile(self.edge_channel_list+
                                                [self.num_attn_heads]),nn.Sigmoid())
        
            self.proj_zero = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
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
        f_sparse_idx_expnode,
        poly_dist,
        attn_mask,  # non-adj is True
        batch=None,
        batched_data=None,
        cluster_pos=None,
        cluster_irreps_input=None,
        **kwargs,
    ):
        """

        irreps_in = o3.Irreps("256x0e+256x1e+256x2e")
        B,L = 4,20
        node_pos = torch.randn(B,L,3)
        node_dis = torch.sqrt(torch.sum((node_pos.view(B,L,1,3)-node_pos.view(B,1,L,3))**2,dim = -1))
        attn_scalar_head = 32
        func = E2AttentionSecondOrder(
            irreps_node_input = irreps_in,
            attn_weight_input_dim = 32, # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads = 8,
            attn_scalar_head = attn_scalar_head,
            irreps_head = "32x0e+32x1e+32x2e",
            alpha_drop=0.1,
            tp_type='easy_alpha'
        )
        out = func(node_pos,
            torch.randn(B,L,9,256),
            node_dis,
            torch.randn(B,L,L,3),
            torch.randn(B,L,L,attn_scalar_head),
            atomic_numbers = torch.randint(0,19,(B,L)),
            attn_mask = torch.randn(B,L,L,1)>0)
        print(out.shape)
        """
        f_N1, _, hidden = node_irreps_input.shape
        topK = attn_weight.shape[1]
        f_N2 = cluster_irreps_input.shape[0]

        attn_weight = attn_weight.masked_fill(attn_mask, 0)
        tgt_node = self.target_embedding(atomic_numbers)
        x_edge = torch.cat(
                [
                    attn_weight,
                    tgt_node.reshape(f_N1, 1, -1).repeat(1,topK,1),
                    cluster_irreps_input[:,0][f_sparse_idx_expnode],
                ],
                dim=-1,
            )
        
        


        ## QK alpha
        query = self.query_linear(node_irreps_input).reshape(f_N1,self.num_attn_heads,-1)
        key = self.key_linear(cluster_irreps_input).reshape(f_N2, self.num_attn_heads, -1)
        value = self.proj_value(cluster_irreps_input)



        if self.use_triton:
            gate  = self.fc_easy(x_edge).contiguous()                  # [N,K,H]
            scale = 1.0 / math.sqrt(query.shape[-1])
            alpha_pre = triton_sparse_qk(query, key, f_sparse_idx_expnode, gate, scale)  # [f_N1, topK, num_heads]
    
            alpha = self.alpha_act(alpha_pre)  
            
        else:
            key = key[f_sparse_idx_expnode]

            alpha = self.alpha_act(
                self.fc_easy(x_edge)
                * torch.sum(query.unsqueeze(dim=1) * key, dim=3)
                / math.sqrt(query.shape[-1])
            )
                     

        poly_dist = poly_dist.to(alpha.dtype)
        alpha = alpha - alpha.max(dim=1, keepdim=True).values
        alpha = torch.exp(alpha) * poly_dist.unsqueeze(-1)
        alpha = alpha.masked_fill(attn_mask, 0)
        alpha = (alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-3))


        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        

        if self.use_triton:
            if self.use_triton == 2:
                raise ValueError("use triton 2 not supported")
                triton_kernel = partial(sparse_v_agg_lastdim_Hand,
                                    idx=f_sparse_idx_expnode.contiguous(),        # [N, K]
                                                alpha=alpha.contiguous(),                                  # [N, K, C]
                                                )

            else:
                triton_kernel = partial(sparse_v_agg_lastdim,
                                    idx=f_sparse_idx_expnode.contiguous(),        # [N, K]
                                                alpha=alpha.contiguous(),                                  # [N, K, C]
                                                )
        else:
            triton_kernel = None


        inputhead  = self.rad_func_intputhead(x_edge)
        alpha = alpha.reshape(f_N1,-1,self.num_attn_heads,1) * inputhead.reshape(alpha.shape[:2]+
                                                                                (self.num_attn_heads,-1)
                                                                                )
        
        alpha = alpha.reshape(alpha.shape[:2]+(-1,))

        value = value.reshape(f_N2,-1,self.num_attn_heads)
        if triton_kernel is not None:
            # value shoudl be N*head*C
            agg = triton_kernel(value=value)
        else:
            agg = torch.sum(
                    alpha.unsqueeze(dim=2)
                    * value[f_sparse_idx_expnode],
                    dim=1,
                    )
        agg = agg.reshape(f_N1,(self.lmax+1)**2,self.irreps_node_input[0][0])
        node_output = self.proj_zero(agg)
        return node_output,attn_weight

class E2formerCluster(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding="128x0e+128x1e+128x2e",
        long_range_layers=2,
        short_max_radius = 4.5,
        max_neighbors=20,
        long_cutoff_lower=2.5,
        long_cutoff_upper=12.0,
        basis_type="gaussian",
        number_of_basis=128,
        num_attn_heads=4,
        attn_scalar_head=32,
        irreps_head="32x0e+128x1e+128x2e",
        norm_layer="rms_norm_sh",  # the default is deprecated
        alpha_drop=0.1,
        tp_type="QK_alpha",
        attn_type="zero-order",
        with_cluster=True,
        **kwargs,
    ):
        super().__init__()
        self.tp_type = tp_type
        self.attn_type = attn_type
        self.max_neighbors = max_neighbors
        self.max_radius = short_max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.norm_layer = norm_layer

        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.irreps_head = irreps_head
        self.long_range_layers = long_range_layers
        self.scalar_dim = self.irreps_node_embedding[0][0]
        self.lmax = len(self.irreps_node_embedding) - 1
        self.long_cutoff_upper = long_cutoff_upper 
        self.long_cutoff_lower = long_cutoff_lower
        self.rbf_cluster = GaussianSmearing(
                self.number_of_basis, cutoff=self.long_cutoff_upper, basis_width_scalar=2
            )

        from .lsrm import Long_Range
        self.lsrm_demo = None
        if with_cluster == "lsrm":
            self.lsrm_demo = Long_Range(self.irreps_node_embedding[0][0],
                            cutoff_upper = self.long_cutoff_upper,
                            layers = 2
                                                        )
        self.cluster_blocks = torch.nn.ModuleList()
        self.norm_ffn = torch.nn.ModuleList()
        self.pre_norm_node = torch.nn.ModuleList()
        self.pre_norm_cluster = torch.nn.ModuleList()
        self.pre_linear = torch.nn.ModuleList()
        self.avg_weight = torch.nn.ModuleList()
        for _ in range(self.long_range_layers):
            self.pre_linear.append(FeedForwardNetwork_s3(self.scalar_dim,
                                                2*self.scalar_dim,
                                                2*self.scalar_dim,
                                                lmax = self.lmax,
                                                ))
            self.avg_weight.append(nn.Sequential(
                                SO3_Linear2Scalar_e2former(self.scalar_dim, 1, self.lmax, hidden_features = self.scalar_dim),nn.Sigmoid()))
            ga = E2AttentionArbOrder_sparse_forcluster(
                self.irreps_node_embedding,
                self.number_of_basis,
                num_attn_heads,
                attn_scalar_head,
                irreps_head,
                alpha_drop=alpha_drop,
                attn_type=attn_type,
                tp_type=tp_type,
            )
            self.cluster_blocks.append(ga)
            self.norm_ffn.append(nn.Sequential(
                                    get_normalization_layer(
                                                norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
                                                ),
                                    FeedForwardNetwork_s3(self.scalar_dim,
                                                self.scalar_dim,
                                                self.scalar_dim,
                                                lmax = self.lmax,
                                                )))
            self.pre_norm_node.append(get_normalization_layer(
                                            norm_layer, lmax=self.lmax, 
                                            num_channels=self.scalar_dim
                                            )
                                        )
            self.pre_norm_cluster.append(
                            nn.Sequential(FeedForwardNetwork_s3(self.scalar_dim,
                                                self.scalar_dim,
                                                self.scalar_dim,
                                                lmax = self.lmax,
                                                ),
                                get_normalization_layer(
                                            norm_layer, lmax=self.lmax, 
                                            num_channels=self.scalar_dim
                                            ))
            )
        self.drop_path = None
        self.norm_cluster = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
        )


        self.final_linear = SO3_Linear_e2former(
            self.scalar_dim*2, self.scalar_dim, lmax=self.lmax, bias=True
        )

    def reset_parameters(self):
        warnings.warn("sorry, output model not implement reset parameters")


    def forward(
        self,
        batched_data: Dict,
        f_node_irreps,
        cluster_neighbor_info,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        
        node_pos = batched_data["pos"]
        padding_mask = ~batched_data["atom_masks"]
        node_mask = logical_not(padding_mask)
 
        f_N1, L, D = f_node_irreps.shape 
        B,N = node_pos.shape[:2]
        device = node_pos.device
        f_node_pos = node_pos[node_mask] 
        f_batch = torch.arange(B).reshape(B, 1).repeat(1, N).to(device)[node_mask] 

        flat_atom_clusterid = cluster_neighbor_info["flat_atom_clusterid"]


        node_cluster_neighbor = cluster_neighbor_info
        f_cluster_pos = node_cluster_neighbor["f_cluster_pos"]
        f_edge_vec= node_cluster_neighbor["f_edge_vec"]
        f_dist= node_cluster_neighbor["f_dist"]
        f_poly_dist= node_cluster_neighbor["f_poly_dist"]
        f_attn_mask = node_cluster_neighbor["f_attn_mask"]
        f_sparse_idx_expnode = node_cluster_neighbor["f_sparse_idx_expnode"]
        f_dist_embedding = self.rbf_cluster(f_dist)  # flattn_N* topK* self.number_of_basis)
        batched_data.update(node_cluster_neighbor)


        f_node_irreps_short = f_node_irreps

        f_atomic_numbers = batched_data["atomic_numbers"].reshape(B, N)[node_mask] 

        f_outcell_index = batched_data["f_outcell_index"]

        for i, blk in enumerate(self.cluster_blocks):
            f_node_irreps_res = f_node_irreps


            aggregate_weight = self.avg_weight[i](f_node_irreps)[f_outcell_index]  # irreps 2 scalar
            aggregate_weight = aggregate_weight/(1e-5+scatter_sum(
                aggregate_weight, flat_atom_clusterid, dim=0, dim_size=cluster_neighbor_info["f_cluster_pos"].shape[0]
            )[flat_atom_clusterid])

            aggregate_weight = aggregate_weight.reshape(-1,1)
            f_cluster_irreps = scatter_sum(aggregate_weight*f_node_irreps.reshape(-1, L*D)[f_outcell_index], flat_atom_clusterid, dim=0).reshape(-1, L, D)  # [B*max_clusters, D]       
            
            # non-adj is True
            f_node_irreps = self.pre_norm_node[i](f_node_irreps[:,:,:self.scalar_dim])
            f_cluster_irreps = self.pre_norm_cluster[i](f_cluster_irreps)
            f_node_irreps_res = f_node_irreps
            f_node_irreps, _ = blk(
                node_pos=f_node_pos,
                node_irreps_input=f_node_irreps,
                edge_dis=f_dist,
                edge_vec=f_edge_vec,
                attn_weight=f_dist_embedding,
                f_sparse_idx_expnode=f_sparse_idx_expnode,
                poly_dist=f_poly_dist,
                attn_mask=f_attn_mask,
                atomic_numbers=f_atomic_numbers,
                batched_data=batched_data,
                cluster_pos = f_cluster_pos,
                cluster_irreps_input=f_cluster_irreps,
                batch = f_batch
            )
            if self.drop_path is not None:
                f_node_irreps = self.drop_path(f_node_irreps,f_batch)

            f_node_irreps = f_node_irreps + f_node_irreps_res
            f_node_irreps_res = f_node_irreps
            f_node_irreps = self.norm_ffn[i](f_node_irreps)
            if self.drop_path is not None:
                f_node_irreps = self.drop_path(f_node_irreps,f_batch)
            f_node_irreps = f_node_irreps_res + f_node_irreps

            
        f_node_irreps_norm = self.norm_cluster(f_node_irreps)
        f_node_irreps_final = self.final_linear(torch.cat([f_node_irreps_short,f_node_irreps_norm],dim=-1))
        

        return f_node_irreps_final

class E2AttentionArbOrder_sparse_formixcluster(torch.nn.Module):
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
        attn_type="first-order", ## second-order
        neighbor_weight=None,
        atom_type_cnt=256,
        **kwargs):
        super().__init__()
        self.atom_type_cnt = atom_type_cnt
        self.neighbor_weight = neighbor_weight
        self.irreps_node_input = (
            e3nn.o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.scalar_dim = self.irreps_node_input[0][0]  # scalar_dim x 0e
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        irreps_head = e3nn.o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        
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

        self.source_embedding = nn.Embedding(
                self.atom_type_cnt, self.node_embed_dim
            )
        self.target_embedding = nn.Embedding(
                self.atom_type_cnt, self.node_embed_dim
            )
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        # *3 means, rij, src_embedding, tgt_embedding
        self.edge_channel_list = [attn_weight_input_dim + self.node_embed_dim + self.scalar_dim,
                                  min(128,attn_weight_input_dim//2),
                                  min(128,attn_weight_input_dim//2)]
        self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        hidden_features = None
        if self.tp_type == "QK_light":
            hidden_features =  num_attn_heads * self.attn_scalar_head // 4
        
        
        self.query_linear = SO3_Linear2Scalar_e2former(
            self.irreps_node_input[0][0],
            num_attn_heads * self.attn_scalar_head,
            lmax=self.lmax,
        )
        self.cluster_key_linear = SO3_Linear2Scalar_e2former(
            self.irreps_node_input[0][0],
            num_attn_heads * self.attn_scalar_head,
            lmax=self.lmax,
            hidden_features=hidden_features
        )
        self.cluster_value_linear = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                lmax=self.lmax,
            )
        
        self.node_key_linear = SO3_Linear2Scalar_e2former(
            self.irreps_node_input[0][0],
            num_attn_heads * self.attn_scalar_head,
            lmax=self.lmax,
            hidden_features=hidden_features
        )
        self.node_value_linear = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                lmax=self.lmax,
            )

        self.fc_easy = RadialProfile(self.edge_channel_list+[2*self.num_attn_heads])
        

        
        self.pos_embedding_proj = nn.Linear(self.attn_weight_input_dim, self.scalar_dim*2)
        self.node_scalar_proj = nn.Linear(self.scalar_dim, self.scalar_dim*2)

        self.rad_func_intputhead = RadialProfile(self.edge_channel_list+
                                            [self.num_attn_heads])
    
        self.proj_zero = SO3_Linear_e2former(
            self.irreps_node_input[0][0],
            self.irreps_node_output[0][0],
            lmax=self.lmax,
        )
            
            
        

    def forward(
        self,
        # node_pos,
        node_irreps_input,
        # edge_dis,
        # edge_vec,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        f_sparse_idx_expnode,
        poly_dist=None,
        attn_mask=None,  # non-adj is True
        batched_data=None,
        cluster_irreps_input=None,
        **kwargs,
    ):
        """

        irreps_in = o3.Irreps("256x0e+256x1e+256x2e")
        B,L = 4,20
        node_pos = torch.randn(B,L,3)
        node_dis = torch.sqrt(torch.sum((node_pos.view(B,L,1,3)-node_pos.view(B,1,L,3))**2,dim = -1))
        attn_scalar_head = 32
        func = E2AttentionSecondOrder(
            irreps_node_input = irreps_in,
            attn_weight_input_dim = 32, # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads = 8,
            attn_scalar_head = attn_scalar_head,
            irreps_head = "32x0e+32x1e+32x2e",
            alpha_drop=0.1,
            tp_type='easy_alpha'
        )
        out = func(node_pos,
            torch.randn(B,L,9,256),
            node_dis,
            torch.randn(B,L,L,3),
            torch.randn(B,L,L,attn_scalar_head),
            atomic_numbers = torch.randint(0,19,(B,L)),
            attn_mask = torch.randn(B,L,L,1)>0)
        print(out.shape)
        """
        # edge_dis: B*L*L
        # attn_weight: B*L*L*rbf_dim
        f_N1, _, hidden = node_irreps_input.shape
        topK = attn_weight.shape[1]
        f_N2 = cluster_irreps_input.shape[0]

        tgt_node = self.target_embedding(atomic_numbers)
        f_outcell_index = batched_data["f_outcell_index"]
        f_N1N2 = f_outcell_index.shape[0]+f_N2
        with record_function(f"QK linear and value linear"):

            query = self.query_linear(node_irreps_input).reshape(f_N1,self.num_attn_heads,-1)


            key_cluster = self.cluster_key_linear(cluster_irreps_input).reshape(f_N2, self.num_attn_heads, -1)
            value_cluster = self.cluster_value_linear(cluster_irreps_input)

            key_node = self.node_key_linear(node_irreps_input).reshape(f_N1, self.num_attn_heads, -1)[f_outcell_index]
            value_node = self.node_value_linear(node_irreps_input)[f_outcell_index]
            
            key = torch.cat([key_cluster,key_node],dim = 0)
            value = torch.cat([value_cluster,value_node],dim = 0)
        invariant_fea = torch.cat([cluster_irreps_input[:,0],node_irreps_input[:,0][f_outcell_index]],dim = 0)

        with record_function(f"rbf"):
                
            x_edge = torch.cat(
                    [
                        attn_weight,
                        tgt_node.reshape(f_N1, 1, -1).repeat(1,topK,1),
                        invariant_fea[f_sparse_idx_expnode],
                    ],
                    dim=-1,
                )
            
            
            edge_gate  = self.fc_easy(x_edge)
            gate = edge_gate[...,:self.num_attn_heads]

            inputhead = edge_gate[...,self.num_attn_heads:].reshape(
                (f_N1,topK,self.num_attn_heads)
            )
        with record_function(f"triton qk"):
    
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
        with record_function(f"triton Value"):

            if self.use_triton:
                triton_kernel = partial(sparse_v_agg_lastdim,
                                    idx=f_sparse_idx_expnode.contiguous(),        # [N, K]
                                                alpha=alpha.contiguous(),                                  # [N, K, C]
                                                )
            else:
                triton_kernel = None

            alpha = alpha.reshape(f_N1, topK, self.num_attn_heads) * inputhead


            value = value.reshape(value.shape[0],-1,self.num_attn_heads)
            if triton_kernel is not None:
                # value shoudl be N*head*C
                agg = triton_kernel(value=value)
            else:
                agg = torch.sum(
                        alpha.unsqueeze(dim=2)
                        * value[f_sparse_idx_expnode],
                        dim=1,
                        )
            agg = agg.reshape(f_N1,(self.lmax+1)**2,self.irreps_node_input[0][0])
            node_output = self.proj_zero(agg)
        #                             None,
        #                             value, 
        #                             f_sparse_idx_expnode,
            


        return node_output,attn_weight

class E2formerMixCluster(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding="128x0e+128x1e+128x2e",
        long_range_layers=2,
        short_max_radius = 4.5,
        max_neighbors=20,
        long_cutoff_lower=2.5,
        long_cutoff_upper=12.0,
        basis_type="gaussian",
        number_of_basis=128,
        num_attn_heads=4,
        attn_scalar_head=32,
        irreps_head="32x0e+128x1e+128x2e",
        norm_layer="rms_norm_sh",  # the default is deprecated
        alpha_drop=0.1,
        tp_type="QK_alpha",
        attn_type="zero-order",
        with_cluster=True,
        **kwargs,
    ):
        super().__init__()
        self.tp_type = tp_type
        self.attn_type = attn_type
        self.max_neighbors = max_neighbors
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.norm_layer = norm_layer

        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding) #[:lmax+1]
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.irreps_head = irreps_head
        self.long_range_layers = long_range_layers
        self.scalar_dim = self.irreps_node_embedding[0][0]
        self.lmax = len(self.irreps_node_embedding) - 1
        self.long_cutoff_lower = long_cutoff_lower
        self.long_cutoff_upper = long_cutoff_upper
        self.rbf_cluster = GaussianSmearing(
                self.number_of_basis, cutoff=long_cutoff_upper, basis_width_scalar=2
            )


        self.rbf_short = GaussianSmearing(
                self.number_of_basis, cutoff=short_max_radius, basis_width_scalar=2
            )


        from .lsrm import Long_Range
        self.lsrm_demo = None
        if with_cluster == "lsrm":
            self.lsrm_demo = Long_Range(self.irreps_node_embedding[0][0],
                            cutoff_upper = long_cutoff_upper,
                            layers = 2
                                                        )
        self.cluster_blocks = torch.nn.ModuleList()
        self.norm_ffn = torch.nn.ModuleList()
        self.pre_norm_node = torch.nn.ModuleList()
        self.pre_norm_cluster = torch.nn.ModuleList()
        self.pre_linear1 = nn.Sequential(
                                SO3_Linear_e2former(
                                    self.scalar_dim//2, self.scalar_dim, lmax=self.lmax, bias=True
                                ),
                                get_normalization_layer(
                                    norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
                                    ),
                            )
        self.pre_linear2 = SO3_Linear_e2former(
            self.scalar_dim//2, self.scalar_dim, lmax=self.lmax, bias=True
        )
        self.avg_weight = torch.nn.ModuleList()
        for _ in range(self.long_range_layers):
            ga = E2AttentionArbOrder_sparse_formixcluster(
                self.irreps_node_embedding,
                self.number_of_basis,
                num_attn_heads,
                attn_scalar_head,
                irreps_head,
                alpha_drop=alpha_drop,
                attn_type=attn_type,
                tp_type=tp_type,
            )
            self.avg_weight.append(nn.Sequential(
                                SO3_Linear2Scalar_e2former(self.scalar_dim, 1, self.lmax, hidden_features = self.scalar_dim),nn.Sigmoid()))

            self.cluster_blocks.append(ga)
            self.norm_ffn.append(nn.Sequential(
                                    get_normalization_layer(
                                                norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
                                                ),
                                    FeedForwardNetwork_s3(self.scalar_dim,
                                                self.scalar_dim,
                                                self.scalar_dim,
                                                lmax = self.lmax,
                                                )))
            self.pre_norm_node.append(get_normalization_layer(
                                            norm_layer, lmax=self.lmax, 
                                            num_channels=self.scalar_dim
                                            )
                                        )
            self.pre_norm_cluster.append(
                            nn.Sequential(
                                get_normalization_layer(
                                            norm_layer, lmax=self.lmax, 
                                            num_channels=self.scalar_dim
                                            ))
            )

        self.drop_path = None
        self.norm_cluster = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
        )


        self.final_linear = SO3_Linear_e2former(
            self.scalar_dim*2, self.scalar_dim, lmax=self.lmax, bias=True
        )

    def reset_parameters(self):
        warnings.warn("sorry, output model not implement reset parameters")


    def forward(
        self,
        batched_data: Dict,
        f_node_irreps,
        cluster_neighbor_info,
        **kwargs,
    ) -> torch.Tensor:
        
        node_pos = batched_data["pos"]
        padding_mask = ~batched_data["atom_masks"]
        node_mask = logical_not(padding_mask)
 
        ########################## cluster part #################
        f_node_irreps = f_node_irreps[:,:(self.lmax+1)**2]
        ### bulid cluster embeddings
        f_N1, L, D = f_node_irreps.shape 
        B,N = node_pos.shape[:2]
        device = node_pos.device
        f_node_pos = node_pos[node_mask] 
        f_batch = torch.arange(B).reshape(B, 1).repeat(1, N).to(device)[node_mask] 


        flat_atom_clusterid = cluster_neighbor_info["flat_atom_clusterid"]
        f_poly_dist = cluster_neighbor_info["f_poly_dist"]
        f_attn_mask = cluster_neighbor_info["f_attn_mask"]
        f_dist_embedding = torch.cat([self.rbf_cluster(cluster_neighbor_info["f_dist_cluster"]),
                                    self.rbf_short(cluster_neighbor_info["f_dist_node"])],dim = 1)
        f_sparse_idx_expnode = cluster_neighbor_info["f_sparse_idx_expnode"]

        f_node_irreps_short = f_node_irreps
        f_atomic_numbers = batched_data["atomic_numbers"].reshape(B, N)[node_mask] 
        f_outcell_index = batched_data["f_outcell_index"]
        for i, blk in enumerate(self.cluster_blocks):
            with record_function(f"cluster block{i}"):
                f_node_irreps_res = f_node_irreps


                aggregate_weight = self.avg_weight[i](f_node_irreps)[f_outcell_index]  # irreps 2 scalar
                aggregate_weight = aggregate_weight/(1e-5+scatter_sum(
                    aggregate_weight, flat_atom_clusterid, dim=0, dim_size=cluster_neighbor_info["f_cluster_pos"].shape[0]
                )[flat_atom_clusterid])

                aggregate_weight = aggregate_weight.reshape(-1,1)
                f_node_irreps = self.pre_norm_node[i](f_node_irreps)
                f_cluster_irreps = scatter_sum(aggregate_weight*f_node_irreps.reshape(-1, L*D)[f_outcell_index], flat_atom_clusterid, dim=0).reshape(-1, L, D)  # [B*max_clusters, D]       
                with record_function(f"cluster GA"):

                    f_node_irreps, _ = blk(
                        node_irreps_input=f_node_irreps,
                        attn_weight=f_dist_embedding,
                        atomic_numbers=f_atomic_numbers,
                        f_sparse_idx_expnode=f_sparse_idx_expnode,
                        poly_dist=f_poly_dist,
                        attn_mask=f_attn_mask,
                        cluster_irreps_input=f_cluster_irreps,
                        batched_data = batched_data
                    )
                    if self.drop_path is not None:
                        f_node_irreps = self.drop_path(f_node_irreps,f_batch)

                f_node_irreps = f_node_irreps + f_node_irreps_res
                f_node_irreps_res = f_node_irreps
                f_node_irreps = self.norm_ffn[i](f_node_irreps)
                if self.drop_path is not None:
                    f_node_irreps = self.drop_path(f_node_irreps,f_batch)
                f_node_irreps = f_node_irreps_res + f_node_irreps

        f_node_irreps_norm = self.norm_cluster(f_node_irreps)
        f_node_irreps_final = self.final_linear(torch.cat([f_node_irreps_short,f_node_irreps_norm],dim=-1))
        

        return f_node_irreps_final
