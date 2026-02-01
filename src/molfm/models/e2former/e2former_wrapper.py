# -*- coding: utf-8 -*-
# Ensure E2formerESCAIP is properly imported or define
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn as nn
import torch_geometric
from e3nn import o3
from loguru import logger

from torch_geometric.data import Data


from .e2former import E2former
from .e2former_cluster import construct_cluster
from .module_utils import CellExpander,scaled_sigmoid,construct_radius_neighbor


from .utils.graph_utils import RandomRotate


_AVG_NUM_NODES = 77.81317



class E2FormerBackbone(nn.Module):
    """
    Physics Science Module backbone model integrated with EScAIP framework.

    unique architectural features.
    """
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs

        self.cell_expander = CellExpander(
            self.kwargs.get("max_radius", 5.0),
            self.kwargs.get("expanded_token_cutoff", 512),  # deprecated
            self.kwargs.get("pbc_expanded_num_cell_per_direction", 4),
        )
        self.scalar_dim = o3.Irreps(kwargs["irreps_node_embedding"])[0][0]
        embeding_dim = self.scalar_dim if kwargs["encoder"] == "default" else kwargs["encoder_embed_dim"]

        # Token embedding layer
        self.embedding = nn.Embedding(256,embeding_dim )
        self.embedding_charge = nn.Embedding(30, embeding_dim)
        self.embedding_multiplicity = nn.Embedding(30, embeding_dim)

        self.mix_cm = nn.Sequential(
                                    nn.Linear(2*embeding_dim,embeding_dim),
                                    nn.SiLU()
                                       )
        self.unifiedtokentoembedding = nn.Linear(
            embeding_dim,embeding_dim
        )

        self.uniform_center_count = 5
        self.sph_grid_channel = 8

        if kwargs["encoder"] != "default":
            self.embed_proj = torch.nn.Sequential(
                nn.Linear(
                    kwargs["encoder_embed_dim"],
                    self.scalar_dim,
                ),
                nn.SiLU(),
                nn.LayerNorm(self.scalar_dim, eps=1e-6),
            )

        self.decoder = E2former(**kwargs)
        use_compile = bool(kwargs.get("use_compile", False) or kwargs.get("compile", False))
        if use_compile:
            try:
                from torch import _dynamo as torch_dynamo
                torch_dynamo.config.suppress_errors = True
            except Exception:
                pass
            try:
                from torch._inductor import config as inductor_config
                if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraphs"):
                    inductor_config.triton.cudagraphs = False
                if hasattr(inductor_config, "cuda_graphs"):
                    inductor_config.cuda_graphs = False
            except Exception:
                pass
            compile_options = {"triton.cudagraphs": False}

            def compile_module(module):
                try:
                    return torch.compile(module, options=compile_options)
                except TypeError:
                    return torch.compile(module, mode="max-autotune")

            try:
                self.decoder = compile_module(self.decoder)
                logger.info("Enabled torch.compile for E2former decoder")
            except Exception as exc:
                logger.warning(f"torch.compile failed for E2former decoder, falling back to eager: {exc}")

        self.max_neighbors = self.decoder.max_neighbors
        self.max_radius = self.decoder.max_radius
        self.with_cluster = self.decoder.with_cluster

        self.long_cutoff_lower = self.decoder.long_cutoff_lower
        self.long_cutoff_upper = self.decoder.long_cutoff_upper
        # Configure logging and compilation
        torch._logging.set_logs(recompiles=True)
        self.construct_neighbor = construct_radius_neighbor

    def BOO_feature(self, pos, expand_pos, local_attention_weight):
        B, N1 = pos.shape[:2]
        expand_pos.shape[1]
        dist = torch.norm(pos.unsqueeze(dim=2) - expand_pos.unsqueeze(dim=1), dim=-1)
        edge_vec = (pos.unsqueeze(dim=2) - expand_pos.unsqueeze(dim=1)) / (
            dist.unsqueeze(dim=-1) + 1e-5
        )
        angel = torch.sum(
            edge_vec * (local_attention_weight.unsqueeze(dim=-1) > 1e-6), dim=2
        )

        angel = torch.sum(angel**2, dim=-1) - torch.sum(
            local_attention_weight > 1e-6, dim=2
        ).unsqueeze(dim=-1)
        return angel

    def forward(
        self,
        batched_data:torch_geometric.data.Batch,
        **kwargs,
    ):
        """
        Forward pass implementation that can be compiled with torch.compile.
        """
        use_grad = True 
        batched_data["pos"].requires_grad_(use_grad)
        # Generate embeddings
        atomic_numbers = batched_data["atomic_numbers"]
        padding_mask = ~batched_data["atom_masks"]
        node_mask = batched_data["atom_masks"]
        batched_data["pos"] = torch.where(
            padding_mask.unsqueeze(dim=-1).repeat(1, 1, 3),
            999.0,
            batched_data["pos"].float(),
        )
        pos = batched_data["pos"]
        B, L = batched_data["pos"].shape[:2]
        device = pos.device

        # Handle periodic boundary conditions

        if "expand_node_pos" not in batched_data:
            if (
                "pbc" in batched_data
                and batched_data["pbc"] is not None
                and torch.any(batched_data["pbc"])
            ):
                pbc_expand_batched = self.cell_expander.expand_includeself(
                    pos,
                    batched_data["pbc"],
                    atomic_numbers,
                    batched_data["cell"],
                    neighbors_radius=(
                        self.kwargs["max_neighbors"],
                        self.kwargs["pbc_max_radius"],
                    ),
                    use_grad=use_grad,
                    padding_mask=padding_mask,
                )
                # dist: B*tgt_len*src_len
                
                batched_data.update(pbc_expand_batched)
            else:

                batched_data.update({"expand_node_pos": pos,
                                    "outcell_index": torch.arange(L).unsqueeze(dim=0).repeat(B, 1).to(device),
                                    "expand_node_mask":node_mask})
        expand_node_pos = batched_data["expand_node_pos"]
        expand_node_mask = batched_data["expand_node_mask"]
        mask = ~expand_node_mask
        if mask.dim() == expand_node_pos.dim() - 1:
            mask = mask.unsqueeze(-1)
        batched_data["expand_node_pos"] = expand_node_pos.masked_fill(mask, 999)
        ptr = torch.nn.functional.pad(node_mask.sum(-1).to(torch.int32).cumsum(0), (1, 0))
        batched_data.update({"f_exp_node_pos":batched_data["expand_node_pos"][batched_data["expand_node_mask"]],
            "f_outcell_index": (batched_data["outcell_index"] + ptr[:B, None])[batched_data["expand_node_mask"]]})
        neighbor_info = self.construct_neighbor(pos,node_mask,
                    batched_data["expand_node_pos"],batched_data["expand_node_mask"],
                        max_dist = self.max_radius,
                        min_dist = 1e-4,
                        max_neighbors = self.max_neighbors)
        


        if self.with_cluster.startswith('node'):
            exp_atomic_numbers = torch.gather(atomic_numbers,dim = 1,index = batched_data["outcell_index"])
            cluster_neighbor_info = self.construct_neighbor(pos,node_mask,
                                batched_data["expand_node_pos"],batched_data["expand_node_mask"],
                                include_mask = ~torch.isin(exp_atomic_numbers,
                                                            torch.Tensor([0,1]).cuda()),
                                max_dist = self.long_cutoff_upper,
                                min_dist = self.long_cutoff_lower,
                                max_neighbors = int(2.5*self.max_neighbors),
                                poly="poly_bell" if "bell" in self.with_cluster else "poly")

        elif self.with_cluster!="False":
            cluster_pos,cluster_mask,flat_atom_clusterid = construct_cluster(
                                                batched_data["expand_node_pos"],
                                                batched_data["expand_node_mask"])
            f_cluster_pos = cluster_pos[cluster_mask]
            cluster_neighbor_info = self.construct_neighbor(pos,node_mask,
                                cluster_pos,cluster_mask,
                                max_dist = self.long_cutoff_upper,
                                min_dist = self.long_cutoff_lower,
                                max_neighbors = self.max_neighbors,
                                poly="poly_bell")


            cluster_atomcnt = torch.bincount(flat_atom_clusterid.reshape(-1), minlength=torch.max(flat_atom_clusterid).long().item()) 
            cnt_poly = scaled_sigmoid(cluster_atomcnt.float(),15,k=1)
            cluster_neighbor_info["f_poly_dist"]=cluster_neighbor_info["f_poly_dist"]*cnt_poly[cluster_neighbor_info["f_sparse_idx_expnode"]]

            f_cluster_pos = cluster_pos[cluster_mask]
            f_N2 = f_cluster_pos.shape[0]
            if self.with_cluster == "mixcluster":
                f_poly_dist= torch.cat([cluster_neighbor_info["f_poly_dist"],
                                        neighbor_info["f_poly_dist"]],dim = 1)
                f_attn_mask = torch.cat([cluster_neighbor_info["f_attn_mask"],
                                        neighbor_info["f_attn_mask"]],dim = 1)
                cluster_neighbor_info.update(
                    {   
                        "flat_atom_clusterid":flat_atom_clusterid,
                        "f_cluster_pos": f_cluster_pos,
                        "f_poly_dist":f_poly_dist,
                        "f_attn_mask":f_attn_mask,
                        "f_dist_node":neighbor_info["f_dist"],
                        "f_dist_cluster":cluster_neighbor_info["f_dist"],
                        "f_sparse_idx_expnode":torch.cat([cluster_neighbor_info["f_sparse_idx_expnode"],
                                                neighbor_info["f_sparse_idx_expnode"]+f_N2],dim = 1)
                    }
                )
            else:
                cluster_neighbor_info.update(
                    {   
                        "flat_atom_clusterid":flat_atom_clusterid,
                        "f_cluster_pos": f_cluster_pos,
                        })
        else:
            cluster_neighbor_info = None

        token_embedding = self.embedding(atomic_numbers)
        if "charge" not in batched_data:
            sys_embedding = None #torch.zeros_like(token_embedding)
        else:
            sys_embedding = self.mix_cm(
                                    torch.cat([
                                    self.embedding_charge(torch.clip(batched_data["charge"],-10,10)+10),
                                    self.embedding_multiplicity(torch.clip(batched_data["multiplicity"],0,20))],dim = -1)
                                    ).reshape(B,-1)
            token_embedding = token_embedding+sys_embedding.unsqueeze(dim=1)
            token_embedding = self.unifiedtokentoembedding(token_embedding)
        # Forward through decoder
        (
            node_features,
            node_vec_features,
            node_irreps,
        ) = self.decoder(
            batched_data,
            token_embedding,
            sys_embedding,
            neighbor_info=neighbor_info,
            cluster_neighbor_info = cluster_neighbor_info,
            padding_mask = padding_mask,
        )


                
        return {
            "node_featuresBxN": node_features,
            "node_vec_featuresBxN": node_vec_features,
            "node_irrepsBxN": node_irreps,
        }

    def flatten_node_features(
        self,
        node_features,
        node_vec_features,
        node_irreps,
        padding_mask,
    ):
        flat_node_irreps = node_irreps.view(
            -1, node_irreps.size(-2), node_irreps.size(-1)
        )
        flat_node_features = node_features.view(-1, node_features.size(-1))  # [B*N, D]
        flat_node_vec_features = node_vec_features.view(
            -1, node_vec_features.size(-2), node_vec_features.size(-1)
        )  # [B*N, D_vec]
        flat_mask = padding_mask.view(-1)  # [B*N]
        # Use the mask to filter out padded nodes
        valid_node_irreps = flat_node_irreps[flat_mask]  # [sum(valid_nodes), D]
        valid_node_features = flat_node_features[flat_mask]  # [sum(valid_nodes), D]
        valid_node_vec_features = flat_node_vec_features[flat_mask]
        return (
            valid_node_features,
            valid_node_vec_features,
            valid_node_irreps,
        )



    def test_equivariant(self, original_data):
        # assume batch size is 1
        assert (
            original_data.batch.max() == 0
        ), "batch size must be 1 for test_equivariant"
        self.eval()  # this is very important
        data_2 = original_data.clone().cpu()
        transform = RandomRotate([-180, 180], [0, 1, 2])
        data_2, matrix, inv_matrix = transform(data_2)
        data_2 = data_2.to(original_data.pos.device)
        data_list = [original_data, data_2]
        data_list.ptr = torch.tensor(
            [
                0,
                original_data.pos.size(0),
                original_data.pos.size(0) + data_2.pos.size(0),
            ],
            device=original_data.pos.device,
        )
        results = self.forward(data_list)
        combined_node_features = results["node_features"]
        # split the node features into two parts
        node_features_1 = combined_node_features[: original_data.pos.size(0)]
        node_features_2 = combined_node_features[original_data.pos.size(0) :]

        assert node_features_1.allclose(
            node_features_2, rtol=1e-2, atol=1e-2
        ), "node features are not equivariant"

        node_vec_features_1 = results["node_vec_features"][: original_data.pos.size(0)]
        node_vec_features_2 = results["node_vec_features"][original_data.pos.size(0) :]
        # rotate the node vec features
        node_vec_features_1 = torch.einsum(
            "bsd, sj -> bjd", node_vec_features_1, matrix.to(node_vec_features_1.device)
        )
        assert node_vec_features_1.allclose(
            node_vec_features_2, rtol=1e-2, atol=1e-2
        ), "node vec features are not equivariant"
