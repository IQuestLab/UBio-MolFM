# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch_geometric
from e3nn import o3


from .e2former import E2former
from .module_utils import CellExpander,scaled_sigmoid,construct_radius_neighbor
from .sp.dist_ner import FaissSearch, distributed_cell_expand


_AVG_NUM_NODES = 77.81317



def _mark_neighbor_dynamic(neighbor_info: dict) -> None:
    """Mark edge-list like tensors as dynamic on dim0 for torch.compile stability."""
    try:
        from torch import _dynamo as torch_dynamo
    except Exception:
        return

    for key in ("query_idx", "neighbor_idx", "f_edge_vec", "f_dist", "f_poly_dist"):
        tensor = neighbor_info.get(key, None)
        if not isinstance(tensor, torch.Tensor) or tensor.dim() == 0:
            continue
        try:
            torch_dynamo.mark_dynamic(tensor, 0)
        except Exception:
            # Marking can fail for some tensor subclasses; skip safely.
            pass


class E2FormerBackbone(nn.Module):
    """
    Physics Science Module backbone model integrated with EScAIP framework.

    unique architectural features.
    """
    def __init__(
        self,
        use_faiss: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        self.use_faiss = use_faiss

        # Cell expansion for periodic boundary conditions
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
        # logger.info("master config: ", kwargs)

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

        # Decoder selection and initialization
        if kwargs.get("max_neighbors",None) == "None":
            kwargs["max_neighbors"] = None

        self.decoder = E2former(**kwargs)

        self.max_neighbors = self.decoder.max_neighbors
        self.max_radius = self.decoder.max_radius
        self.long_cutoff_upper = self.max_radius
        self.with_cluster = self.decoder.with_cluster
        if "node" in self.with_cluster:
            self.long_cutoff_lower = self.decoder.long_cutoff_lower
            self.long_cutoff_upper = self.decoder.long_cutoff_upper
        else:
            self.long_cutoff_lower = 0

        # Configure logging and compilation
        torch._logging.set_logs(recompiles=True)

        self.construct_neighbor = construct_radius_neighbor
        if self.use_faiss:
            self._faiss_neighbor_constructor = FaissSearch(rank=0)
            self.construct_neighbor = self._faiss_neighbor_constructor.construct_radius_neighbor

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
        # Enable gradient computation for forces if needed
        use_grad = True
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
                if self.use_faiss:
                    if batched_data["pos"].shape[0] != 1:
                        raise ValueError("Faiss neighbor mode only supports batch size 1")
                    cell_bound_min = torch.min(batched_data["pos"].reshape(-1, 3), dim=0)[0]
                    cell_bound_max = torch.max(batched_data["pos"].reshape(-1, 3), dim=0)[0]
                    pbc_expand_batched = distributed_cell_expand(
                        batched_data["pos"],
                        batched_data["cell"][0],
                        self.long_cutoff_upper,
                        cell_bound_min,
                        cell_bound_max,
                        self.kwargs.get("pbc_expanded_num_cell_per_direction", 1),
                        batched_data["pbc"],
                    )
                else:
                    pbc_expand_batched = self.cell_expander.expand_includeself(
                        pos,
                        batched_data["pbc"],
                        atomic_numbers,
                        batched_data["cell"],
                        neighbors_radius=(None, self.long_cutoff_upper),
                        use_grad=use_grad,
                        padding_mask=padding_mask,
                    )
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

        if "node" in self.with_cluster:
            exp_atomic_numbers = torch.gather(
                atomic_numbers, dim=1, index=batched_data["outcell_index"]
            )
            include_mask = ~torch.isin(
                exp_atomic_numbers,
                torch.tensor([0, 1], device=exp_atomic_numbers.device),
            )
            cluster_neighbor_info = self.construct_neighbor(
                pos,
                node_mask,
                batched_data["expand_node_pos"],
                batched_data["expand_node_mask"],
                include_mask=include_mask,
                max_dist=self.long_cutoff_upper,
                min_dist=self.long_cutoff_lower,
                max_neighbors=None if self.max_neighbors is None else int(2.5 * self.max_neighbors),
                poly="poly_bell" if "bell" in self.with_cluster else "poly",
            )
        else:
            cluster_neighbor_info = None

        _mark_neighbor_dynamic(neighbor_info)
        if cluster_neighbor_info is not None:
            _mark_neighbor_dynamic(cluster_neighbor_info)

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
            cluster_neighbor_info=cluster_neighbor_info,
            padding_mask = padding_mask,
        )

        # # flatten the node features from num batchs times num nodes to num nodes (to pyG style ), note that nodes are padded
        # (
        #     node_features_flatten,
        #     node_vec_features_flatten,
        #     node_irreps_flatten,
        # ) = self.flatten_node_features(
        #     node_features,
        #     node_vec_features,
        #     node_irreps,
        #     ~padding_mask,
        # )
        # #  #[batched_data["node_mask"]]
        # # f_force_weight = torch.sum(
        # #         ~torch.isin(atomic_numbers[node_mask][neighbor_info["f_sparse_idx_expnode"]],torch.Tensor([1,6,7,8]).cuda()),dim = 1)
        
        # pred_energy = 0.1*torch.sum(1/neighbor_info["f_dist"][~neighbor_info["f_attn_mask"].squeeze()]**2).reshape(1)
        # grad_outputs = [torch.ones_like(pred_energy)]

        # pred_forces = -torch.autograd.grad(
        #     outputs=pred_energy,
        #     inputs=batched_data["pos"],
        #     grad_outputs=grad_outputs,
        #     create_graph=self.training,
        #     retain_graph=self.training,
        #     only_inputs=True,
        # )[0]

        # pred_energy = 0.002*lj_switching(
        #     batched_data["pos"],
        #     epsilon=0.15,
        #     sigma=1,
        #     r_switch=8.0,
        #     r_cut=10.0,
        #     softcore=0.25,
        #         ).reshape(-1)
        # # batched_data["pos"].requires_grad_(True)
        # # dist = torch.sum(
        # #         (batched_data["pos"].reshape(1,-1,3)-batched_data["pos"].reshape(-1,1,3))**2,dim = -1)
            
        # # pred_energy = 0.001*torch.sum(1/dist[dist>1e-3]).reshape(1)

        # grad_outputs = [torch.ones_like(pred_energy)]

        # pred_forces = -torch.autograd.grad(
        #     outputs=pred_energy,
        #     inputs=batched_data["pos"],
        #     grad_outputs=grad_outputs,
        #     create_graph=False,
        #     retain_graph=False,
        #     only_inputs=True,
        # )[0]
        return {
            # "dist":dist,
            # "pred_energy":pred_energy,
            #     "pred_forces":pred_forces,

            "node_featuresBxN": node_features,
            "node_vec_featuresBxN": node_vec_features,
            "node_irrepsBxN": node_irreps,
            "data": batched_data,
            # "node_irreps": node_irreps_flatten,
            # "node_features": node_features_flatten,
            # "node_vec_features": node_vec_features_flatten,
            # "f_force_weight":f_force_weight,
            "neighbor_info":neighbor_info
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




def lj_switching(
    pos,                    # [1,N,3]
    epsilon=0.2,
    sigma=3.4,
    r_switch=8.0,
    r_cut=10.0,
    softcore=0.0,
):
    assert r_switch < r_cut

    diff = pos.unsqueeze(2) - pos.unsqueeze(1)   # [1,N,N,3]
    r2 = (diff**2).sum(-1)                       # [1,N,N]
    N = r2.size(-1)

    # remove diagonal
    eye = torch.eye(N, device=r2.device, dtype=r2.dtype).unsqueeze(0)
    r2 = r2 + eye * 1e12

    # optional softcore
    if softcore > 0:
        r2 = r2 + softcore**2

    r = torch.sqrt(r2)

    # -------- LJ core: 4ε[(σ/r)^12 - (σ/r)^6] --------
    inv_r = 1.0 / r
    sr = sigma * inv_r
    sr6 = sr**6
    sr12 = sr6**2
    V_lj = 4.0 * epsilon * (sr12 - sr6)

    # -------- C^2 switching S(r) --------
    S = torch.ones_like(r)

    # switch region
    mask_sw = (r >= r_switch) & (r <= r_cut)
    x = (r - r_switch) / (r_cut - r_switch)
    x = torch.clamp(x, 0.0, 1.0)
    S_sw = 1 - 10*x**3 + 15*x**4 - 6*x**5
    S = torch.where(mask_sw, S_sw, S)

    # beyond cutoff
    S = torch.where(r > r_cut, torch.zeros_like(S), S)

    V = S * V_lj

    # i<j only
    tri = torch.triu(torch.ones_like(V), diagonal=1)
    E = (tri * V).sum()

    return E
