# -*- coding: utf-8 -*-
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn as nn
import torch_geometric
from e3nn import o3
from loguru import logger

from torch_geometric.data import Data


from .e2former import E2former
from .e2former_mixed_resolution import E2formerMR
from .e2former_cluster import construct_cluster
from .module_utils import CellExpander,scaled_sigmoid,construct_radius_neighbor
from .mr_routing import resolve_mr_is_light
from .sp.dist_ner import FaissSearch,distributed_cell_expand


from .utils.graph_utils import RandomRotate


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


def _mark_dynamic_dim0(t: torch.Tensor) -> None:
    """Mark dim0 of a tensor as dynamic, ignoring any errors."""
    if not isinstance(t, torch.Tensor) or t.dim() == 0:
        return
    try:
        torch._dynamo.mark_dynamic(t, 0)
    except Exception:
        pass


def _mark_mr_subgraph_dynamic(d: dict) -> None:
    """Mark all variable-size dim0 tensors inside a mixed-resolution sub-graph dict.

    Covers:
    - Standard edge keys shared with the base neighbor graph
    - MR-specific keys for cross-attention (HL / LH) edges
    - The 6-tensor ``metadata`` / ``metadata_l2h`` / ``metadata_h2l`` CSR
      tuples returned by ``prepare_sparse_qk_edge_index_metadata``
    - Wigner rotation matrices and expanded-node position arrays that are
      sliced per atom type and therefore have atom-count-dependent dim0
    - ``f_outcell_index`` (expanded node → query-local index)
    """
    _EDGE_KEYS = (
        # base neighbor keys
        "query_idx", "neighbor_idx", "f_edge_vec", "f_dist", "f_poly_dist",
        "f_outcell_index",
        # per-subgraph Wigner / position arrays
        "wigner", "wigner_inv", "wigner_exp", "wigner_inv_exp",
        "f_exp_node_pos",
        # HL cross keys
        "cross_edge_vec", "cross_dist", "cross_rbf", "cross_poly_dist",
        "cross_sph",
        "query_idx_l2h", "neighbor_idx_l2h",
        "light_local_idx", "light_atomic_numbers_exp",
        # LH cross keys
        "query_idx_h2l", "neighbor_idx_h2l",
        "heavy_local_idx", "heavy_atomic_numbers_exp",
        "cross_edge_vec_h2l", "cross_rbf_h2l", "cross_poly_dist_h2l",
        # atomic number arrays inside sub-graph dicts
        "heavy_atomic_numbers", "light_atomic_numbers",
    )

    for key in _EDGE_KEYS:
        t = d.get(key)
        if isinstance(t, torch.Tensor):
            _mark_dynamic_dim0(t)

    # metadata tuples: (q_row_ptr, q_col_index, q_alpha_index,
    #                   key_row_ptr, key_col_index, key_alpha_index)
    for meta_key in ("metadata", "metadata_l2h", "metadata_h2l"):
        meta = d.get(meta_key)
        if isinstance(meta, tuple):
            for t in meta:
                _mark_dynamic_dim0(t)

    # Recursively handle any nested batched_data sub-dict (cross_batched).
    inner = d.get("cross_batched")
    if isinstance(inner, dict):
        for key in ("wigner", "wigner_inv"):
            t = inner.get(key)
            if isinstance(t, torch.Tensor):
                _mark_dynamic_dim0(t)


def _mark_mr_atom_tensors_dynamic(*tensors: torch.Tensor) -> None:
    """Mark a sequence of per-atom tensors (sliced by type) as dynamic on dim0."""
    for t in tensors:
        _mark_dynamic_dim0(t)


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
        self._faiss_neighbor_constructor = None

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
        # kwargs["long_cutoff_lower"] = 0
        # kwargs["long_cutoff_upper"] = 8
        if kwargs.get("max_neighbors",None) == "None":
            kwargs["max_neighbors"] = None

        # Dispatch to the mixed-resolution E2former variant when enabled.
        # Keeps main's default E2former path untouched for non-MR runs.
        _mr_cfg = kwargs.get("mixed_resolution", None)
        if _mr_cfg and _mr_cfg.get("enabled", False):
            self.decoder = E2formerMR(**kwargs)
            self._mr_enabled = True
            self.register_buffer(
                "_lf_atoms_t",
                torch.tensor(list(_mr_cfg.get("lf_atoms", [1])), dtype=torch.long),
                persistent=False,
            )
        else:
            self.decoder = E2former(**kwargs)
            self._mr_enabled = False

        self.max_neighbors = self.decoder.max_neighbors
        self.max_radius = self.decoder.max_radius
        # Mixed-resolution cross-stream cutoff. Every shipping mode shares the
        # self cutoff, so the neighbor search radius is simply max_radius.
        self.r_cross = float(getattr(self.decoder, "r_cross", self.max_radius))
        self._mr_search_radius = max(float(self.max_radius), self.r_cross)
        self.with_cluster = self.decoder.with_cluster
        if "node" in self.with_cluster:
            self.long_cutoff_lower = self.decoder.long_cutoff_lower
            self.long_cutoff_upper = self.decoder.long_cutoff_upper
        else:
            self.long_cutoff_lower = 0
            self.long_cutoff_upper = self._mr_search_radius
        

        # Configure logging and compilation
        torch._logging.set_logs(recompiles=True)

        # Mixed-resolution shares modules (e.g. Linear biases) between heavy/light
        # paths with different hidden dimensions. Disable forced parameter-static
        # shapes so torch.compile does not recompile when the shared param moves
        # between the two paths.
        if _mr_cfg and _mr_cfg.get("enabled", False):
            try:
                torch._dynamo.config.force_parameter_static_shapes = False
            except AttributeError:
                pass

        self.construct_neighbor = construct_radius_neighbor

    def _ensure_faiss_neighbor_constructor(self):
        if not self.use_faiss:
            return None
        if self._faiss_neighbor_constructor is None:
            if not torch.cuda.is_available():
                raise RuntimeError("use_faiss=True requires CUDA, but CUDA is not available.")
            device_idx = torch.cuda.current_device()
            self._faiss_neighbor_constructor = FaissSearch(rank=device_idx)
        return self._faiss_neighbor_constructor

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
        # print('before norm',angel[0],torch.sum(local_attention_weight>1e-6,dim =2)[0])
        # angel = 2*angel/(torch.sum(local_attention_weight>1e-6,dim =2).unsqueeze(dim = -1))
        # angel = angel
        angel = torch.sum(angel**2, dim=-1) - torch.sum(
            local_attention_weight > 1e-6, dim=2
        ).unsqueeze(dim=-1)
        # angel = self.boo_embedding(torch.sum(local_attention_weight>1e-6,dim =2))
        # print('after norm',angel[0],torch.max(angel),torch.min(angel))
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

        # Mixed-resolution: build the canonical [B, L] is_light mask once here
        # (atoms with Z in lf_atoms, optionally OR'd with the HOH-O mask when
        # lf_O_in_HOH=True is plumbed through the dataloader). E2formerMR and
        # head.py both consume batched_data["is_light"] when present.
        #
        # resolve_mr_is_light, NOT a local recomputation: a caller (for example
        # the ASE frontend's HOH-O mask) may already have supplied a canonical
        # mask that carries information the atomic numbers alone cannot recover,
        # such as an HOH oxygen identified from residue names. Overwriting it
        # with the atomic-number-only result would silently reroute those atoms
        # to the heavy stream. The helper returns the incoming mask when one is
        # present and only recomputes when it is absent.
        if self._mr_enabled:
            batched_data["is_light"] = resolve_mr_is_light(
                batched_data, self._lf_atoms_t, where="E2FormerBackbone.forward",
            )
        batched_data["pos"] = torch.where(
            padding_mask.unsqueeze(dim=-1).repeat(1, 1, 3),
            999.0,
            batched_data["pos"].float(),
        )
        pos = batched_data["pos"]
        B, L = batched_data["pos"].shape[:2]
        device = pos.device

        # Handle periodic boundary conditions

        neighbor_constructor = self.construct_neighbor
        if self.use_faiss:
            neighbor_constructor = self._ensure_faiss_neighbor_constructor().construct_radius_neighbor
        if "expand_node_pos" not in batched_data:
            if (
                "pbc" in batched_data
                and batched_data["pbc"] is not None
                and torch.any(batched_data["pbc"])
            ):
                if self.use_faiss:
                    if batched_data["pos"].shape[0] != 1:
                        raise ValueError("Faiss neighbor mode only supports batch size 1")

                    cell_boundMIN = torch.min(batched_data["pos"].reshape(-1, 3), dim=0)[0]
                    cell_boundMAX = torch.max(batched_data["pos"].reshape(-1, 3), dim=0)[0]
                    pbc_expand = distributed_cell_expand(
                        batched_data['pos'],
                        batched_data["cell"][0],
                        self.long_cutoff_upper,
                        cell_boundMIN,
                        cell_boundMAX,
                        self.kwargs.get("pbc_expanded_num_cell_per_direction", 1),
                        batched_data["pbc"])
                    batched_data.update(pbc_expand)
                else:
                    pbc_expand_batched = self.cell_expander.expand_includeself(
                        pos,
                        batched_data["pbc"],
                        atomic_numbers,
                        batched_data["cell"],
                        neighbors_radius=(
                            None,
                            self.long_cutoff_upper,
                        ),
                        use_grad=use_grad,
                        padding_mask=padding_mask,
                    )
                    
                    batched_data.update(pbc_expand_batched)
            else:

                batched_data.update({"expand_node_pos": pos,
                                    "outcell_index": torch.arange(L).unsqueeze(dim=0).repeat(B, 1).to(device),
                                    "expand_node_mask":node_mask})
        # Avoid in-place op on a leaf tensor when grad is enabled.
        expand_node_pos = batched_data["expand_node_pos"]
        expand_node_mask = batched_data["expand_node_mask"]
        mask = ~expand_node_mask
        if mask.dim() == expand_node_pos.dim() - 1:
            mask = mask.unsqueeze(-1)
        batched_data["expand_node_pos"] = expand_node_pos.masked_fill(mask, 999)
        ptr = torch.nn.functional.pad(node_mask.sum(-1).to(torch.int32).cumsum(0), (1, 0))
        batched_data.update({"f_exp_node_pos":batched_data["expand_node_pos"][batched_data["expand_node_mask"]],
            "f_outcell_index": (batched_data["outcell_index"] + ptr[:B, None])[batched_data["expand_node_mask"]]})
        # print("pre f_outcell_index:",batched_data["f_outcell_index"].shape)
        neighbor_info = neighbor_constructor(pos,node_mask,
                    batched_data["expand_node_pos"],batched_data["expand_node_mask"],
                        max_dist = self._mr_search_radius,
                        min_dist = 1e-4,
                        max_neighbors = self.max_neighbors)
        
        if "node" in self.with_cluster:
            exp_atomic_numbers = torch.gather(atomic_numbers,dim = 1,index = batched_data["outcell_index"])
            cluster_neighbor_info = neighbor_constructor(pos,node_mask,
                                batched_data["expand_node_pos"],batched_data["expand_node_mask"],
                                include_mask = ~torch.isin(exp_atomic_numbers,
                                                            torch.tensor([0, 1], device=device)),
                                max_dist = self.long_cutoff_upper,
                                min_dist = self.long_cutoff_lower,
                                max_neighbors = None if self.max_neighbors is None else int(2.5*self.max_neighbors),
                                poly="poly_bell" if "bell" in self.with_cluster else "poly")
            
        else:
            cluster_neighbor_info = None
        
        _mark_neighbor_dynamic(neighbor_info)
        if cluster_neighbor_info is not None:
            _mark_neighbor_dynamic(cluster_neighbor_info)

        

            

        token_embedding = self.embedding(atomic_numbers)
        # print(batched_data["charge"],batched_data["multiplicity"])
        # batched_data["multiplicity"] *= 0
        # print(batched_data["pbc"],batched_data["pos"].shape)
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
