# -*- coding: utf-8 -*-
import os
import math
import warnings
from typing import Dict, Optional
from functools import partial

from .mr_routing import resolve_mr_is_light

import e3nn
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
    
from torch import logical_not, nn
from torch.profiler import record_function
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
# from torch.nn.attention import SDPBackend, sdpa_kernel
# from .moe import MoE
# from fairchem.core.models.escn.so3 import SO3_Grid


# from torch.nn.attention import SDPBackend, sdpa_kernel

from loguru import logger
from .module_utils import init_edge_rot_euler_angles,eulers_to_wigner #
from .e2former_cluster import E2formerMixCluster,E2formerCluster
from .triton_dr.triton_sparse_qk_autograd import (
    prepare_sparse_qk_edge_index_metadata,
    sparse_qk_edge_index_triton,
    sparse_qk_triton_n_tiled as triton_sparse_qk,
)
from .triton_dr.sparse_v_agg_lastdim_autograd import (
    sparse_v_agg_edge_index_triton,
    sparse_v_agg_triton_n_tiled as sparse_v_agg_lastdim,
)
from .triton_dr.edge_degree_outer_autograd import edge_degree_outer_triton

# # QM9
# _MAX_ATOM_TYPE = 20
# # Statistics of QM9 with cutoff radius = 5
# _AVG_NUM_NODES = 18.03065905448718
from .module_utils import (  # ,\; EquivariantInstanceNorm,EquivariantGraphNorm; EquivariantRMSNormArraySphericalHarmonicsV2,; GaussianLayer,; irreps2gate,; sort_irreps_even_first,
    FeedForwardNetwork_s3,
    FeedForwardNetwork_swiglu_s2,
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


# Lazy import to avoid circular dependency: e2former_wrapper imports E2former,
# so we defer these imports to function-level below where they are first used.
def _get_mr_mark_dynamic_fns():
    from .e2former_wrapper import _mark_mr_subgraph_dynamic, _mark_mr_atom_tensors_dynamic  # noqa: E402
    return _mark_mr_subgraph_dynamic, _mark_mr_atom_tensors_dynamic


def _a2h_kv_union_indices(f_outcell_index_long, neighbor_idx_heavy, f_N1, n_exp):
    """Heavy-neighbour union of an All2Heavy edge pack.

    An All2Heavy TransBlock compresses the QUERY side to heavy atoms but projects
    K / V / source-embedding over the FULL node stream and gathers to the FULL
    PBC-expanded set. Heavy queries only ever read the KV rows named by
    `neighbor_idx`, so every other row is projected and thrown away -- on large
    systems this dominates the block's cost compared with the sparse attention
    itself.

    Returns a dict with
      U_pri      [|U_pri|] ascending primary atom ids that neighbour some heavy atom
      U_exp      [|U_exp|] ascending expanded row ids read by heavy-query edges
      f_outcell_index  [|U_exp|] primary rank of each U_exp row, in [0, |U_pri|)
      neighbor_idx     [E_h]     compressed row rank, in [0, |U_exp|)
      n_pri      0-d tensor |U_pri|, so the caller can batch the host read of the
                 union size with the heavy/light counts it already reads

    Restricting the KV projection to ``U_pri`` is EXACT, not an approximation:
    the all-side (key / value / src_node) is consumed only through
    ``neighbor_idx``; the block output is per-query (heavy) with no all-side
    pooling and no all-side self term; and ``norm_1``
    (EquivariantRMSNormArraySphericalHarmonicsV2_BL) reduces only over the SH and
    channel dims, never over nodes. PBC is fine too -- KV features are
    per-primary (image-invariant) and the image offset lives in the per-edge
    edge_vec / wigner.

    ``cumsum(0) - 1`` is used instead of the ``full(-1)`` + ``scatter(arange)``
    dense-inverse idiom because that needs ``torch.arange`` over an unbacked SymInt (a ``nonzero``
    output size), which dynamo cannot handle. INVARIANT: the ranks at ``False``
    positions are garbage (they repeat the previous True rank), but they are
    never read -- ``neighbor_idx_heavy`` is a subset of ``U_exp`` and
    ``outcell_U`` a subset of ``U_pri`` by construction.
    """
    device = neighbor_idx_heavy.device
    ni_h = neighbor_idx_heavy.to(torch.long)

    mask_exp = torch.zeros((n_exp,), dtype=torch.bool, device=device)
    mask_exp[ni_h] = True
    U_exp = mask_exp.nonzero(as_tuple=True)[0]                 # ascending
    exp_g2l = mask_exp.cumsum(0) - 1                           # int64

    outcell_U = f_outcell_index_long.index_select(0, U_exp)     # [|U_exp|] primary ids
    mask_pri = torch.zeros((f_N1,), dtype=torch.bool, device=device)
    mask_pri[outcell_U] = True
    U_pri = mask_pri.nonzero(as_tuple=True)[0]                 # ascending
    pri_g2l = mask_pri.cumsum(0) - 1

    return {
        "U_pri": U_pri,
        "U_exp": U_exp,
        # [|U_exp|] -> [0, |U_pri|): the compressed expanded->primary map the
        # block's three KV gathers (source_embedding / key_linear / proj_value)
        # index with. Kept int64, matching the aligned branch's existing dtypes.
        "f_outcell_index": pri_g2l.index_select(0, outcell_U),
        # [E_h] -> [0, |U_exp|)
        "neighbor_idx": exp_g2l.index_select(0, ni_h),
        "n_pri": mask_pri.sum(),
    }




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
        normalized_radial_profile: bool = True,
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
        self.use_softplus = "softplus" in tp_type
        self.use_sigmoid = "sigmoid" in tp_type
        self.node_embed_dim = 128
        self._use_layer_norm_radial = normalized_radial_profile

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
                                            [self.attn_weight_input_dim * (self.lmax+1)],
                                            use_layer_norm=self._use_layer_norm_radial)
            self.fc_m0 = nn.Linear(self.attn_weight_input_dim*(self.lmax+1),self.num_attn_heads)

            self.fc_easy = RadialProfile(
                self.edge_channel_list + [self.num_attn_heads],
                use_layer_norm=self._use_layer_norm_radial,
            )
        elif self.tp_type == "QKdotS_alpha":
            self.sum_vec = torch.zeros([(self.lmax+1)**2,self.lmax+1])
            for l in range(self.lmax+1):
                self.sum_vec[l**2:(l+1)**2,l] = 1
            self.sum_vec = torch.nn.Parameter(self.sum_vec,requires_grad=False)
            # non-persistent: derived from sum_vec, not saved to checkpoint
            s_idx = torch.tensor([int(i**0.5) for i in range((self.lmax+1)**2)], dtype=torch.long)
            self.register_buffer("sum_vec_idx", s_idx, persistent=False)
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
                self.edge_channel_list + [self.num_attn_heads + self.num_attn_heads], #+ self.num_attn_heads * (1+self.lmax)
                use_layer_norm=self._use_layer_norm_radial,
            )
        else:

            self.fc_easy = RadialProfile(
                self.edge_channel_list + [2*self.num_attn_heads],
                use_layer_norm=self._use_layer_norm_radial,
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
            # logger.info(f"Layer {layer_id}  legal l1l2l3{self.first_order_tp.so2_tp.legal_l1l2l3}" )
            # print(f"Layer {layer_id}  legal l1l2l3{self.first_order_tp.so2_tp.legal_l1l2l3}" )
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
        kv_source=None,
        kv_atomic_numbers=None,
        **kwargs,
    ):
        f_N1, _, hidden = node_irreps_input.shape
        # --- merged_pre_loop path B: separate query source and kv source ---
        # When kv_source is provided, key/value are projected from kv_source
        # (e.g. [N_H+N_L, Lh, C_H]) while query comes from node_irreps_input
        # (e.g. [N_H, Lh, C_H]).  kv_atomic_numbers provides the atomic
        # numbers aligned with kv_source for the source embedding lookup.
        _kv = kv_source if kv_source is not None else node_irreps_input
        _kv_an = kv_atomic_numbers if kv_atomic_numbers is not None else atomic_numbers
        value = _kv
        f_N2 = batched_data["f_outcell_index"].shape[0]
        query_idx = batched_data["query_idx"]
        neighbor_idx = batched_data["neighbor_idx"]
        metadata = batched_data["metadata"]
        query_idx_long = query_idx.to(torch.long)
        neighbor_idx_long = neighbor_idx.to(torch.long)
        num_edges = query_idx.shape[0]

        tgt_node = self.target_embedding(atomic_numbers)
        src_node = self.source_embedding(_kv_an)[batched_data["f_outcell_index"]]
        with record_function("query key linear function"):

            _q = self.query_linear(node_irreps_input)
            query = _q.reshape(
                f_N1, self.num_attn_heads, _q.shape[-1] // self.num_attn_heads
            )
            key = self.key_linear(_kv)[batched_data["f_outcell_index"]]
            key_lastdim = key.shape[-1]//self.num_attn_heads
        

        with record_function("rbf function-a"):
            x_edge = torch.cat(
                [
                    attn_weight,
                    tgt_node[query_idx_long],
                    src_node[neighbor_idx_long],
                ],
                dim=-1,
            )


        with record_function("qk attention"):
            if self.tp_type == "QKdot_alpha":
                x_0_extra = []
                node_irreps_input_dot = self.dot_linear(node_irreps_input)
                for l in range(self.lmax+1):
                    rij_l = e3nn.o3.spherical_harmonics(l, edge_vec, normalize=True).unsqueeze(dim=-1)
                    node_l = node_irreps_input_dot[query_idx_long, l**2:(l+1)**2]
                    x_0_extra.append(torch.sum(rij_l * node_l, dim=-2))
                edge_m0 = self.rad_func_m0(x_edge)
                gate = self.fc_m0(torch.cat(x_0_extra, dim=-1) * edge_m0)
                edge_gate  = self.fc_easy(x_edge)
                inputhead = edge_gate.reshape(num_edges, self.num_attn_heads)
            elif self.tp_type == "QKdotS_alpha":
                lxs = o3._spherical_harmonics._spherical_harmonics(
                    lmax=self.lmax,
                    x=edge_vec[...,0],
                    y=edge_vec[...,1],
                    z=edge_vec[...,2],
                )
                node_irreps_input_dot = self.dot_linear(node_irreps_input)
                # (e, l, c) * (e, l, 1) → (e, l, c)
                D = node_irreps_input_dot[query_idx_long] * lxs.unsqueeze(-1)
                # scatter l → s using one-hot index (avoids two dense GEMMs)
                _c = D.shape[-1]
                x_0_extra = torch.zeros(num_edges, self.lmax + 1, _c, device=D.device, dtype=D.dtype)
                x_0_extra.scatter_add_(
                    1,
                    self.sum_vec_idx[None, :, None].expand(num_edges, -1, _c),
                    D,
                )
                # Explicit flattened size, NOT -1: on a pure-water box with
                # lf_O_in_HOH=true every atom is light, so f_N_heavy == 0 and the
                # All2Heavy blocks still run, just with num_edges == 0. reshape(0, -1)
                # is ambiguous to PyTorch (0 * anything == 0) and raises; the size is
                # known exactly from how x_0_extra was built two lines up. Identical
                # result for num_edges > 0. Only this tp_type branch was affected --
                # QK_alpha+softplus has no such reshape, which is why the r5 HOHO
                # checkpoints ran 1 ns of water while the r6 QKdotS_alpha one died at
                # step 0.
                x_0_extra = x_0_extra.reshape(num_edges, (self.lmax + 1) * _c)[
                    :, : self.num_attn_heads
                ]
                edge_gate  = self.fc_easy(x_edge)
                edge_m0 = edge_gate[:, self.num_attn_heads:]
                gate = self.direction_linear(x_0_extra*edge_m0)
                inputhead = edge_gate[:, :self.num_attn_heads].reshape(
                    num_edges, self.num_attn_heads
                )
            else:
                edge_gate  = self.fc_easy(x_edge)
                gate = edge_gate[:, :self.num_attn_heads]
                inputhead = edge_gate[:, self.num_attn_heads:].reshape(
                    num_edges, self.num_attn_heads
                )

            
            key = key.reshape(-1, self.num_attn_heads, key_lastdim)

            alpha = gate * sparse_qk_edge_index_triton(
                query / math.sqrt(query.shape[-1]),
                key,
                query_idx,
                neighbor_idx,
                metadata=metadata,
            )

        if self.use_softplus:
            alpha = F.softplus(alpha) * inputhead * poly_dist.unsqueeze(-1)
        elif self.use_sigmoid:
            alpha = torch.sigmoid(alpha) * inputhead * poly_dist.unsqueeze(-1)
        else:
            query_index_heads = query_idx_long.unsqueeze(-1).expand(-1, self.num_attn_heads)
            alpha_max = torch.full(
                (f_N1, self.num_attn_heads),
                -torch.inf,
                device=alpha.device,
                dtype=alpha.dtype,
            )
            alpha_max.scatter_reduce_(
                0,
                query_index_heads,
                alpha,
                reduce="amax",
                include_self=True,
            )
            alpha = alpha - alpha_max[query_idx_long]
            alpha = torch.exp(alpha) * poly_dist.unsqueeze(-1)
            alpha_sum = torch.zeros(
                (f_N1, self.num_attn_heads),
                device=alpha.device,
                dtype=alpha.dtype,
            )
            alpha_sum.scatter_add_(0, query_index_heads, alpha)
            alpha = alpha * inputhead / (alpha_sum[query_idx_long] + 1e-3)

        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)



        def edge_triton_kernel(value=None, alpha=None,
                               value_branch=None, alpha_branch=None, **_):
            # value/alpha and value_branch/alpha_branch are the SAME two arguments
            # under two names. The zero-order branch below calls this with
            # value=/alpha=; E2TensorProductSO2_FirstOrder.forward (the
            # "first-order" / "so2-first-order" branch) calls the callback it is
            # handed with value_branch=/alpha_branch=, which is the name the
            # non-MR e2former.py's edge_agg_kernel uses. Accepting only the first
            # pair made every first-order attn_type in the MIXED-RESOLUTION model
            # die with "missing 2 required positional arguments" at the first
            # forward -- unreachable until now because every shipped MR config is
            # zero-order, and load-bearing from here on: cross_mode=
            # "interleaved_flash" with flash_up_proj="linear" and
            # lmax_light < lmax REQUIRES an order-creating attention.
            # Contract is unchanged: [M, C, H] in, [N, C, H] out.
            value = value if value is not None else value_branch
            alpha = alpha if alpha is not None else alpha_branch
            agg = sparse_v_agg_edge_index_triton(
                value=value.permute(0, 2, 1).contiguous(),
                alpha=alpha,
                query_index=query_idx,
                value_index=neighbor_idx,
                num_queries=f_N1,
                metadata=metadata,
            )
            return agg.permute(0, 2, 1).contiguous()


        value = self.proj_value(value)[batched_data["f_outcell_index"]]
        if self.attn_type == "zero-order":
            with record_function("zero-order"):
                # Explicit middle dim, NOT -1: PyTorch cannot infer a -1 dim when
                # the tensor has 0 elements (0 * anything == 0, so the size is
                # ambiguous) and raises. proj_value is an
                # SO3_Linear_e2former(C -> C), so `value` is [f_N2, (lmax+1)^2, C]
                # and the flattened head dim is exactly (lmax+1)^2 * C / heads.
                # f_N2 is the KV row count; it is > 0 for the full-graph packs, but
                # the all2heavy KV-union pack (mixed_resolution.a2h_kv_union)
                # compresses it to the rows heavy queries actually read, which is 0
                # on a pure-water box (lf_O_in_HOH -> every atom light, no heavy
                # queries) or for a heavy atom with no neighbor inside the cutoff.
                # reshape(0, -1, heads) is ambiguous to PyTorch and raises.
                # Identical result for f_N2 > 0.
                value = value.reshape(
                    f_N2,
                    (self.lmax + 1) ** 2
                    * self.irreps_node_input[0][0]
                    // self.num_attn_heads,
                    self.num_attn_heads,
                )
                agg = edge_triton_kernel(value=value, alpha=alpha)
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
                        triton_kernel=edge_triton_kernel,
                        f_sparse_idx_expnode=None,
                        batched_data=batched_data,
                    )
                node_output = self.proj_first(node_output)


        return node_output, attn_weight



class E2AttentionArbOrder_sparse_mixneighbor(torch.nn.Module):
    """
    Mixed neighbor attention with separate linear paths for long/heavy vs local neighbors.
    This expects f_sparse_idx_expnode to index into a concatenated [long, local] node list.
    """

    def __init__(
        self,
        irreps_node_input="256x0e+256x1e+256x2e",
        attn_weight_input_dim: int = 32,
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+32x1e+32x2e",
        alpha_drop=0.1,
        tp_type="QK_alpha",
        attn_type="first-order",
        atom_type_cnt=256,
        norm_layer="identity",
        normalized_radial_profile: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.atom_type_cnt = atom_type_cnt
        self.irreps_node_input = (
            e3nn.o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.scalar_dim = self.irreps_node_input[0][0]
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        irreps_head = (
            e3nn.o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        )

        self.irreps_head = irreps_head
        self.irreps_node_output = self.irreps_node_input
        self.lmax = self.irreps_node_input[-1][1][0]
        self.attn_type = attn_type
        self.tp_type = tp_type.split("+")[0]
        self.use_triton = "triton" in tp_type
        self.use_softplus = "softplus" in tp_type
        self.use_sigmoid = "sigmoid" in tp_type
        self.node_embed_dim = 128
        self._use_layer_norm_radial = normalized_radial_profile

        self.source_embedding = nn.Embedding(self.atom_type_cnt, self.node_embed_dim)
        self.target_embedding = nn.Embedding(self.atom_type_cnt, self.node_embed_dim)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.alpha_act = SmoothLeakyReLU(0.2)
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
            self.rad_func_m0 = RadialProfile(
                self.edge_channel_list + [self.attn_weight_input_dim * (self.lmax + 1)],
                use_layer_norm=self._use_layer_norm_radial,
            )
            self.fc_m0 = nn.Linear(
                self.attn_weight_input_dim * (self.lmax + 1), self.num_attn_heads
            )
            self.fc_easy = RadialProfile(
                self.edge_channel_list + [self.num_attn_heads],
                use_layer_norm=self._use_layer_norm_radial,
            )
        elif self.tp_type == "QKdotS_alpha":
            self.sum_vec = torch.zeros([(self.lmax + 1) ** 2, self.lmax + 1])
            for l in range(self.lmax + 1):
                self.sum_vec[l**2 : (l + 1) ** 2, l] = 1
            self.sum_vec = torch.nn.Parameter(self.sum_vec, requires_grad=False)
            self.dot_linear = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                self.num_attn_heads // (self.lmax + 1) + 1,
                lmax=self.lmax,
            )
            self.direction_linear = nn.Sequential(
                nn.Linear(self.num_attn_heads, self.num_attn_heads),
                nn.LayerNorm(self.num_attn_heads),
                nn.SiLU(),
                nn.Linear(self.num_attn_heads, self.num_attn_heads),
            )
            self.fc_easy = RadialProfile(
                self.edge_channel_list + [self.num_attn_heads + self.num_attn_heads],
                use_layer_norm=self._use_layer_norm_radial,
            )
        else:
            self.fc_easy = RadialProfile(
                self.edge_channel_list + [2 * self.num_attn_heads],
                use_layer_norm=self._use_layer_norm_radial,
            )

        self.query_linear = SO3_Linear2Scalar_e2former(
            self.irreps_node_input[0][0],
            num_attn_heads * self.attn_scalar_head,
            lmax=self.lmax,
            hidden_features=hidden_features,
        )
        self.key_linear_long = SO3_Linear2Scalar_e2former(
            self.irreps_node_input[0][0],
            num_attn_heads * self.attn_scalar_head,
            lmax=self.lmax,
            hidden_features=hidden_features,
        )
        self.key_linear_local = SO3_Linear2Scalar_e2former(
            self.irreps_node_input[0][0],
            num_attn_heads * self.attn_scalar_head,
            lmax=self.lmax,
            hidden_features=hidden_features,
        )

        self.proj_value_long = SO3_Linear_e2former(
            self.irreps_node_input[0][0],
            self.irreps_node_input[0][0],
            lmax=self.lmax,
        )
        self.proj_value_local = SO3_Linear_e2former(
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
            self.first_order_tp = E2TensorProductSO2_FirstOrder(
                self.irreps_node_input,
                (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
                order=1,
                head=self.num_attn_heads,
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
        attn_weight,
        atomic_numbers,
        poly_dist=None,
        attn_mask=None,
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        if self.attn_type != "zero-order":
            raise ValueError(
                "E2AttentionArbOrder_sparse_mixneighbor only supports zero-order attention"
            )

        f_N1, _, _ = node_irreps_input.shape
        topK = attn_weight.shape[1]
        value = node_irreps_input

        tgt_node = self.target_embedding(atomic_numbers)
        src_node = self.source_embedding(atomic_numbers)[batched_data["f_outcell_index"]]
        f_sparse_idx_expnode = batched_data["f_sparse_idx_expnode"]

        with record_function("query key linear function"):
            _q = self.query_linear(node_irreps_input)
            query = _q.reshape(
                f_N1, self.num_attn_heads, _q.shape[-1] // self.num_attn_heads
            )
            key_long = self.key_linear_long(node_irreps_input)[
                batched_data["f_outcell_index"]
            ]
            key_local = self.key_linear_local(node_irreps_input)[
                batched_data["f_outcell_index"]
            ]
            key_lastdim = key_long.shape[-1] // self.num_attn_heads

        src_node = torch.cat([src_node, src_node], dim=0)
        key = torch.cat([key_long, key_local], dim=0)

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
                for l in range(self.lmax + 1):
                    rij_l = e3nn.o3.spherical_harmonics(
                        l, edge_vec, normalize=True
                    ).unsqueeze(dim=-1)
                    node_l = node_irreps_input_dot[:, l**2 : (l + 1) ** 2]
                    x_0_extra.append(torch.sum(rij_l * node_l.unsqueeze(dim=1), dim=-2))
                edge_m0 = self.rad_func_m0(x_edge)
                gate = self.fc_m0(torch.cat(x_0_extra, dim=-1) * edge_m0)
                edge_gate = self.fc_easy(x_edge)
                inputhead = edge_gate.reshape((f_N1, topK, self.num_attn_heads))
            elif self.tp_type == "QKdotS_alpha":
                lxs = o3._spherical_harmonics._spherical_harmonics(
                    lmax=self.lmax,
                    x=edge_vec[..., 0],
                    y=edge_vec[..., 1],
                    z=edge_vec[..., 2],
                )
                node_irreps_input_dot = self.dot_linear(node_irreps_input)
                x_0_extra = torch.einsum(
                    "nlc,nkl,ls->nksc", node_irreps_input_dot, lxs, self.sum_vec
                ).reshape(f_N1, topK, -1)[:, :, : self.num_attn_heads]
                edge_gate = self.fc_easy(x_edge)
                edge_m0 = edge_gate[:, :, self.num_attn_heads :]
                gate = self.direction_linear(x_0_extra * edge_m0)
                inputhead = edge_gate[..., : self.num_attn_heads].reshape(
                    (f_N1, topK, self.num_attn_heads)
                )
            else:
                edge_gate = self.fc_easy(x_edge)
                gate = edge_gate[..., : self.num_attn_heads]
                inputhead = edge_gate[..., self.num_attn_heads :].reshape(
                    (f_N1, topK, self.num_attn_heads)
                )

            key = key.reshape(-1, self.num_attn_heads, key_lastdim)

            if self.use_triton:
                gate = gate.contiguous()
                alpha = triton_sparse_qk(
                    query / math.sqrt(query.shape[-1]),
                    key,
                    f_sparse_idx_expnode,
                    gate,
                )
            else:
                key = key[f_sparse_idx_expnode]
                alpha = (
                    gate
                    * torch.sum(query.unsqueeze(dim=1) * key, dim=3)
                    / math.sqrt(query.shape[-1])
                )

        alpha = alpha.reshape(f_N1, topK, self.num_attn_heads)
        if self.use_softplus:
            alpha = F.softplus(alpha) * inputhead * poly_dist.unsqueeze(-1)
            alpha = alpha.masked_fill(attn_mask, 0)
        elif self.use_sigmoid:
            alpha = torch.sigmoid(alpha) * inputhead * poly_dist.unsqueeze(-1)
            alpha = alpha.masked_fill(attn_mask, 0)
        else:
            alpha = alpha.masked_fill(attn_mask, -1e9)
            alpha = alpha - alpha.max(dim=1, keepdim=True).values
            alpha = torch.exp(alpha) * poly_dist.unsqueeze(-1)
            alpha = alpha * inputhead / (torch.sum(alpha, dim=1, keepdim=True) + 1e-3)

        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

        triton_kernel = partial(sparse_v_agg_lastdim) if self.use_triton else None

        value_long = self.proj_value_long(value)[batched_data["f_outcell_index"]]
        value_local = self.proj_value_local(value)[batched_data["f_outcell_index"]]
        value = torch.cat([value_long, value_local], dim=0)

        with record_function("zero-order"):
            value = value.reshape(value.shape[0], -1, self.num_attn_heads)
            if self.use_triton:
                agg = triton_kernel(
                    value=value,
                    idx=batched_data["f_sparse_idx_expnode"].contiguous(),
                    alpha=alpha,
                )
            else:
                agg = torch.sum(
                    alpha.unsqueeze(dim=2) * value[batched_data["f_sparse_idx_expnode"]],
                    dim=1,
                )
            agg = agg.reshape(f_N1, (self.lmax + 1) ** 2, self.irreps_node_input[0][0])
            node_output = self.proj_zero(agg)

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
        ffn_grid_resolution: int = 18,
        ffn_hidden_multiplier: int = 1,
        layer_id = 0,
        mix_neighbor: bool = False,
        mixed_resolution: dict = None,
        normalized_radial_profile: bool = True,
    ):
        super().__init__()
        self.mixed_resolution = mixed_resolution
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
            func = E2AttentionArbOrder_sparse_mixneighbor if mix_neighbor else E2AttentionArbOrder_sparse
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
            normalized_radial_profile=normalized_radial_profile,
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
                grid_resolution=ffn_grid_resolution,
                use_grid_mlp=False,  # notice in eqv2, default is True
            )
        elif "s3" in ffn_type:
            _s3_hidden = self.irreps_node_input[0][0] * max(1, int(ffn_hidden_multiplier))
            self.ffn = FeedForwardNetwork_s3(
                self.irreps_node_input[0][0],
                _s3_hidden,
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
        elif "swiglu" in ffn_type:
            # BRANCH ORDER IS LOAD-BEARING. This branch is reachable only for strings
            # that contain neither "s2" nor "s3": ffn_type="swiglu_s2" is swallowed by
            # the ("s2" in ffn_type) branch above and therefore builds
            # FeedForwardNetwork_s2 -- which is what existing checkpoints contain
            # (their ffn.* keys are gating_linear/so3_linear_1/so3_linear_2, never
            # so3_linear_kv). Do NOT hoist this branch above it "to fix the name":
            # that silently repoints existing training configs at a different FFN
            # family and invalidates their checkpoints. Use ffn_type="swiglu" for the
            # real SwiGLU-S^2.
            self.ffn = FeedForwardNetwork_swiglu_s2(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                lmax=self.lmax,
                grid_resolution=ffn_grid_resolution,
            )
        else:
            # self.ffn is called unconditionally in forward(), so an unmatched
            # ffn_type used to crash much later with "NoneType is not callable".
            raise ValueError(
                f"unrecognised ffn_type {ffn_type!r}; expected a string containing one "
                "of 'default'/'s2' (FeedForwardNetwork_s2, and note 'swiglu_s2' lands "
                "here for backward compatibility), 's3' (FeedForwardNetwork_s3), "
                "'spectral' (SpectralAtomwise) or 'swiglu' "
                "(FeedForwardNetwork_swiglu_s2)."
            )

        self.use_recompute = False

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
        kv_source=None,
        kv_atomic_numbers=None,
        kv_sys_embedding=None,
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
        # Normalize kv_source through the same norm as the query when provided.
        if kv_source is not None:
            _kv_normed = self.norm_1(kv_source)
            # Mirror the query's sys-embedding add onto the kv. Needed when the
            # kv IS the full node stream (aligned all2heavy: kv == query source),
            # so a self-attention-equivalent kv carries the per-atom system
            # embedding on l=0 exactly as the kv_source==query path would. mpl
            # leaves this None -> kv stays sys-free (its trained behavior).
            if kv_sys_embedding is not None:
                _kv_normed[:, 0, :] = _kv_normed[:, 0, :] + kv_sys_embedding
        else:
            _kv_normed = None
        # node_irreps, attn_weight = checkpoint(self.ga,
        #     node_pos=node_pos,
        #     node_irreps_input=node_irreps,
        #     edge_dis=edge_dis,
        #     poly_dist=poly_dist,
        #     edge_vec=edge_vec,
        #     attn_weight=attn_weight,
        #     atomic_numbers=atomic_numbers,
        #     attn_mask=attn_mask,
        #     batched_data=batched_data,
        #     use_reentrant=False)
        if self.use_recompute:
            node_irreps, attn_weight = checkpoint(
                self.ga,
                use_reentrant=False,
                node_pos=node_pos,
                node_irreps_input=node_irreps,
                edge_dis=edge_dis,
                poly_dist=poly_dist,
                edge_vec=edge_vec,
                attn_weight=attn_weight,
                atomic_numbers=atomic_numbers,
                attn_mask=attn_mask,
                batched_data=batched_data,
                kv_source=_kv_normed,
                kv_atomic_numbers=kv_atomic_numbers,
            )
        else:
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
                kv_source=_kv_normed,
                kv_atomic_numbers=kv_atomic_numbers,
            )
        node_irreps = node_irreps + node_irreps_res

        # # Mixed Resolution Zero-Masking (Attention Layer)
        # if self.mixed_resolution and self.mixed_resolution.get("enabled", False):
        #     lf_atoms = self.mixed_resolution.get("lf_atoms", [1])
        #     is_lf = torch.isin(atomic_numbers, torch.tensor(lf_atoms, device=atomic_numbers.device))
        #     if is_lf.any():
        #         node_irreps[is_lf, 1:, :] = 0.0

        with record_function("FeedForwardFunction"):
            ## residual connection
            node_irreps_res = node_irreps
            node_irreps = self.norm_ffn(node_irreps)
            node_irreps = self.ffn(node_irreps)

            node_irreps = node_irreps_res + node_irreps

        # # Mixed Resolution Zero-Masking (FFN Layer)
        # if self.mixed_resolution and self.mixed_resolution.get("enabled", False):
        #     lf_atoms = self.mixed_resolution.get("lf_atoms", [1])
        #     is_lf = torch.isin(atomic_numbers, torch.tensor(lf_atoms, device=atomic_numbers.device))
        #     if is_lf.any():
        #         node_irreps[is_lf, 1:, :] = 0.0

        # node_irreps_res = node_irreps
        # node_irreps = self.norm_3(node_irreps, batch=batch)
        # node_irreps = self.ffn_vec2scalar(node_irreps)
        # node_irreps = node_irreps_res + node_irreps
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
        # self.gbf = GaussianLayer(number_of_basis)  # default output_dim = 128
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
        kv_atomic_numbers=None,
        **kwargs,
    ):


        query_idx = batched_data["query_idx"]
        neighbor_idx = batched_data["neighbor_idx"]
        query_idx_long = query_idx.to(torch.long)
        neighbor_idx_long = neighbor_idx.to(torch.long)
        num_queries = node_input.shape[0]

        # `atomic_numbers` is read on BOTH sides of the edge: target_embedding is
        # gathered by the QUERY-local query_idx, source_embedding by
        # f_outcell_index (expanded row -> KV row). Those two index spaces coincide
        # only when the queries ARE the KV bank (the full-graph init) or when the
        # KV bank starts with the query rows (the a2h [heavy; light] layout, which
        # is why it passes _merged_an_a2h).
        # cross_mode="interleaved_flash" initialises each stream from a pack whose
        # queries are a strict SUBSET of an all-atom KV bank, so the two spaces
        # differ and the KV side must be named separately. None -> the historical
        # single-array behaviour, bit-exact for every existing caller.
        _kv_an = atomic_numbers if kv_atomic_numbers is None else kv_atomic_numbers
        if self.use_atom_edge:
            tgt_atm = self.target_embedding(atomic_numbers)
            src_atm = self.source_embedding(_kv_an)[batched_data["f_outcell_index"]]
            edge_dis_embed = torch.cat(
                [edge_scalars, tgt_atm[query_idx_long], src_atm[neighbor_idx_long]],
                dim=-1,
            )
        else:
            edge_dis_embed = edge_scalars

        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)
        with record_function("e3nn sph"):
            lxs = o3._spherical_harmonics._spherical_harmonics(
                lmax=self.lmax,
                x=edge_vec[..., 0],
                y=edge_vec[..., 1],
                z=edge_vec[..., 2],
            )

        # Reuse the sparse-QK CSR metadata (q_row_ptr / q_alpha_index) already
        # built for this (sub-)graph so the fused kernel needs no extra prep.
        # Falls back to recomputing it internally when absent (None). This works
        # for every cross_mode (full-graph batched_data, a2h_data, a2l_data) since
        # query_idx and the per-edge tensors share the same edge ordering as the
        # metadata's q_alpha_index permutation.
        # batched_data may be a torch_geometric Data object (the aligned node-init
        # passes the raw Data, not a dict copy), not a plain dict — but BOTH
        # support .get(). The old `isinstance(batched_data, dict)` guard rejected
        # Data and forced metadata=None, so edge_degree_outer_triton rebuilt the
        # CSR (bincount+cumsum) on every l-order. Duck-type on .get() instead so
        # the prebuilt metadata is reused (matches the non-MR e2former.py path,
        # which reads batched_data["metadata"] directly).
        _degree_metadata = (
            batched_data.get("metadata", None)
            if hasattr(batched_data, "get")
            else None
        )

        node_features = []
        for idx in range(len(self.irreps_node_embedding)):
            edge_fea = self.gbf_projs[idx](edge_dis_embed) * poly_dist.reshape(-1,1)
            l_start = idx**2
            l_end = (idx + 1) ** 2
            edge_sh = lxs[:, l_start:l_end].to(edge_fea.dtype)
            # Fused outer-product + scatter (no [E, 2l+1, H] intermediate, fp32
            # accumulation, custom double-backward). Mirrors the main E2former
            # EdgeDegreeEmbeddingNetwork_higherorder path.
            lx_embed = edge_degree_outer_triton(
                edge_sh=edge_sh,
                edge_fea=edge_fea,
                query_index=query_idx,
                num_queries=num_queries,
                metadata=_degree_metadata,
            )
            node_features.append(lx_embed)

        node_features = torch.cat(node_features, dim=1) / self.avg_aggregate_num
        return node_features
        # base_features = torch.zeros_like(node_features)
        # base_features[:, 0] = node_input
        # return self.proj(torch.cat([base_features, node_features], dim=-1))


class LightAngularEmbedding(torch.nn.Module):
    """Element-agnostic AEV-style angular descriptor for light (H) atoms.

    For each light atom i, enumerates all pairs of neighbours (j, k) drawn
    from the *union* of the light-light (ll) and light-heavy (lh) sub-graphs,
    and accumulates:

        G[i, s, z] = Σ_{j≠k ∈ N(i)}
                       (1 + cos θ_ijk)^ζ_z
                     × exp(-η × ((r_ij + r_ik)/2 - r_s)^2)
                     × f_c(r_ij) × f_c(r_ik)

    where f_c is the polynomial cutoff (poly_dist), already computed upstream.
    The descriptor is element-agnostic (no per-element buckets), so its
    dimension  n_ang_shift × n_zeta  is constant regardless of the number of
    chemical species present.

    The output is projected to `out_dim` via a learnable Linear layer and added
    to the l=0 (scalar) channel of f_light_irreps.
    """

    def __init__(
        self,
        out_dim: int,
        n_ang_shift: int,
        n_zeta: int = 4,
        cutoff: float = 5.0,
        eta: float = 1.0,
    ):
        super().__init__()
        self.n_ang_shift = n_ang_shift
        self.n_zeta = n_zeta
        self.eta = eta
        self.in_dim = n_ang_shift * n_zeta

        # Fixed (non-learnable) basis parameters
        self.register_buffer("ang_shifts", torch.linspace(0.0, cutoff, n_ang_shift))
        self.register_buffer(
            "zetas",
            torch.tensor([float(2 ** k) for k in range(n_zeta)], dtype=torch.float32),
        )

        # Learnable projection: in_dim → out_dim
        self.proj = nn.Linear(self.in_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        query_idx: torch.Tensor,   # [E_all] int64, light-local [0..n_nodes-1]
        edge_vec:  torch.Tensor,   # [E_all, 3]  raw (NOT normalised), center→nbr
        dist:      torch.Tensor,   # [E_all]     distances
        poly_dist: torch.Tensor,   # [E_all]     polynomial cutoff f_c(r)
        n_nodes:   int,
    ) -> torch.Tensor:             # [n_nodes, out_dim]
        device = edge_vec.device
        dtype  = edge_vec.dtype
        E_all  = query_idx.shape[0]

        # Guard: trivially empty
        if E_all == 0 or n_nodes == 0:
            return torch.zeros(n_nodes, self.proj.out_features, device=device, dtype=dtype)

        q_long = query_idx.to(torch.long)   # [E_all]

        # Step 1: per-node degree
        deg = torch.zeros(n_nodes, dtype=torch.long, device=device)
        deg.index_add_(0, q_long, torch.ones(E_all, dtype=torch.long, device=device))

        if deg.max().item() <= 1:
            # No node has ≥2 neighbours → no valid triplets
            return torch.zeros(n_nodes, self.proj.out_features, device=device, dtype=dtype)

        # Step 2: sort edges by query_idx so each node's edges form a contiguous block
        sort_order = torch.argsort(q_long, stable=True)
        q_long    = q_long[sort_order]
        edge_vec  = edge_vec[sort_order]
        dist      = dist[sort_order]
        poly_dist = poly_dist[sort_order]

        # Step 3: segment start indices — seg_start[i] = first edge index for node i
        seg_start = torch.zeros(n_nodes, dtype=torch.long, device=device)
        if n_nodes > 1:
            seg_start[1:] = deg.cumsum(0)[:-1]

        # Step 4: enumerate all ordered pairs (left_edge, right_edge) per node
        deg_per_edge     = deg[q_long]          # [E_all]
        n_pairs_per_node = deg * deg            # [n_nodes]
        total_pairs      = int(n_pairs_per_node.sum().item())

        if total_pairs == 0:
            return torch.zeros(n_nodes, self.proj.out_features, device=device, dtype=dtype)

        # left_edge[p]: edge acting as "j" — each edge repeated deg[query[e]] times
        left_edge = torch.repeat_interleave(
            torch.arange(E_all, dtype=torch.long, device=device),
            deg_per_edge,
        )  # [total_pairs]

        # right_edge[p]: edge acting as "k"
        pair_start = torch.zeros(n_nodes, dtype=torch.long, device=device)
        if n_nodes > 1:
            pair_start[1:] = n_pairs_per_node.cumsum(0)[:-1]

        node_of_pair    = q_long[left_edge]     # [total_pairs]
        global_pair_idx = torch.arange(total_pairs, dtype=torch.long, device=device)
        slot_in_node    = global_pair_idx - pair_start[node_of_pair]
        right_local     = slot_in_node % deg[node_of_pair]
        right_edge      = seg_start[node_of_pair] + right_local   # [total_pairs]

        # Remove diagonal (j == k)
        keep = left_edge != right_edge
        if not keep.any():
            return torch.zeros(n_nodes, self.proj.out_features, device=device, dtype=dtype)

        left_edge    = left_edge[keep]
        right_edge   = right_edge[keep]
        node_of_pair = node_of_pair[keep]   # [n_trips]

        # Step 5: cosine angle at centre i
        vec_j = edge_vec[left_edge]         # [n_trips, 3]
        vec_k = edge_vec[right_edge]        # [n_trips, 3]
        r_ij  = dist[left_edge]             # [n_trips]
        r_ik  = dist[right_edge]            # [n_trips]

        cos_theta = (vec_j * vec_k).sum(-1) / (
            r_ij.clamp(min=1e-8) * r_ik.clamp(min=1e-8)
        )
        cos_theta = cos_theta.clamp(-1.0, 1.0)   # [n_trips]

        # Step 6: angular factor (1 + cos θ)^ζ — [n_trips, n_zeta]
        ang_factor = (1.0 + cos_theta).unsqueeze(-1) ** self.zetas.to(dtype)

        # Step 7: radial Gaussian factor — [n_trips, n_ang_shift]
        r_mean = 0.5 * (r_ij + r_ik)
        dr = r_mean.unsqueeze(-1) - self.ang_shifts.to(dtype).unsqueeze(0)
        radial_factor = torch.exp(-self.eta * dr * dr)

        # Step 8: cutoff damping — [n_trips]
        cutoff_factor = poly_dist[left_edge] * poly_dist[right_edge]

        # Step 9: combine → [n_trips, n_ang_shift * n_zeta]
        # radial_factor: [n_trips, n_ang_shift, 1]
        # ang_factor:    [n_trips, 1,           n_zeta]
        descriptor = (
            radial_factor.unsqueeze(-1)
            * ang_factor.unsqueeze(-2)
            * cutoff_factor.reshape(-1, 1, 1)
        ).reshape(-1, self.in_dim)

        # Step 10: scatter-sum over triplets per node
        out = torch.zeros(n_nodes, self.in_dim, device=device, dtype=dtype)
        out.index_add_(0, node_of_pair, descriptor)

        # Step 11: project to out_dim
        return self.proj(out)   # [n_nodes, out_dim]


class A2LUpProjection(torch.nn.Module):
    """Light irreps -> heavy irreps for the ``"A2L"`` block role (``cross_mode="interleaved"``).

    Two regimes, selected by ``lmax_light`` vs ``lmax``
    --------------------------------------------------
    ``SO3_Linear_e2former``'s weight is ``[(lmax+1), out, in]`` -- ONE matrix per
    ``l``, applied to that ``l``'s ``2l+1`` components. It is a per-order channel
    mixer and cannot create angular orders that are not already in its input.

    * ``lmax_light < lmax`` (**tensor-product regime**): a light space truncated
      below the heavy ``lmax`` cannot be lifted back into the heavy irreps by any
      linear map -- every ``l > lmax_light`` component of the result would be
      identically zero, which is the direct-force-head pathology (light atoms'
      ``l >= 1`` output identically zero) reproduced at EVERY A2L block instead of
      only at the tail, and it would leave the light ``l = 2`` content frozen at
      its ``edge_deg`` initialisation forever. The lift must therefore be a
      genuine tensor product against a geometric quantity -- see "What it
      computes" below. This is the expensive regime: its per-EDGE radial trunk
      (``gbf_proj``) dominates the whole up-projection, far more than the
      Clebsch-Gordan contraction itself.

    * ``lmax_light == lmax`` (**linear regime**, ``self.linear_lift``): the light
      space already carries every order the heavy space has, so a plain
      ``SO3_Linear`` is a COMPLETE equivariant lift -- no order comes out
      identically zero and nothing has to be synthesised from the geometry. The
      radial trunk, the CG buffer and the per-path weights are then not built at
      all and the edge arguments of :meth:`forward` are ignored (the caller may
      keep passing them; they are already materialised for the light TransBlock,
      so there is nothing to skip on its side). The up-projection then collapses
      to a single ``SO3_Linear`` instead of the per-edge radial trunk plus
      Clebsch-Gordan contraction.

      What this trades away is the lift's *geometric bilinearity*: the tensor
      product makes ``delta`` a bilinear function of (light feature, local
      geometry), so terms like ``(light l=0) x Y_2(neighbour directions)`` are
      three-body at the lift, whereas the linear lift applies the same matrix in
      every environment and its ``l = 2`` content can only come from the light
      stream's own ``l = 2``. That coupling is not lost from the model -- the
      light TransBlocks see the edge geometry -- but the body order AT THE LIFT
      drops by one. Raising ``lmax_light`` also costs one more per-``l`` radial
      trunk in ``light_edge_deg_embed`` (once per forward, not per A2L block).

    What it computes (``lmax_light < lmax`` only)
    ---------------------------------------------
    With ``x`` the light-space block output of light atom ``q`` (``[Ll, C_L]``),
    ``E(q)`` its light-query edges, ``r̂_e`` the edge direction and ``u(d_e)`` the
    polynomial cutoff::

        G[q, l2]   = (1 / avg_aggregate_num) * Σ_{e∈E(q)} Y_{l2}(r̂_e) ⊗ R_{l2}(rbf_e) · u(d_e)
        delta[q]   = P_out( Σ_p  w_p ⊙ ( x[q] ⊗^{(l1,l2)->l3}_{CG} G[q] ) )

    i.e. the "edge tensor product ``out_light ⊗ Y_l(r̂_ij)``, aggregated over
    light-query edges". Because ``x[q]`` is the QUERY's own
    feature it does not depend on ``e`` and factors straight out of the edge sum,
    so the algebraically identical form above is used instead: the per-edge
    message ``[E, (lmax+1)^2, C_L]`` -- ~0.8 GB on a 6000-atom water box -- is
    never materialised, and ``G`` reuses the same fused ``edge_degree_outer_triton``
    kernel that ``EdgeDegreeEmbeddingNetwork_higherorder`` uses. The output is
    exactly equivariant: ``G`` is equivariant by construction (invariant radial
    weights times ``Y_l``), the contraction is a Clebsch-Gordan contraction, and
    the per-path weights are scalars.

    ``P_out`` (``SO3_Linear``, ``C_L -> C_H``) runs AFTER the edge aggregation, so
    the channel expansion is ``O(N_light)`` rather than ``O(E_light)``.

    Conventions: ``o3.wigner_3j`` and ``o3._spherical_harmonics`` are both taken
    from e3nn and are mutually consistent, which is the same pair the rest of the
    model's irreps are built from. Do NOT substitute ``module_utils.sph_fromxyz``
    here -- it uses a different l=1 component order.
    """

    def __init__(
        self,
        C_light,
        C_heavy,
        lmax_light,
        lmax,
        number_of_basis,
        avg_aggregate_num=10,
        use_layer_norm=True,
    ):
        super().__init__()
        assert 0 <= lmax_light <= lmax
        self.C_light = C_light
        self.C_heavy = C_heavy
        self.lmax_light = lmax_light
        self.lmax = lmax
        self.avg_aggregate_num = avg_aggregate_num
        _Ll = (lmax_light + 1) ** 2
        _Lh = (lmax + 1) ** 2

        # Linear regime: the light space spans every heavy order, so out_proj
        # alone is a complete lift. Build nothing else -- no radial trunk, no CG
        # buffer, no path weights (see the class docstring).
        self.linear_lift = lmax_light == lmax
        if self.linear_lift:
            self.num_paths = 0
            self.paths = ()
            self.out_proj = SO3_Linear_e2former(
                C_light, C_heavy, lmax=lmax, bias=True
            )
            return

        # One radial trunk for every l2 (the per-l heads are the last GEMM's
        # output slices), mirroring EdgeDegreeEmbeddingNetwork_higherorder's
        # [nb, nb/2, nb/2, C] shape but with a single shared trunk -- this trunk,
        # not the channel count, dominates the cost of the light stream's edge
        # term.
        _hidden = min(number_of_basis // 2, 128)
        self.gbf_proj = RadialProfile(
            [number_of_basis, _hidden, _hidden, (lmax + 1) * C_light],
            use_layer_norm=use_layer_norm,
        )

        # Clebsch-Gordan paths (l1 <= lmax_light) x (l2 <= lmax) -> (l3 <= lmax),
        # stacked into one dense [P, Ll, Lsh, Lh] buffer. Tiny (9x4x9x9 at the
        # reference hyperparameters) and deterministic, so it is registered
        # non-persistent: it never enters the state_dict.
        _paths = []
        _cg = []
        for _l1 in range(lmax_light + 1):
            for _l2 in range(lmax + 1):
                for _l3 in range(abs(_l1 - _l2), min(_l1 + _l2, lmax) + 1):
                    _w = o3.wigner_3j(_l1, _l2, _l3)   # unit Frobenius norm
                    _t = torch.zeros(_Ll, _Lh, _Lh)
                    _t[
                        _l1 ** 2:(_l1 + 1) ** 2,
                        _l2 ** 2:(_l2 + 1) ** 2,
                        _l3 ** 2:(_l3 + 1) ** 2,
                    ] = _w * math.sqrt(2 * _l3 + 1)    # "component" normalisation
                    _paths.append((_l1, _l2, _l3))
                    _cg.append(_t)
        self.paths = tuple(_paths)
        self.num_paths = len(_paths)
        self.register_buffer(
            "cg", torch.stack(_cg, dim=0).reshape(self.num_paths, _Ll * _Lh, _Lh),
            persistent=False,
        )
        # Per-path, per-channel mixing weight. Deliberately NOT zero-initialised:
        # a zero init makes the block an exact no-op at random weights, which
        # would make the "light l=2 is non-zero and changes between blocks" gate
        # pass vacuously.
        self.path_weight = torch.nn.Parameter(
            torch.empty(self.num_paths, C_light)
        )
        _bound = 1 / math.sqrt(self.num_paths)
        torch.nn.init.uniform_(self.path_weight, -_bound, _bound)

        self.out_proj = SO3_Linear_e2former(C_light, C_heavy, lmax=lmax, bias=True)

    def extra_repr(self):
        return (
            f"C_light={self.C_light}, C_heavy={self.C_heavy}, "
            f"lmax_light={self.lmax_light}, lmax={self.lmax}, "
            f"linear_lift={self.linear_lift}, num_paths={self.num_paths}"
        )

    def forward(
        self,
        x_light,        # [N_l, Ll, C_L]  light-space block output (queries)
        query_idx,      # [E_l]           light-LOCAL query id per light-query edge
        edge_vec,       # [E_l, 3]
        poly_dist,      # [E_l]
        edge_scalars,   # [E_l, number_of_basis]
        metadata,       # CSR tuple of the light-query edge pack
        num_queries,    # N_l
    ):
        _Ll = (self.lmax_light + 1) ** 2
        _Lh = (self.lmax + 1) ** 2

        if self.linear_lift:
            # _Ll == _Lh here, so the slice only guards against a caller handing
            # over a WIDER tensor. A narrower one means out_proj (built at the
            # heavy lmax) would silently receive fewer orders than it has weights
            # for; fail here rather than inside SO3_Linear's einsum, where the
            # message names neither this module nor lmax_light.
            assert x_light.shape[1] >= _Lh, (
                f"linear_lift expects at least {_Lh} irreps components "
                f"(lmax={self.lmax}), got {x_light.shape[1]}. linear_lift is only "
                "valid when the light space spans every heavy order "
                f"(lmax_light={self.lmax_light} == lmax={self.lmax}); a truncated "
                "light space needs the tensor-product regime."
            )
            # Every edge argument is unused by design.
            return self.out_proj(x_light[:, :_Lh])

        edge_fea = self.gbf_proj(edge_scalars) * poly_dist.reshape(-1, 1)
        _dir = torch.nn.functional.normalize(edge_vec, dim=-1)
        lxs = o3._spherical_harmonics._spherical_harmonics(
            lmax=self.lmax, x=_dir[..., 0], y=_dir[..., 1], z=_dir[..., 2],
        ).to(edge_fea.dtype)

        # G[q, l2-block, c] -- one fused outer-product+scatter per l2.
        _g = []
        for _l2 in range(self.lmax + 1):
            _g.append(
                edge_degree_outer_triton(
                    edge_sh=lxs[:, _l2 ** 2:(_l2 + 1) ** 2],
                    edge_fea=edge_fea[
                        :, _l2 * self.C_light:(_l2 + 1) * self.C_light
                    ].contiguous(),
                    query_index=query_idx,
                    num_queries=num_queries,
                    metadata=metadata,
                )
            )
        G = torch.cat(_g, dim=1) / self.avg_aggregate_num       # [N_l, Lh, C_L]

        # Fold the per-path weights into the CG tensor FIRST (a [Ll*Lh, Lh, C_L]
        # object, independent of N) so the node-side contraction is a single
        # per-channel bmm instead of a per-path loop over [N, Lh, C_L] tensors.
        cgw = torch.einsum("pmk,pc->mkc", self.cg.to(G.dtype), self.path_weight)
        y = (x_light[:, :_Ll].unsqueeze(2) * G.unsqueeze(1)).reshape(
            -1, _Ll * _Lh, self.C_light
        )                                                       # [N_l, Ll*Lh, C_L]
        out = torch.einsum("nmc,mkc->nkc", y, cgw)              # [N_l, Lh, C_L]
        return self.out_proj(out)                               # [N_l, Lh, C_H]


class FlashUpProjection(torch.nn.Module):
    """Light state -> heavy-KV rows for ``"A2H-flash"`` (``cross_mode="interleaved_flash"``).

    Unlike :class:`A2LUpProjection` this lift's output is **never** a persistent
    state -- it is only ever the light half of one block's KV bank, which the
    heavy attention immediately contracts against the edge geometry. That is what
    makes the linear regime admissible at ``lmax_light < lmax``, and it is the
    single decision this module exists to encode:

    * ``regime="linear"`` (**default**): an ``SO3_Linear_e2former(C_L -> C_H)``
      built at ``lmax_light``, zero-padded up to ``(lmax+1)^2``. The padded orders
      are NOT dead: with ``attn_type`` in ``{"first-order", "so2-first-order"}``
      the value path aggregates ``Σ_j α_ij · (v_j ⊗ r̂_ij)`` over one ``l2 = 1``
      edge channel, so a KV row carrying orders up to ``lmax_light`` reaches heavy
      output orders up to ``lmax_light + 1`` within the SAME block. At
      ``lmax_light = 1``, ``lmax = 2`` that covers the whole heavy space.

      This is exactly the argument that does NOT hold for ``A2LUpProjection``,
      whose output IS the persistent light state: there the missing orders would
      be frozen at the ``edge_deg`` init forever, which is why that module needs a
      tensor product below ``lmax_light == lmax``. The distinction is the whole
      point of the two-stream rewrite -- do not "unify" the two modules.

      Two preconditions, both asserted by the caller at ``__init__``:
      ``attn_type`` must create orders (``"zero-order"`` does not: its value path
      is ``proj_zero``, another per-order channel mixer, so light rows would
      contribute *exactly zero* to heavy ``l > lmax_light`` at every layer -- the
      same silent class of failure as a light-atom direct-force head that reads
      structurally zero rows), and at
      ``lmax_light = 0`` the heavy ``l = 2`` needs two hops through the heavy
      stream (legal, warned about, not asserted).

    * ``regime="tp"``: delegates to :class:`A2LUpProjection` unchanged, i.e. the
      per-edge radial trunk + Clebsch-Gordan contraction. Kept because the linear
      regime is an algebraic argument, not a measurement; its edge arguments are
      the light-query edge pack, already materialised for the light TransBlock.
    """

    def __init__(
        self,
        C_light,
        C_heavy,
        lmax_light,
        lmax,
        number_of_basis,
        regime="linear",
        avg_aggregate_num=10,
        use_layer_norm=True,
    ):
        super().__init__()
        assert 0 <= lmax_light <= lmax
        assert regime in ("linear", "tp"), (
            f"flash_up_proj must be 'linear' or 'tp', got {regime!r}"
        )
        self.regime = regime
        self.lmax_light = lmax_light
        self.lmax = lmax
        self.C_light = C_light
        self.C_heavy = C_heavy
        if regime == "linear":
            self.proj = SO3_Linear_e2former(
                C_light, C_heavy, lmax=lmax_light, bias=True
            )
            self.tp = None
        else:
            self.proj = None
            self.tp = A2LUpProjection(
                C_light=C_light,
                C_heavy=C_heavy,
                lmax_light=lmax_light,
                lmax=lmax,
                number_of_basis=number_of_basis,
                avg_aggregate_num=avg_aggregate_num,
                use_layer_norm=use_layer_norm,
            )

    def extra_repr(self):
        return (
            f"regime={self.regime}, C_light={self.C_light}, "
            f"C_heavy={self.C_heavy}, lmax_light={self.lmax_light}, "
            f"lmax={self.lmax}"
        )

    def forward(
        self,
        x_light,        # [N_l, Ll, C_L]  the persistent light state
        query_idx=None,     # tp regime only: light-LOCAL query id per light-query edge
        edge_vec=None,      # tp regime only
        poly_dist=None,     # tp regime only
        edge_scalars=None,  # tp regime only
        metadata=None,      # tp regime only
        num_queries=None,   # tp regime only
    ):
        _Ll = (self.lmax_light + 1) ** 2
        _Lh = (self.lmax + 1) ** 2
        if self.regime == "tp":
            return self.tp(
                x_light=x_light,
                query_idx=query_idx,
                edge_vec=edge_vec,
                poly_dist=poly_dist,
                edge_scalars=edge_scalars,
                metadata=metadata,
                num_queries=num_queries,
            )
        out = self.proj(x_light[:, :_Ll])                       # [N_l, Ll, C_H]
        if _Ll < _Lh:
            # Pad the (lmax_light, lmax] orders with zeros. F.pad's last pair
            # addresses dim -2 for a 3-D tensor, i.e. the SH component axis.
            out = F.pad(out, (0, 0, 0, _Lh - _Ll))
        return out                                              # [N_l, Lh, C_H]


def compile_module(module,compile_options = {"triton.cudagraphs": False}):
    try:
        return torch.compile(module, options=compile_options)
    except TypeError:
        return torch.compile(module, mode="max-autotune")


def cat_layer_slot_indices(num_blocks, cat_layer_output, cat_layer_indices):
    """0-based block indices whose outputs are concatenated by ``cat_layer_output``.

    This is the single source of truth shared by ``E2formerMR`` (which concatenates
    that many per-layer outputs along the channel axis) and
    ``MDEnergyForceMultiHead`` (which widens its head input dim by the same
    factor). They MUST agree, otherwise the head sees a channel-dim mismatch.

    ``cat_layer_indices`` is **1-based** (block 1 .. block num_blocks) and must be
    strictly increasing: the order fixes the channel layout of the concat, so a
    repeated or out-of-order entry is a typo rather than an intent. Returns an
    empty tuple when the feature is off.
    """
    if not cat_layer_output:
        if cat_layer_indices:
            raise ValueError(
                "cat_layer_indices was given but cat_layer_output is False; a "
                "silently ignored layer selection is worse than an error."
            )
        return ()
    if not cat_layer_indices:
        raise ValueError(
            "cat_layer_output=True requires cat_layer_indices (1-based, strictly "
            f"increasing, within [1, {num_blocks}]), e.g. [3, {num_blocks}]."
        )
    idx = [int(i) for i in cat_layer_indices]
    if any(i < 1 or i > num_blocks for i in idx):
        raise ValueError(
            f"cat_layer_indices entries must be 1-based within [1, {num_blocks}], "
            f"got {idx}"
        )
    if any(b <= a for a, b in zip(idx, idx[1:])):
        raise ValueError(
            f"cat_layer_indices must be strictly increasing, got {idx}"
        )
    return tuple(i - 1 for i in idx)


class E2formerMR(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding="128x0e+128x1e+128x2e",
        num_layers=6,
        max_neighbors=300,
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
        ffn_grid_resolution: int = 18,
        ffn_hidden_multiplier: int = 1,
        # use_compile = False,

        avg_degree=23.01,
        # force_head=None,
        with_cluster=False,
        long_range_layers = 2,
        long_cutoff_lower  = 0.1,
        long_cutoff_upper = 12,
        # mean=None,
        # std=None,
        # scale=None,
        # atomref=None,
        zero_H_high_order: bool = False,
        mixed_resolution: dict = None,
        layerwise_shared_KV_cache: bool = False,
        normalized_radial_profile: bool = True,
        cat_layer_output: bool = False,
        cat_layer_indices=None,
        **kwargs,
    ):
        super().__init__()
        self.mixed_resolution = mixed_resolution
        self.zero_H_high_order = zero_H_high_order
        self.layerwise_shared_KV_cache = layerwise_shared_KV_cache
        self.normalized_radial_profile = normalized_radial_profile
        self.config = locals()
        del self.config['self']


        self.tp_type = tp_type
        if "+" in attn_type:
            self.attn_type = attn_type.split("+")
        else:
            self.attn_type = [attn_type,attn_type]

        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        # The cross stream shares the self cutoff, so this is always max_radius.
        # Defined because E2FormerBackbone reads it via getattr.
        self.r_cross = float(max_radius)
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

        ## this is for f( r_ij )
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
            use_layer_norm="wolayernorm" not in edge_embedtype and normalized_radial_profile,
        )
        # if use_compile:
        #     self.edge_deg_embed_dense = compile_module(self.edge_deg_embed_dense)

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
                ffn_grid_resolution=ffn_grid_resolution,
                ffn_hidden_multiplier=ffn_hidden_multiplier,
                layer_id=i,
                mixed_resolution=self.mixed_resolution,
                normalized_radial_profile=normalized_radial_profile,
            )
            # if use_compile:
            #     self.blocks.append(compile_module(blk))
            # else:
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
        
        ### for long range part
        self.long_range = None
        self.long_cutoff_lower = long_cutoff_lower
        self.long_cutoff_upper = long_cutoff_upper
        self.with_cluster = str(with_cluster)
        if 'node' in self.with_cluster:
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
                    ffn_grid_resolution=ffn_grid_resolution,
                    ffn_hidden_multiplier=ffn_hidden_multiplier,
                    layer_id=i,
                    mix_neighbor=True if "mixnode" in self.with_cluster else False,
                )
                # if use_compile:
                #     self.long_blocks.append(compile_module(blk))
                # else:
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
            
        # ### for moe part
        # self.embedding_charge = nn.Embedding(30, 128)
        # self.embedding_multiplicity = nn.Embedding(30, 128)

        
        if len(self.irreps_node_embedding) == 1:
            self.f_linear = nn.Sequential(
                nn.Linear(self.scalar_dim, self.scalar_dim),
                nn.LayerNorm(self.scalar_dim),
                nn.SiLU(),
                nn.Linear(self.scalar_dim, 3 * self.scalar_dim),
            )

        # Angular descriptor for light atoms (enabled per config; default off)
        self._light_ang_n = 0

        # --- Mixed-Resolution (heavy + light node sets) ---
        self._mr = mixed_resolution
        self._mr_enabled = bool(
            mixed_resolution and mixed_resolution.get("enabled", False)
            and "irreps_light_embedding" in mixed_resolution
        )
        if self._mr_enabled:
            mr = mixed_resolution
            _light_irreps = o3.Irreps(mr["irreps_light_embedding"])
            self.lmax_light = _light_irreps[-1][1][0]
            self.light_scalar_dim = _light_irreps[0][0]

            # Separate token embedding for light nodes
            self.light_default_node_embedding = nn.Embedding(
                atom_type_cnt, self.light_scalar_dim
            )

            # Separate edge-degree initialiser for light nodes
            self.light_edge_deg_embed = EdgeDegreeEmbeddingNetwork_higherorder(
                _light_irreps,
                avg_degree,
                cutoff=self.max_radius,
                number_of_basis=self.number_of_basis,
                use_atom_edge=True,
                use_layer_norm="wolayernorm" not in edge_embedtype and normalized_radial_profile,
            )

            # Separate RBF for light sub-graph
            self.light_rbf = GaussianSmearing(
                self.number_of_basis, cutoff=self.max_radius, basis_width_scalar=2
            )

            # Light TransBlock stack (one per layer, fully independent weights)
            _light_heads   = mr.get("light_attn_heads", self.num_attn_heads)
            _light_head_d  = mr.get("light_attn_scalar_head", self.attn_scalar_head)
            _light_drop    = mr.get("light_alpha_drop", self.alpha_drop)
            _light_ffn     = mr.get("light_ffn_type", ffn_type)
            # `or` rather than a plain .get default: the key is now DECLARED in
            # config_file/backbone_config/e2former.yaml (it has to be -- hydra's
            # struct mode rejects undeclared keys on the command line) with a
            # value of null, so .get finds the key and returns None instead of
            # falling back. None/0 both mean "inherit the heavy grid".
            _light_grid_res = mr.get("light_ffn_grid_resolution", None) or ffn_grid_resolution
            _light_ffn_mult = mr.get("light_ffn_hidden_multiplier", ffn_hidden_multiplier)
            # Head irreps of the light attention. NOTE this is INERT for the light
            # stream as configured: the light blocks are built with
            # attn_type="zero-order" (hardcoded below), whose value path is
            # E2AttentionArbOrder_sparse.proj_zero -- an SO3_Linear built from
            # irreps_node_input/output at self.lmax, which never reads
            # irreps_head. Only the first-order / so2-first-order branches use it.
            # So a scalar-looking default does NOT truncate the light value space
            # (verified: overriding it to "<h>x0e" at lmax_light=2 changes the
            # output by exactly 0), and this must NOT be "fixed" to span
            # lmax_light without also switching the light attn_type -- doing so
            # would silently change behaviour the day someone does switch it.
            _light_irreps_head = mr.get(
                "light_irreps_head", f"{_light_heads}x0e"
            )
            self.light_blocks = nn.ModuleList([
                TransBlock(
                    irreps_node_input=_light_irreps,
                    irreps_node_output=_light_irreps,
                    attn_weight_input_dim=self.number_of_basis,
                    num_attn_heads=_light_heads,
                    attn_scalar_head=_light_head_d,
                    irreps_head=_light_irreps_head,
                    alpha_drop=_light_drop,
                    tp_type=self.tp_type,
                    attn_type="zero-order",  # light nodes are scalar
                    norm_layer=self.norm_layer,
                    ffn_type=_light_ffn,
                    ffn_grid_resolution=_light_grid_res,
                    ffn_hidden_multiplier=_light_ffn_mult,
                    layer_id=i,
                    normalized_radial_profile=normalized_radial_profile,
                )
                for i in range(self.num_layers)
            ])

            # Supported cross_mode values — exactly the two that ship:
            #   "interleaved"      — mixed_resolution.cross_mode_list gives the role
            #                        of EVERY block, so All2Heavy blocks can be
            #                        interleaved with blocks that update the light
            #                        atoms inside the loop. Tokens (X2Y = KV/source is
            #                        X, query/target is Y — the convention used
            #                        throughout this file):
            #                          "A2H"  heavy queries, all-atom KV (or the
            #                                 a2h_kv_union heavy-neighbour union);
            #                                 light rows are not written.
            #                          "A2A"  all-atom queries, all-atom KV. Plain
            #                                 self-attention.
            #                          "H2A"  all-atom queries, HEAVY-ONLY KV. Light
            #                                 atoms are updated from their heavy
            #                                 neighbours mid-loop and thereafter carry
            #                                 an evolving full-irreps state (the H2A
            #                                 interleaved mode).
            #                          "A2L"  LIGHT queries, all-atom KV, computed in the
            #                                 LIGHT irreps space: down-project every atom
            #                                 to [Ll, C_L], run light_blocks[i] with
            #                                 light-only queries, then lift the result back
            #                                 to the heavy irreps with an EDGE TENSOR
            #                                 PRODUCT (A2LUpProjection) and add it to the
            #                                 light rows. Like H2A it updates light atoms
            #                                 mid-loop, but they stay low-resolution
            #                                 instead of being promoted into the full
            #                                 heavy space.
            #                        interleaved_tail picks the read-out. Only "v1" (the
            #                        default) remains: norm_final + full-irreps scatter,
            #                        the block loop feeding the energy head directly.
            #   "interleaved_flash" — TWO persistent node states instead of one:
            #                        H [N_H, (lmax+1)^2, C_H] and L [N_L,
            #                        (lmax_light+1)^2, C_L]. Under "interleaved" the
            #                        light atoms are ROWS of the single heavy-shaped
            #                        tensor, so "mixed resolution" is a statement
            #                        about which rows are queries in which block,
            #                        never about how the state is stored. Here each
            #                        block updates exactly one stream and passes the
            #                        other through unchanged:
            #                          "A2H-flash"  heavy queries, KV = [H ; up(L)],
            #                                       heavy space; writes H only.
            #                          "A2L-flash"  light queries, KV = [down(H) ; L],
            #                                       light space; writes L only.
            #                        Read-out is the usual heavy-shaped tensor with
            #                        light rows left at ZERO, plus a separate light
            #                        tensor published on batched_data, so the energy
            #                        is a sum over two heads on two disjoint atom sets
            #                        (energy_head="distinct" is required; a DIRECT
            #                        force head is likewise two-headed, see
            #                        head.py force_heads_light).
            # Both require an explicit mixed_resolution.cross_mode_list (asserted
            # below), so this default decides which mode an unconfigured run gets,
            # not whether it has a schedule: "interleaved" is the single-stream
            # Standard mode.
            self._cross_mode = mr.get("cross_mode", "interleaved")
            assert self._cross_mode in ("interleaved", "interleaved_flash"), (
                f"cross_mode must be 'interleaved' or 'interleaved_flash', "
                f"got {self._cross_mode!r}"
            )
            # Single source of truth for "is this the two-stream mode?". A plain
            # bool, so dynamo guards it as a constant rather than re-tracing a
            # string comparison inside the traced region.
            self._flash = self._cross_mode == "interleaved_flash"

            # ---- per-block schedule -----------------------------------------
            # Each block's role is given explicitly by cross_mode_list, so the
            # forward loop has a single dispatch.
            #
            # Validated against len(self.blocks), NOT self.num_layers: they are
            # equal today but the forward loop iterates self.blocks, so this is
            # the bound that actually keeps the dispatch in range.
            _sched_raw = mr.get("cross_mode_list", None)
            _n_blocks_cfg = len(self.blocks)
            _VALID_BLOCK_ROLES = ("A2H", "A2A", "H2A", "A2L")
            # The two families are mutually exclusive by construction, not by
            # taste. "H2A" and "A2A" write full heavy irreps onto light rows, so
            # after one of them a light atom is no longer light and the two-stream
            # invariant is gone; "A2H"/"A2L" merely read a single node stream that
            # does not exist under flash. Tokens are matched case-insensitively
            # (upper()) so CLI overrides may spell them in any case.
            _VALID_FLASH_BLOCK_ROLES = ("A2H-FLASH", "A2L-FLASH")
            if self._cross_mode in ("interleaved", "interleaved_flash"):
                _roles = (
                    _VALID_FLASH_BLOCK_ROLES if self._flash else _VALID_BLOCK_ROLES
                )
                _other_roles = (
                    _VALID_BLOCK_ROLES if self._flash else _VALID_FLASH_BLOCK_ROLES
                )
                _other_mode = (
                    "interleaved" if self._flash else "interleaved_flash"
                )
                assert _sched_raw is not None, (
                    f"cross_mode={self._cross_mode!r} requires "
                    "mixed_resolution.cross_mode_list, a list of "
                    f"{_n_blocks_cfg} tokens from {_roles}"
                )
                _sched = [str(tok).strip().upper() for tok in _sched_raw]
                assert len(_sched) == _n_blocks_cfg, (
                    f"cross_mode_list must have exactly num_layers={_n_blocks_cfg} "
                    f"entries (one per block), got {len(_sched)}: {_sched}"
                )
                _crossed = [tok for tok in _sched if tok in _other_roles]
                assert not _crossed, (
                    f"cross_mode={self._cross_mode!r} does not accept the token(s) "
                    f"{_crossed} -- those belong to cross_mode={_other_mode!r}. "
                    + (
                        "'A2A'/'H2A' promote light atoms into the full heavy irreps, "
                        "which destroys the two-stream invariant outright, and "
                        "'A2H'/'A2L' read a single node stream that does not exist "
                        "here."
                        if self._flash else
                        "The flash tokens carry two persistent node states; this mode "
                        "has only one."
                    )
                )
                _bad = [tok for tok in _sched if tok not in _roles]
                assert not _bad, (
                    f"cross_mode_list tokens must be one of {_roles}, got {_bad}"
                )
                # tuple, not list: dynamo guards it as a constant.
                self._block_schedule = tuple(_sched)
                if self._flash:
                    # Flash owns its read-out: the heavy stream is scattered
                    # as usual with light rows left at ZERO, and the light stream is
                    # published alongside. Neither "v1" (which scatters full irreps
                    # for every atom) nor "v2" (which reduces the single stream to a
                    # light scalar) describes it, so interleaved_tail is not read
                    # here -- and because the key ships with a non-null default in
                    # e2former.yaml, "was it set?" is not answerable. A distinct
                    # sentinel keeps every `_block_tail == "v1"/"v2"` test below
                    # correctly False.
                    self._block_tail = "flash"
                else:
                    # Default "v1": the semantically coherent tail for the H2A mode. It keeps
                    # full irreps for EVERY atom (so a mid-loop H2A block's light l>=1
                    # output survives to the read-out, and the direct force head is usable
                    # again), and it costs less than "v2" — no reduction and no All2Light
                    # block over the light-query edge set.
                    self._block_tail = str(mr.get("interleaved_tail", "v1")).lower()
                    assert self._block_tail == "v1", (
                        "interleaved_tail must be 'v1', got "
                        f"{self._block_tail!r}"
                    )
            # Which edge packs the forward has to build. A pure-H2A schedule must
            # not pay for the All2Heavy CSR (and vice versa).
            #
            # Under flash BOTH packs are unconditional: the a2l pack feeds
            # the light INITIALISATION and the a2h_heavy pack the heavy one, so a
            # schedule with no A2L-flash block still needs the light pack (and vice
            # versa). Inheriting the role-driven gating would leave a stream
            # uninitialised, which is a shape error at best and silent zeros at
            # worst.
            self._needs_a2h = (
                self._flash
                or (self._block_schedule is not None
                    and "A2H" in self._block_schedule)
            )
            self._needs_h2a = (
                not self._flash
                and self._block_schedule is not None
                and "H2A" in self._block_schedule
            )
            self._needs_a2l = (
                self._flash
                or (self._block_schedule is not None
                    and "A2L" in self._block_schedule)
            )
            # Block index -> A2L slot (-1 for every other role). Each A2L block
            # gets its OWN down/up projection: they bracket different points of
            # the heavy stream and there is no reason to tie them. A tuple, so
            # dynamo guards it as a constant.
            self._a2l_slot = tuple(
                (
                    sum(1 for r in self._block_schedule[:i] if r == "A2L")
                    if self._block_schedule is not None
                    and self._block_schedule[i] == "A2L"
                    else -1
                )
                for i in range(_n_blocks_cfg)
            ) if self._block_schedule is not None else ()
            _n_a2l = sum(1 for s in self._a2l_slot if s >= 0)
            # NOT self._needs_a2l: that flag is now "does the forward have to build
            # the light-query EDGE PACK", which flash turns on unconditionally for
            # the light initialisation. The "A2L" token's own modules must stay tied
            # to the token, or a flash config would carry a dead a2l_down_proj /
            # a2l_up_proj stack in its state_dict.
            _has_a2l_token = (
                self._block_schedule is not None and "A2L" in self._block_schedule
            )
            if _has_a2l_token:
                # A scalar light space makes the whole point of A2L vacuous: the
                # block would carry no l>=1 state of its own, and the up-projection
                # would be lifting a pure scalar.
                assert self.lmax_light >= 1, (
                    "cross_mode_list contains 'A2L' but irreps_light_embedding has "
                    f"lmax_light={self.lmax_light}: the A2L block runs in the LIGHT "
                    "irreps space, so a scalar light space leaves it with no "
                    "equivariant state to attend with. Use e.g. "
                    f"'{self.light_scalar_dim}x0e+{self.light_scalar_dim}x1e'."
                )
                assert self.lmax_light <= self.lmax, (
                    f"irreps_light_embedding lmax ({self.lmax_light}) exceeds the "
                    f"heavy lmax ({self.lmax}); the A2L down-projection truncates "
                    "the heavy irreps."
                )
                # Down-projection: heavy irreps -> light irreps, per A2L block.
                # Built at lmax_light (not at the heavy lmax then truncated): the
                # SO3 linear's per-l weights are independent, so the two are the
                # same function, and this one does not carry weights for orders it
                # immediately discards.
                self.a2l_down_proj = nn.ModuleList([
                    SO3_Linear_e2former(
                        self.scalar_dim, self.light_scalar_dim,
                        lmax=self.lmax_light, bias=True,
                    )
                    for _ in range(_n_a2l)
                ])
                # Up-projection: light irreps -> heavy irreps, via an edge tensor
                # product (a linear map cannot synthesise l > lmax_light -- see
                # A2LUpProjection's docstring).
                self.a2l_up_proj = nn.ModuleList([
                    A2LUpProjection(
                        C_light=self.light_scalar_dim,
                        C_heavy=self.scalar_dim,
                        lmax_light=self.lmax_light,
                        lmax=self.lmax,
                        number_of_basis=self.number_of_basis,
                        avg_aggregate_num=avg_degree,
                        use_layer_norm=(
                            "wolayernorm" not in edge_embedtype
                            and normalized_radial_profile
                        ),
                    )
                    for _ in range(_n_a2l)
                ])

            # ---- cross_mode="interleaved_flash": two persistent streams --------
            # Per-block slot tables, the flash twins of _a2l_slot. Each block gets
            # its OWN projection for the same reason the A2L slots do: consecutive
            # A2H-flash blocks bracket different points of the light stream (and
            # vice versa), and there is no reason to tie them.
            self._a2h_flash_slot = tuple(
                (
                    sum(1 for r in self._block_schedule[:i] if r == "A2H-FLASH")
                    if self._flash and self._block_schedule[i] == "A2H-FLASH"
                    else -1
                )
                for i in range(_n_blocks_cfg)
            ) if self._flash else ()
            self._a2l_flash_slot = tuple(
                (
                    sum(1 for r in self._block_schedule[:i] if r == "A2L-FLASH")
                    if self._flash and self._block_schedule[i] == "A2L-FLASH"
                    else -1
                )
                for i in range(_n_blocks_cfg)
            ) if self._flash else ()
            self._flash_up_proj_regime = str(
                mr.get("flash_up_proj", None) or "linear"
            ).lower()
            # The linear regime is a PER-NODE map, so up(L[rows]) == up(L)[rows]
            # and the a2h_kv_union path can select the light rows it needs BEFORE
            # projecting. The tp regime aggregates over the light-query edge pack
            # (num_queries == f_N_light), so it must run on every light row and
            # be sliced after. Hoisted to a bool so the forward branches on a
            # dynamo constant rather than a string compare.
            self._flash_up_proj_linear = self._flash_up_proj_regime == "linear"
            if self._flash:
                _n_a2h_f = sum(1 for s in self._a2h_flash_slot if s >= 0)
                _n_a2l_f = sum(1 for s in self._a2l_flash_slot if s >= 0)
                assert self._flash_up_proj_regime in ("linear", "tp"), (
                    "mixed_resolution.flash_up_proj must be 'linear' or 'tp', got "
                    f"{self._flash_up_proj_regime!r}"
                )
                # lmax_light >= 1: the light stream is a persistent equivariant
                # state here, not a scalar summary. lmax_light == 0 is not
                # forbidden by the algebra (the down-projection would just carry
                # l=0), but it makes A2L-flash a scalar message-passing layer and
                # forces heavy l=2 to travel two hops through the heavy stream.
                assert self.lmax_light <= self.lmax, (
                    f"irreps_light_embedding lmax ({self.lmax_light}) exceeds the "
                    f"heavy lmax ({self.lmax}); the A2L-flash down-projection "
                    "truncates the heavy irreps."
                )
                assert self.lmax_light >= 1, (
                    "cross_mode='interleaved_flash' needs irreps_light_embedding "
                    f"with lmax_light >= 1, got {self.lmax_light}: the light stream "
                    "is a PERSISTENT equivariant state, and a scalar one leaves "
                    "A2L-flash with nothing equivariant to attend with. Use e.g. "
                    f"'{self.light_scalar_dim}x0e+{self.light_scalar_dim}x1e'."
                )
                # The linear lift is admissible ONLY because the heavy
                # attention's value path is Sum_j alpha_ij * (v_j (x) r_hat_ij),
                # which raises a KV row's orders by one per block. With
                # attn_type="zero-order" the value is aggregated as-is through
                # proj_zero -- another per-order channel mixer -- so light rows
                # would contribute EXACTLY ZERO to every heavy l > lmax_light at
                # every layer. Silent, and the same class of failure as a light
                # atom whose direct-force head reads structurally zero rows.
                if (
                    self._flash_up_proj_regime == "linear"
                    and self.lmax_light < self.lmax
                ):
                    # self.attn_type is a 2-element list (first half / second half
                    # of the block stack), so EVERY entry has to create orders --
                    # a single zero-order half would silently freeze the top heavy
                    # orders of the light contribution for those layers.
                    assert all(
                        _a in ("first-order", "so2-first-order")
                        for _a in self.attn_type
                    ), (
                        "cross_mode='interleaved_flash' with flash_up_proj='linear' "
                        f"and lmax_light={self.lmax_light} < lmax={self.lmax} "
                        f"requires an order-creating attn_type, got "
                        f"{self.attn_type!r}. A linear lift creates no angular "
                        "order by itself; only the first-order value path "
                        "(v_j (x) r_hat_ij) carries the light stream into heavy "
                        "l > lmax_light. With 'zero-order' those orders would be "
                        "identically zero at every layer. Use "
                        "flash_up_proj='tp' if the attention must stay zero-order."
                    )
                    if self.lmax_light + 1 < self.lmax:
                        logger.warning(
                            "[MR flash] lmax_light="
                            f"{self.lmax_light} with lmax={self.lmax}: one "
                            "A2H-flash block lifts light rows to heavy order "
                            f"{self.lmax_light + 1} at most, so heavy l > "
                            f"{self.lmax_light + 1} needs a further hop through "
                            "the heavy stream. Legal, but the light stream's "
                            "influence on the top orders is delayed by a layer."
                        )
                # The read-out is a sum over two heads on two DISJOINT atom sets,
                # and the heavy-shaped irreps tensor has structurally zero light
                # rows. A single energy head would read those zeros; the
                # direct force head would predict exactly 0 N on every light atom
                # -- a silent failure.
                assert mr.get("energy_head", "standard") == "distinct", (
                    "cross_mode='interleaved_flash' requires "
                    "mixed_resolution.energy_head='distinct': the light stream is "
                    "never projected into the heavy irreps tensor, whose light rows "
                    "are therefore structurally ZERO. A single ('standard') energy "
                    "head would read those zeros and assign every light atom the "
                    "same constant energy."
                )
                # Up-projection: light state -> the light half of the heavy KV bank.
                self.flash_up_proj = nn.ModuleList([
                    FlashUpProjection(
                        C_light=self.light_scalar_dim,
                        C_heavy=self.scalar_dim,
                        lmax_light=self.lmax_light,
                        lmax=self.lmax,
                        number_of_basis=self.number_of_basis,
                        regime=self._flash_up_proj_regime,
                        avg_aggregate_num=avg_degree,
                        use_layer_norm=(
                            "wolayernorm" not in edge_embedtype
                            and normalized_radial_profile
                        ),
                    )
                    for _ in range(_n_a2h_f)
                ])
                # Down-projection: heavy state -> the heavy half of the light KV
                # bank. Built at lmax_light, so it carries no weights for the heavy
                # orders it immediately discards.
                self.flash_down_proj = nn.ModuleList([
                    SO3_Linear_e2former(
                        self.scalar_dim, self.light_scalar_dim,
                        lmax=self.lmax_light, bias=True,
                    )
                    for _ in range(_n_a2l_f)
                ])
                if _n_a2h_f == 0:
                    logger.warning(
                        "[MR flash] cross_mode_list has no 'A2H-flash' block: the "
                        "heavy stream is initialised and then read out untouched."
                    )
                if _n_a2l_f == 0:
                    logger.warning(
                        "[MR flash] cross_mode_list has no 'A2L-flash' block: the "
                        "light stream is initialised and then read out untouched."
                    )

            # All2Heavy blocks only: restrict each block's
            # KV projection to the heavy-neighbour union (the rows heavy queries
            # actually read) instead of the full node stream. Algebraically
            # lossless -- see _a2h_kv_union_indices.
            #
            # a2h_kv_union_max_ratio skips the compression when the union is that
            # large a fraction of the atoms. Compression only pays off while it
            # actually drops rows: near 100% union U_pri == arange(N), the
            # index_select is a plain copy, and the result is bit-exact. Hence
            # the default is 0.9 rather than 0.5: it keeps the memory saving
            # across the common partial-union band and only declines the
            # degenerate all-heavy case where the copy buys nothing.
            #
            # Gate on the TRUE union size, never on the heavy fraction -- a few
            # ions in water are a tiny fraction of the atoms yet their neighbour
            # union can cover most of the box, so the heavy fraction does not
            # predict the union at all.
            self._a2h_kv_union = bool(mr.get("a2h_kv_union", True))
            self._a2h_kv_union_max_ratio = float(mr.get("a2h_kv_union_max_ratio", 0.9))
            # Environment override so this can be flipped without editing the
            # checkpoint's yaml (mirrors MOLFM_MR_RADIUS_DEBUG below).
            _a2h_kv_union_env = os.environ.get("MOLFM_MR_A2H_KV_UNION", "")
            if _a2h_kv_union_env != "":
                self._a2h_kv_union = _a2h_kv_union_env == "1"
            # cross_mode="interleaved_flash" DOES honour a2h_kv_union. The two
            # remaps of f_outcell_index compose (see the "flash-kv-perm" block in
            # forward): the union's compressed bank is reordered from ascending
            # global atom id into the flash [heavy-part ; light-part] layout, and
            # the union's own expanded->primary map is relabelled through that
            # reordering. The compression is worth MORE here than it is under
            # `interleaved`, because it also removes light rows from the A2H-flash
            # up-projection -- and at f_N_heavy == 0 (pure water under
            # lf_O_in_HOH) the union is empty, which collapses the whole
            # A2H-flash block, up-projection and concat included, to zero rows.
            #
            # The LIGHT side is still untouched: A2L-flash's KV bank is every atom
            # by definition, and a light-neighbour union is not implemented (it
            # would matter for ions in water but not for bulk water).
            # Opt-in per-forward diagnostic (union size + gate decision). Off by
            # default and deliberately NOT a one-shot flag: mutating a module
            # attribute inside the traced region would add a dynamo guard and force
            # a recompile after the first call. Under torch.compile the log only
            # appears while tracing; verify compressed runs from the trace's GEMM
            # row counts instead.
            self._a2h_kv_union_debug = (
                os.environ.get("MOLFM_MR_A2H_KV_UNION_DEBUG", "") == "1"
            )

            # The cross stream shares the self cutoff: r_cross is always
            # max_radius, so cross_rbf aliases self.rbf (zero extra params, and
            # bit-exact against a dedicated RBF at the same width).
            self._cross_radius_decoupled = False
            self.cross_rbf = self.rbf

            # Final norm for light nodes
            self.light_norm_final = get_normalization_layer(
                norm_layer, lmax=self.lmax_light, num_channels=self.light_scalar_dim
            )

            # Project system embedding (scalar_dim=64) down to light_scalar_dim (e.g. 32)
            # so it can be added to light node features in TransBlock.
            self.light_sys_proj = nn.Linear(
                self.scalar_dim, self.light_scalar_dim, bias=False
            )

            # --- Optional angular descriptor for light atoms (ANI-2x style) ---
            # Activated by setting  mixed_resolution.light_angular_basis > 0.
            # Adds element-agnostic three-body angle information to the light
            # node initialisation.  Config keys (all backward-compatible):
            #   light_angular_basis    int  N_ang radial shift values (0 = off)
            #   light_angular_zeta_cnt int  number of ζ exponents    (default 4)
            #   light_angular_eta      float Gaussian width η         (default 1.0)
            self._light_ang_n = mr.get("light_angular_basis", 0)
            if self._light_ang_n > 0:
                self.light_angular_embed = LightAngularEmbedding(
                    out_dim=self.light_scalar_dim,
                    n_ang_shift=self._light_ang_n,
                    n_zeta=mr.get("light_angular_zeta_cnt", 4),
                    cutoff=self.max_radius,
                    eta=float(mr.get("light_angular_eta", 1.0)),
                )

            # Pre-build the light-atom atomic-number tensor as a persistent buffer
            # so it is moved to the correct device together with the model and is
            # never re-allocated inside forward().
            self.register_buffer(
                "_light_atoms_t",
                torch.tensor(mr.get("light_atoms", [1]), dtype=torch.long),
                persistent=False,
            )

        # --- cat_layer_output: per-layer output concat ------------------------
        # The read-out normally sees only the last block's output. With
        # cat_layer_output, the blocks named by cat_layer_indices (1-based) each
        # get their own RMSNorm and are concatenated along the channel axis, so
        # the backbone emits len(indices) * scalar_dim channels and the energy
        # head is widened by the same factor (head.py calls the same
        # cat_layer_slot_indices -- do not duplicate the arithmetic there).
        #
        # Scope: the "v1" tail of the mixed-resolution block loop only. Under
        # "interleaved_flash" a block writes exactly ONE of the two streams, so
        # "the output of block k" is not one object across the atom set; the
        # non-MR path has its own implementation in e2former.py.
        self.cat_layer_output = bool(cat_layer_output)
        if self.cat_layer_output:
            if not (self._mr_enabled and getattr(self, "_block_tail", None) == "v1"):
                raise NotImplementedError(
                    "cat_layer_output is only implemented for the mixed-resolution "
                    "block loop with the 'v1' read-out (cross_mode='interleaved' "
                    "with interleaved_tail='v1'); got mr_enabled="
                    f"{self._mr_enabled}, cross_mode="
                    f"{getattr(self, '_cross_mode', None)!r}, tail="
                    f"{getattr(self, '_block_tail', None)!r}."
                    + (
                        " Under 'interleaved_flash' a block writes exactly ONE of "
                        "the two streams, so \"the output of block k\" is not one "
                        "object across the atom set."
                        if getattr(self, "_flash", False) else ""
                    )
                )
            self._cat_layer_slots = cat_layer_slot_indices(
                len(self.blocks), True, cat_layer_indices
            )
            self.cat_layer_norms = torch.nn.ModuleList([
                get_normalization_layer(
                    norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
                )
                for _ in self._cat_layer_slots
            ])
        else:
            if cat_layer_indices:
                cat_layer_slot_indices(len(self.blocks), False, cat_layer_indices)
            self._cat_layer_slots = ()
        # Channel width of the backbone output. Every read-out path uses this
        # instead of _node_scalar_dim so the concat widens exactly one thing.
        self._out_scalar_dim = (
            len(self._cat_layer_slots) * self.scalar_dim
            if self.cat_layer_output else self._node_scalar_dim
        )

        # self.apply(self._init_weights)

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
        
        tensortype = self.default_node_embedding.weight.dtype
        device = batched_data["pos"].device
        B, L = batched_data["pos"].shape[:2]

        node_pos = batched_data["pos"]
        padding_mask = ~batched_data["atom_masks"]
        node_mask = logical_not(padding_mask)

        # node_mask is a pure function of batched_data["atom_masks"] and never
        # changes during forward, so compute the flat (B*L) valid-atom index ONCE
        # and reuse it for every per-atom gather below.  Each `tensor[node_mask]`
        # boolean-mask index implicitly calls nonzero() — a data-dependent-output
        # op that graph-breaks under torch.compile and runs a redundant kernel —
        # so sharing one index collapses the per-gather nonzeros (and their graph
        # breaks) into a single one.  Integer advanced indexing
        # (`x.reshape(-1, ...)[flat_idx]`) is traceable after that single break.
        flat_idx = node_mask.reshape(-1).nonzero(as_tuple=True)[0]  # [f_N1] ascending

        # node_pos.requires_grad = True
        node_pos = torch.where(
            padding_mask.unsqueeze(dim=-1).repeat(1, 1, 3), 999.0, node_pos
        )

        f_atomic_numbers = batched_data["atomic_numbers"].reshape(-1)[flat_idx]

        f_node_pos = node_pos.reshape(-1, 3)[flat_idx]
        f_N1 = f_node_pos.shape[0]

        f_exp_node_pos = batched_data["f_exp_node_pos"]

        # row-major flat position p = b * L + l  =>  batch index = p // L
        f_batch = flat_idx // L
        f_attn_mask = None

        f_atom_embedding = token_embedding.reshape(-1, token_embedding.shape[-1])[flat_idx]
        
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

        if "query_idx" not in neighbor_info:
            dense_neighbor_idx = neighbor_info["f_sparse_idx_expnode"]
            dense_attn_mask = neighbor_info["f_attn_mask"]
            valid_mask = ~dense_attn_mask.squeeze(-1)
            num_queries, topk = dense_neighbor_idx.shape
            query_idx = (
                torch.arange(num_queries, device=dense_neighbor_idx.device)
                .unsqueeze(1)
                .expand(num_queries, topk)[valid_mask]
                .contiguous()
            )
            neighbor_idx = dense_neighbor_idx[valid_mask].contiguous()
            _sparse_update = {
                "query_idx": query_idx,
                "neighbor_idx": neighbor_idx,
                "f_edge_vec": neighbor_info["f_edge_vec"][valid_mask].contiguous(),
                "f_dist": neighbor_info["f_dist"][valid_mask].contiguous(),
                "f_poly_dist": neighbor_info["f_poly_dist"][valid_mask].contiguous(),
            }
            # MR path never consumes the full-graph CSR — the 4 sub-graph CSRs
            # built in build_mr_subgraph_metadata are what TransBlocks read.
            if not self._mr_enabled:
                _sparse_update["metadata"] = prepare_sparse_qk_edge_index_metadata(
                    query_idx,
                    neighbor_idx,
                    num_queries,
                    batched_data["f_outcell_index"].shape[0],
                )
            neighbor_info.update(_sparse_update)

        batched_data.update(neighbor_info)
        f_edge_vec = batched_data["f_edge_vec"]
        f_dist = batched_data["f_dist"]
        f_poly_dist = batched_data["f_poly_dist"]
        if not self._mr_enabled:
            f_dist_embedding = self.rbf(f_dist)

        # ===== Mixed-Resolution path (heavy + light node sets) =====
        # Pre-compute the heavy/light split BEFORE node initialisation so that
        # we can avoid allocating the full [f_N1, Lh, C_heavy] tensor during the
        # forward pass.  In MR mode only [f_N_heavy, Lh, C_heavy] is ever live;
        # the [f_N_light, 1, C_light] light tensor is much smaller.
        if self._mr_enabled:
            # ----------------------------------------------------------
            # 1. Compute heavy/light split masks (moved up, before init)
            # ----------------------------------------------------------
            # Canonical mask: the explicit batched_data["is_light"] when the
            # caller produced one (the wrapper always does), else a
            # recomputation from atomic_numbers + hoh_o_mask. Same priority order
            # as E2FormerBackbone.forward, so a caller that already resolved the
            # routing can never be overruled here.
            _is_light_2d = resolve_mr_is_light(
                batched_data, self._light_atoms_t, where="E2formerMR.forward",
            ).reshape(-1)
            is_light = _is_light_2d[flat_idx]   # reuse hoisted flat_idx
            is_heavy = ~is_light

            # ==============================================================
            # The block-schedule forward. Both shipping modes run the SAME
            # schedule machinery and differ only in the per-block tokens and the
            # read-out:
            #   interleaved       -> ONE node state in the heavy irreps; the
            #                        schedule (A2H / A2A / H2A / A2L) decides which
            #                        atoms each block writes, and the v1 tail keeps
            #                        full irreps for every atom.
            #   interleaved_flash  -> TWO persistent node states (heavy and light);
            #                        A2H-flash and A2L-flash each write exactly one
            #                        of them, and the flash tail publishes both.
            # Self-contained: mirrors the non-MR init/loop/tail and returns here,
            # never touching the heavy/light-split machinery below.
            # ==============================================================
            _n_kv = batched_data["f_outcell_index"].shape[0]
            _qi = batched_data["query_idx"]
            _ni = batched_data["neighbor_idx"]
            # Hoisted int64 views: both the a2h union and the h2a pack index
            # with these, and each .to(torch.long) is a full [E] / [N_exp] copy.
            _ni_long = _ni.to(torch.long)
            _foc_long = batched_data["f_outcell_index"].to(torch.long)

            # ── Consolidated heavy/light index bookkeeping (computed ONCE) ──
            # Every heavy/light split in this branch is a pure function of the
            # input masks, so build all integer index tensors up front rather
            # than lazily inside / after the block loop. This (a) hoists the
            # only host-device syncs (the nonzero()/.item() here) ahead of the
            # heavy compute — nothing is in flight yet, so they are cheap — and
            # (b) lets every downstream slice use INTEGER advanced indexing
            # (which never syncs), removing the ~15 scattered boolean-mask
            # syncs, including the mid-pipeline TransBlock -> All2Light stall.
            _he = is_heavy[_qi]                                  # bool[E] (gather, no sync)
            _le = is_light[_qi]                                  # bool[E]
            f_heavy_idx = is_heavy.nonzero(as_tuple=True)[0]     # [f_N_heavy] ascending
            f_light_idx = is_light.nonzero(as_tuple=True)[0]     # [f_N_light] ascending
            _he_idx     = _he.nonzero(as_tuple=True)[0]          # heavy-query edge rows
            _le_idx     = _le.nonzero(as_tuple=True)[0]          # light-query edge rows
            _valid_flat = flat_idx  # hoisted once at top of forward (was node_mask.reshape(-1).nonzero())

            # All2Heavy KV heavy-neighbour union (mixed_resolution.a2h_kv_union).
            # Built here, before the host reads below, so the union size can be
            # read in the SAME batched sync as the heavy/light counts. The
            # topology is fixed for the whole forward, so one union serves every
            # All2Heavy block. `_kv_union` stays None when disabled or when the
            # gate below declines, and every downstream use is `is None`-guarded.
            # Skipped outright when the schedule has no All2Heavy block: the
            # union compresses the A2H KV bank and nothing else (H2A's queries
            # ARE all the atoms, so it has no counterpart to compress).
            _ni_heavy = _ni[_he_idx].contiguous() if self._needs_a2h else None
            _kv_union = (
                _a2h_kv_union_indices(
                    _foc_long, _ni_heavy, f_N1, _n_kv,
                )
                if (self._a2h_kv_union and self._needs_a2h) else None
            )
            _n_union_pri = (
                torch.zeros((), dtype=torch.long, device=device)
                if _kv_union is None else _kv_union["n_pri"]
            )
            # One host read for all three counts (was two separate .item()s).
            f_N_heavy, f_N_light, _n_union_pri = (
                int(v) for v in torch.stack(
                    [is_heavy.sum(), is_light.sum(), _n_union_pri]
                ).tolist()
            )
            # Decline compression when the union is not small enough to pay for
            # the extra gather / index_add. Compare against the host atom count
            # (f_N_heavy + f_N_light), NOT f_N1 — f_N1 comes from a nonzero() and
            # is an unbacked SymInt under torch.compile, so a Python comparison
            # on it would graph-break.
            #
            # _n_union_pri == 0 (a pure-water box under lf_O_in_HOH, or a heavy
            # atom with no neighbour inside the cutoff) deliberately PASSES the
            # gate. With no heavy queries these blocks produce nothing that is
            # ever read, yet uncompressed they still run the full O(N) KV
            # projection and throw all of it away; compressing to an empty KV
            # bank skips it outright. This is the case the `:434` explicit
            # reshape exists for.
            if _kv_union is not None and (
                _n_union_pri
                > self._a2h_kv_union_max_ratio * (f_N_heavy + f_N_light)
            ):
                _kv_union = None
            if self._a2h_kv_union_debug:
                _u_state = (
                    "disabled" if not self._a2h_kv_union
                    else "ON" if _kv_union is not None else "OFF (gated)"
                )
                _n_atoms_host = max(f_N_heavy + f_N_light, 1)
                logger.info(
                    f"[MR a2h_kv_union] {_u_state}"
                    f"  heavy={f_N_heavy} light={f_N_light}"
                    + ("  |U_pri|=n/a (not computed)" if not self._a2h_kv_union else
                       f"  |U_pri|={_n_union_pri} ({_n_union_pri / _n_atoms_host:.2%} of atoms)"
                       f"  |U_exp|={0 if _kv_union is None else int(_kv_union['U_exp'].shape[0])}")
                    + f"  N_exp={_n_kv}  max_ratio={self._a2h_kv_union_max_ratio}"
                )
            # Global flat (B*L) indices of heavy / light valid atoms, for the
            # final scatter-back into the padded [B, L, ...] tensor.
            _heavy_flat = _valid_flat.index_select(0, f_heavy_idx)
            _light_flat = _valid_flat.index_select(0, f_light_idx)

            # Full-graph CSR (skipped in MR mode at the top of forward);
            # consumed by edge_deg_embed_dense and the last All2All block.
            # query_idx is built sorted (arange-expand-masked in the radius
            # neighbor search), neighbor_idx never is — pass the hints so the
            # metadata prep skips the _edge_index_is_sorted .item() sync.
            batched_data["metadata"] = prepare_sparse_qk_edge_index_metadata(
                _qi, _ni, f_N1, _n_kv,
                query_sorted=True, neighbor_sorted=False,
            )

            # Heavy-query edge pack: only HEAVY atoms are queries. The QUERY
            # side is restricted to heavy —
            # heavy-local query_idx, heavy-sliced wigner, num_queries=f_N_heavy
            # — while the KV side (wigner_exp / wigner_inv_exp / f_exp_node_pos /
            # f_outcell_index) is inherited FULL from batched_data so heavy
            # queries still attend to ALL atoms. This replaces the old "query
            # all atoms then torch.where-discard light" path: ~2/3 of the
            # node-wise compute (query proj, FFN, output proj, norms) was wasted
            # on light atoms whose results were thrown away.

            # Heavy-local query remap: heavy-local row k <-> global atom
            # f_heavy_idx[k].
            heavy_g2l = torch.full((f_N1,), -1, dtype=torch.long, device=device)
            heavy_g2l[f_heavy_idx] = torch.arange(f_N_heavy, device=device)

            # Heavy query slices (f_heavy_* are otherwise only defined in the
            # mpl path below, which this branch never reaches).
            f_heavy_pos   = f_node_pos.index_select(0, f_heavy_idx)
            f_heavy_an    = f_atomic_numbers.index_select(0, f_heavy_idx)
            f_heavy_batch = f_batch.index_select(0, f_heavy_idx)
            _sys_emb_heavy = (
                None if sys_embedding is None
                else sys_embedding[f_heavy_batch]
            )

            a2h_heavy = dict(batched_data) if self._needs_a2h else None
            if self._needs_a2h:
                a2h_heavy["query_idx"]    = heavy_g2l[_qi[_he_idx]].contiguous()
                a2h_heavy["neighbor_idx"] = _ni_heavy
                a2h_heavy["f_edge_vec"]   = f_edge_vec[_he_idx]
                a2h_heavy["f_dist"]       = f_dist[_he_idx]
                a2h_heavy["f_poly_dist"]  = f_poly_dist[_he_idx]
                a2h_heavy["wigner"]       = batched_data["wigner"].index_select(0, f_heavy_idx)
                a2h_heavy["wigner_inv"]   = batched_data["wigner_inv"].index_select(0, f_heavy_idx)

            # KV-union compression: swap the FULL KV bank inherited from
            # batched_data for the heavy-neighbour union. `a2h_heavy` is a fresh
            # dict, so these reassignments cannot reach batched_data — the
            # All2All block and the All2Light path keep reading the full pack.
            # wigner_exp / wigner_inv_exp / f_exp_node_pos are per-expanded-row,
            # so they slice by U_exp; they
            # are only read by the first-order attn_type (all shipped MR configs
            # are zero-order, so that path is untested here — see the module
            # header note).
            _n_kv_heavy = _n_kv
            if _kv_union is not None:
                _U_exp = _kv_union["U_exp"]
                a2h_heavy["neighbor_idx"]    = _kv_union["neighbor_idx"]
                a2h_heavy["f_outcell_index"] = _kv_union["f_outcell_index"]
                a2h_heavy["wigner_exp"] = batched_data["wigner_exp"].index_select(0, _U_exp)
                a2h_heavy["wigner_inv_exp"] = batched_data["wigner_inv_exp"].index_select(0, _U_exp)
                a2h_heavy["f_exp_node_pos"] = batched_data["f_exp_node_pos"].index_select(0, _U_exp)
                # num_keys for the dkey / dvalue CSR must equal the compressed
                # key/value row count. May stay symbolic: it only feeds
                # bincount(minlength=) and torch.empty(n+1).
                _n_kv_heavy = _U_exp.shape[0]

            if self._needs_a2h:
                a2h_heavy["metadata"] = prepare_sparse_qk_edge_index_metadata(
                    a2h_heavy["query_idx"], a2h_heavy["neighbor_idx"], f_N_heavy, _n_kv_heavy,
                    query_sorted=True, neighbor_sorted=False,  # heavy_g2l[_qi[_he_idx]] stays ascending
                )

            # Not under flash: nothing there reads the full-graph radial basis
            # (the heavy stream is initialised from the a2h pack and there is no
            # A2A block), and this is an [E_full, number_of_basis] tensor -- 400
            # MB fp32 on a 6000-atom water box at r6. Materialising it would
            # hand back a large part of the two-stream memory saving before the
            # first block ran.
            _rbf_full  = None if self._flash else self.rbf(f_dist)
            _rbf_heavy = self.rbf(a2h_heavy["f_dist"]) if self._needs_a2h else None

            # ── Heavy2All (H2A) edge pack ───────────────────────────────────
            # The transpose of All2Heavy: ALL atoms are queries (so the query
            # side is inherited full — wigner / wigner_inv / num_queries=f_N1)
            # and the KV bank is restricted to HEAVY atoms. Edges are the ones
            # whose NEIGHBOR (key) is heavy; because max_neighbors=None makes
            # the radius graph symmetric, that is exactly as many edges as the
            # A2H pack has, so the two differ only in how the node-count terms
            # are assigned (which changes only how the per-node terms are counted).
            #
            # a2h_kv_union is deliberately NOT applied here: its f_outcell_index
            # maps into [0, |U_pri|) while this pack's kv_source has f_N_heavy
            # rows, and there is nothing to compress anyway (the queries ARE all
            # the atoms).
            #
            # f_N_heavy == 0 (pure water under lf_O_in_HOH) leaves an empty KV
            # bank and zero edges. That is handled rather than guarded: the
            # zero-size reshapes are already fixed, the Triton kernels early-return
            # at E == 0, and the block degenerates to a residual + FFN node-wise
            # pass. Skipping the block instead would drop the FFN and diverge from
            # the f_N_heavy == 1 case. It is logged below so it is never silent.
            h2a = None
            _rbf_h2a = None
            _h_exp_l2g = None
            if self._needs_h2a:
                _is_h_exp  = is_heavy[_foc_long]                    # [N_exp] bool
                _h_exp_l2g = _is_h_exp.nonzero(as_tuple=True)[0]    # ascending
                # cumsum(0)-1 rather than full(-1) + scatter(arange): the latter
                # needs arange over an unbacked SymInt, which dynamo cannot
                # handle (see _a2h_kv_union_indices' docstring). Ranks at False
                # positions are garbage but are never read — neighbor_idx is a
                # subset of _h_exp_l2g by construction.
                _h_exp_g2l = _is_h_exp.cumsum(0) - 1
                _hk_idx    = _is_h_exp.index_select(
                    0, _ni_long
                ).nonzero(as_tuple=True)[0]                        # heavy-KEY edge rows
                _n_h_exp   = _h_exp_l2g.shape[0]

                h2a = dict(batched_data)
                # _qi is ascending and _hk_idx is a nonzero() subsequence, so the
                # gather stays non-decreasing -> query_sorted=True is sound.
                h2a["query_idx"]       = _qi[_hk_idx].contiguous()
                h2a["neighbor_idx"]    = _h_exp_g2l.index_select(
                    0, _ni_long.index_select(0, _hk_idx)
                )
                # expanded heavy row -> heavy-LOCAL primary row, in [0, f_N_heavy):
                # key_linear / proj_value / source_embedding are applied to the
                # primary rows (kv_source, f_N_heavy of them) and only then
                # gathered by f_outcell_index.
                h2a["f_outcell_index"] = heavy_g2l.index_select(
                    0, _foc_long.index_select(0, _h_exp_l2g)
                )
                h2a["f_edge_vec"]      = f_edge_vec[_hk_idx]
                h2a["f_dist"]          = f_dist[_hk_idx]
                h2a["f_poly_dist"]     = f_poly_dist[_hk_idx]
                # wigner / wigner_inv inherited FULL: the queries are all f_N1
                # atoms. Only the per-expanded-row tensors are sliced.
                h2a["wigner_exp"]      = batched_data["wigner_exp"].index_select(0, _h_exp_l2g)
                h2a["wigner_inv_exp"]  = batched_data["wigner_inv_exp"].index_select(0, _h_exp_l2g)
                h2a["f_exp_node_pos"]  = batched_data["f_exp_node_pos"].index_select(0, _h_exp_l2g)
                h2a["metadata"]        = prepare_sparse_qk_edge_index_metadata(
                    h2a["query_idx"], h2a["neighbor_idx"], f_N1, _n_h_exp,
                    query_sorted=True, neighbor_sorted=False,
                )
                _rbf_h2a = self.rbf(h2a["f_dist"])
                if f_N_heavy == 0:
                    logger.warning(
                        "[MR interleaved] f_N_heavy == 0: every H2A block has an "
                        "empty KV bank and degenerates to a residual + FFN "
                        "node-wise pass (no attention). Heavy-stream capacity is "
                        "unused on this system."
                    )

            # ── All2Light (A2L) edge pack ───────────────────────────────────
            # LIGHT atoms are the queries and the KV bank is every atom, so
            # this is the mirror image of the A2H pack: the QUERY side is
            # restricted (light-local query_idx, light-sliced wigner,
            # num_queries=f_N_light) while the KV side (f_outcell_index /
            # wigner_exp / wigner_inv_exp / f_exp_node_pos) is inherited FULL.
            #
            # The block itself runs in the LIGHT irreps space, so this pack
            # also feeds the up-projection's edge tensor product -- same edge
            # subset, same CSR, same radial embedding.
            #
            # a2h_kv_union does not apply: the KV is all atoms by definition,
            # and a light-neighbour union covers essentially the whole box.
            a2l = None
            _rbf_light = None
            f_light_pos = f_light_an = f_light_batch = None
            _sys_emb_light_q = _sys_emb_light_kv = None
            if self._needs_a2l:
                with record_function("interleaved-a2l-pack"):
                    # Light-local query remap, the light twin of heavy_g2l.
                    light_g2l = torch.full(
                        (f_N1,), -1, dtype=torch.long, device=device,
                    )
                    light_g2l[f_light_idx] = torch.arange(f_N_light, device=device)

                    f_light_pos   = f_node_pos.index_select(0, f_light_idx)
                    f_light_an    = f_atomic_numbers.index_select(0, f_light_idx)
                    f_light_batch = f_batch.index_select(0, f_light_idx)
                    _sys_emb_light_q = (
                        None if sys_embedding is None
                        else self.light_sys_proj(sys_embedding[f_light_batch])
                    )
                    _sys_emb_light_kv = (
                        None if sys_embedding is None
                        else self.light_sys_proj(sys_embedding[f_batch])
                    )

                    a2l = dict(batched_data)
                    # _qi is ascending and _le_idx is a nonzero() subsequence, so
                    # the gather stays non-decreasing; light_g2l is monotone on
                    # light atoms -> query_sorted=True is sound.
                    a2l["query_idx"]    = light_g2l[_qi[_le_idx]].contiguous()
                    a2l["neighbor_idx"] = _ni[_le_idx].contiguous()
                    a2l["f_edge_vec"]   = f_edge_vec[_le_idx]
                    a2l["f_dist"]       = f_dist[_le_idx]
                    a2l["f_poly_dist"]  = f_poly_dist[_le_idx]
                    a2l["wigner"]       = batched_data["wigner"].index_select(0, f_light_idx)
                    a2l["wigner_inv"]   = batched_data["wigner_inv"].index_select(0, f_light_idx)
                    a2l["metadata"]     = prepare_sparse_qk_edge_index_metadata(
                        a2l["query_idx"], a2l["neighbor_idx"], f_N_light, _n_kv,
                        query_sorted=True, neighbor_sorted=False,
                    )
                    # The LIGHT rbf (light_rbf), not self.rbf: the light stream has
                    # its own radial basis, and both the light TransBlock and the
                    # up-projection's radial trunk are sized for it.
                    _rbf_light = self.light_rbf(a2l["f_dist"])
                    if f_N_light == 0:
                        logger.warning(
                            "[MR interleaved] f_N_light == 0: every A2L block has an "
                            "empty query set and writes nothing. The light stream is "
                            "unused on this system."
                        )

            # ── interleaved_flash: the concatenated KV row space ─────────────
            # Both flash blocks need a KV bank assembled from two row groups
            # (H / L for A2H-flash, down(H) / L for A2L-flash). Scattering
            # those into an [N_all, ...] buffer would cost exactly the memory
            # the two-stream layout saves, so the bank is the CONCATENATION
            # [heavy rows ; light rows] and f_outcell_index is relabelled
            # through a fixed permutation built once per forward.
            #
            # This is a pure RELABELLING of today's all-atom KV bank: the same
            # rows, in a different order. A2H-flash's KV membership is
            # therefore identical to A2H's. wigner_exp / wigner_inv_exp /
            # f_exp_node_pos are indexed by EXPANDED row and are untouched --
            # only the primary-row labels move. The attention already receives a
            # pre-merged [heavy, upgraded-light]
            # KV with an extended f_outcell_index.
            a2h_flash = a2l_flash = None
            _flash_kv_an = _flash_kv_sys_l = None
            _flash_kv_an_h = _flash_kv_sys_h = None
            _flash_u_h = _flash_u_l = None
            if self._flash:
                with record_function("flash-kv-perm"):
                    _flash_perm = torch.empty(
                        f_N1, dtype=torch.long, device=device,
                    )
                    _flash_perm[f_heavy_idx] = torch.arange(
                        f_N_heavy, device=device,
                    )
                    _flash_perm[f_light_idx] = f_N_heavy + torch.arange(
                        f_N_light, device=device,
                    )
                    _foc_flash = _flash_perm[_foc_long]  # expanded row -> KV row

                    # Both packs are the existing ones with ONLY
                    # f_outcell_index replaced. dict(...) so the reassignment
                    # cannot reach batched_data or the original packs.
                    a2h_flash = dict(a2h_heavy)
                    a2l_flash = dict(a2l)
                    # A2L-flash's KV bank is every atom by definition, so it
                    # always takes the uncompressed relabelling.
                    a2l_flash["f_outcell_index"] = _foc_flash

                    # KV-side per-row tensors in the concatenated order. The
                    # sys embedding must be carried at the BANK's channel
                    # width, which differs between the two blocks -- hence two
                    # tensors, not one. The heavy bank keeps
                    # the raw C_H embedding; the light bank goes through
                    # light_sys_proj exactly as the light QUERY side does.
                    _flash_kv_an = torch.cat([f_heavy_an, f_light_an], dim=0)
                    _flash_kv_batch = torch.cat(
                        [f_heavy_batch, f_light_batch], dim=0,
                    )
                    if sys_embedding is not None:
                        _flash_kv_sys_l = self.light_sys_proj(
                            sys_embedding[_flash_kv_batch]
                        )

                # ── A2H-flash's KV bank: the two remaps, composed ───────────
                # Without a2h_kv_union the A2H-flash bank IS the full
                # [heavy ; light] concatenation, so it takes _foc_flash and
                # the full per-row tensors.
                #
                # With it, the bank is compressed to the heavy-neighbour union
                # U_pri. _a2h_kv_union_indices hands back rows in ASCENDING
                # GLOBAL ATOM ID, which is not the flash layout, so the
                # compressed bank is reordered into [heavy-part ; light-part]
                # (the two halves are gathered from the two streams, which is
                # what lets the concat itself shrink) and the union's own
                # expanded->primary map is relabelled through that reordering.
                # Both remaps are pure relabellings of the same row SET, so
                # composing them is exact -- the union's losslessness argument
                # (see _a2h_kv_union_indices) is untouched.
                if _kv_union is None:
                    a2h_flash["f_outcell_index"] = _foc_flash
                    _flash_kv_an_h = _flash_kv_an
                    _flash_kv_sys_h = (
                        None if sys_embedding is None
                        else sys_embedding[_flash_kv_batch]
                    )
                else:
                    with record_function("flash-kv-union"):
                        _U_pri = _kv_union["U_pri"]
                        _u_is_heavy = is_heavy.index_select(0, _U_pri)
                        # Rank within [heavy-part ; light-part]. cumsum, not
                        # arange over a .sum() -- the union size is a nonzero()
                        # output and reading it on the host would be a sync
                        # (and an unbacked SymInt under torch.compile).
                        _u_n_heavy = _u_is_heavy.sum()
                        _u_rank = torch.where(
                            _u_is_heavy,
                            _u_is_heavy.cumsum(0) - 1,
                            _u_n_heavy + ((~_u_is_heavy).cumsum(0) - 1),
                        )
                        a2h_flash["f_outcell_index"] = _u_rank.index_select(
                            0, _kv_union["f_outcell_index"],
                        )
                        # Row selectors into the two LIVE streams. These index
                        # H / L directly (heavy-local / light-local), so the
                        # block loop never has to build the full concatenation.
                        _u_h_pos = _u_is_heavy.nonzero(as_tuple=True)[0]
                        _u_l_pos = (~_u_is_heavy).nonzero(as_tuple=True)[0]
                        _u_pri_h = _U_pri.index_select(0, _u_h_pos)
                        _u_pri_l = _U_pri.index_select(0, _u_l_pos)
                        _flash_u_h = heavy_g2l.index_select(0, _u_pri_h)
                        _flash_u_l = light_g2l.index_select(0, _u_pri_l)
                        # Per-row atomic numbers / sys embedding in the SAME
                        # [heavy-part ; light-part] order as the bank.
                        _u_pri_cat = torch.cat([_u_pri_h, _u_pri_l], dim=0)
                        _flash_kv_an_h = f_atomic_numbers.index_select(
                            0, _u_pri_cat,
                        )
                        _flash_kv_sys_h = (
                            None if sys_embedding is None
                            else sys_embedding[
                                f_batch.index_select(0, _u_pri_cat)
                            ]
                        )

            # All2Heavy KV-side per-atom tensors, hoisted out of the block loop
            # (only the irreps themselves change between blocks). When compressed
            # these are the heavy-neighbour-union rows, aligned with a2h_heavy's
            # remapped f_outcell_index; otherwise the full node stream.
            _sys_emb = (
                None if sys_embedding is None else sys_embedding[f_batch]
            )
            if _kv_union is None:
                _kv_an_U   = f_atomic_numbers
                _sys_emb_U = _sys_emb
            else:
                _U_pri = _kv_union["U_pri"]
                _kv_an_U = f_atomic_numbers.index_select(0, _U_pri)
                _sys_emb_U = (
                    None if sys_embedding is None
                    else sys_embedding[f_batch.index_select(0, _U_pri)]
                )

            # torch.compile: heavy-query dims are data-dependent; mark dynamic
            # (mirror the mpl path) so a single graph serves any heavy count.
            # U_pri / U_exp size the compressed KV GEMMs and index the
            # index_select inside the compiled block loop, so they must be marked
            # too or every MD step recompiles as the union drifts. Marked here,
            # before the block loop consumes them.
            # E_h2a and N_heavy_exp are two further independent data-dependent
            # sizes. _rbf_h2a in particular sizes the whole fc_easy RadialProfile
            # GEMM, and a missing mark here recompiles on every MD step as the
            # counts drift — a silent failure that looks like a slow model.
            _mr_subgraph_dyn, _mr_atom_dyn = _get_mr_mark_dynamic_fns()
            if self._needs_a2h:
                _mr_subgraph_dyn(a2h_heavy)
            if self._needs_h2a:
                _mr_subgraph_dyn(h2a)
            if self._needs_a2l:
                _mr_subgraph_dyn(a2l)
            # The flash packs carry the relabelled f_outcell_index, and the
            # concatenated KV rows are a THIRD data-dependent size (N_H + N_L)
            # independent of N_H and N_L separately. A missing mark here
            # recompiles on every MD step as the counts drift -- which reads as
            # a slow model, not as a bug.
            if self._flash:
                _mr_subgraph_dyn(a2h_flash)
                _mr_subgraph_dyn(a2l_flash)
            _mr_atom_dyn(
                f_heavy_pos, f_heavy_an, f_heavy_batch, f_heavy_idx,
                *(() if not self._needs_a2h else (_rbf_heavy,)),
                *(() if not self._needs_h2a else (_rbf_h2a, _h_exp_l2g)),
                *(() if not self._needs_a2l else (
                    _rbf_light, f_light_pos, f_light_an, f_light_batch,
                    f_light_idx,
                )),
                *(() if not self._needs_a2l or _sys_emb_light_q is None else (
                    _sys_emb_light_q,
                )),
                *(() if _kv_union is None else (
                    _kv_union["U_pri"], _kv_union["U_exp"], _kv_an_U,
                )),
                *(() if _sys_emb_U is None or _kv_union is None else (_sys_emb_U,)),
                *(() if not self._flash else (
                    _flash_perm, _flash_kv_an, _flash_kv_batch,
                    _flash_kv_an_h,
                )),
                # |U_pri ∩ heavy| and |U_pri ∩ light| are two FURTHER
                # data-dependent sizes, independent of |U_pri| itself: they
                # size the two halves of the compressed A2H-flash bank and the
                # index_selects the block loop runs on every layer.
                *(() if not self._flash or _flash_u_h is None else (
                    _flash_u_h, _flash_u_l,
                )),
                *(() if not self._flash or _flash_kv_sys_h is None else (
                    _flash_kv_sys_h, _flash_kv_sys_l,
                )),
            )

            # ==========================================================
            # cross_mode="interleaved_flash": TWO persistent node states.
            # Self-contained -- own init, own block loop, own tail; it
            # returns here and never reaches the single-stream code below.
            # ==========================================================
            if self._flash:
                _Lh = (self.lmax + 1) ** 2
                _Ll = (self.lmax_light + 1) ** 2

                # ---- two-stream initialisation -------------------------
                # Heavy init is restricted to HEAVY QUERIES via the a2h
                # pack. Light atoms still SHAPE it -- they are on the
                # KV/neighbour side of that pack -- they just stop being
                # queries of it. Light init uses the a2l pack for the
                # mirror-image reason: heavy neighbours contribute to the
                # light init.
                #
                # Both packs have a query set that is a strict subset of an
                # all-atom KV bank, so target_embedding (query-local) and
                # source_embedding (KV-row) need DIFFERENT atomic-number
                # arrays -- hence kv_atomic_numbers, which every other
                # caller leaves at None.
                with record_function("flash-heavy-node-init"):
                    f_heavy_irreps = self.edge_deg_embed_dense(
                        node_input=f_atom_embedding.index_select(0, f_heavy_idx),
                        node_pos=f_heavy_pos,
                        edge_dis=a2h_flash["f_dist"],
                        atomic_numbers=f_heavy_an,
                        # The COMPRESSED array when a2h_kv_union is on: this
                        # pack's f_outcell_index indexes the union bank, and
                        # source_embedding is gathered through it. Restricting
                        # the init the same way is lossless for the same reason
                        # the block is -- heavy-query edges never name a row
                        # outside U_exp.
                        kv_atomic_numbers=_flash_kv_an_h,
                        edge_vec=a2h_flash["f_edge_vec"],
                        batched_data=a2h_flash,
                        attn_mask=None,
                        poly_dist=a2h_flash["f_poly_dist"],
                        edge_scalars=_rbf_heavy,
                    )
                with record_function("flash-light-node-init"):
                    f_light_irreps = self.light_edge_deg_embed(
                        node_input=self.light_default_node_embedding(f_light_an),
                        node_pos=f_light_pos,
                        edge_dis=a2l_flash["f_dist"],
                        atomic_numbers=f_light_an,
                        kv_atomic_numbers=_flash_kv_an,
                        edge_vec=a2l_flash["f_edge_vec"],
                        batched_data=a2l_flash,
                        attn_mask=None,
                        poly_dist=a2l_flash["f_poly_dist"],
                        edge_scalars=_rbf_light,
                    )

                # ---- block loop ----------------------------------------
                for i, blk in enumerate(self.blocks):
                    _role = self._block_schedule[i]
                    if _role == "A2H-FLASH":
                        # Heavy queries over KV = [H ; up(L)]. The light
                        # state is passed through by NOT being written --
                        # no gather, no scatter, no outer residual, so
                        # "leave the light stream alone" is the zero
                        # update rather than (I + up.down).
                        _slot = self._a2h_flash_slot[i]
                        # a2h_kv_union: build the bank from only the rows
                        # heavy queries actually read. _flash_u_h / _flash_u_l
                        # are None when the union is off or gated out, in
                        # which case both halves are the whole stream.
                        #
                        # The light selection is applied BEFORE the lift in
                        # the linear regime (a per-node map, so this is exact
                        # AND removes the discarded rows from the projection
                        # itself) and AFTER it in the tp regime, whose edge
                        # aggregation is sized by the full light query set.
                        with record_function(f"flash-a2h-upproj {i}"):
                            _l_src = (
                                f_light_irreps
                                if _flash_u_l is None
                                or not self._flash_up_proj_linear
                                else f_light_irreps.index_select(0, _flash_u_l)
                            )
                            _kv_l = self.flash_up_proj[_slot](
                                _l_src,
                                # Ignored in the "linear" regime; the "tp"
                                # regime aggregates over the light-query
                                # edges, exactly as A2L's lift does.
                                query_idx=a2l_flash["query_idx"],
                                edge_vec=a2l_flash["f_edge_vec"],
                                poly_dist=a2l_flash["f_poly_dist"],
                                edge_scalars=_rbf_light,
                                metadata=a2l_flash["metadata"],
                                num_queries=f_N_light,
                            )                       # [N_L or |U_l|, Lh, C_H]
                            if (
                                _flash_u_l is not None
                                and not self._flash_up_proj_linear
                            ):
                                _kv_l = _kv_l.index_select(0, _flash_u_l)
                        _kv_h = (
                            f_heavy_irreps if _flash_u_h is None
                            else f_heavy_irreps.index_select(0, _flash_u_h)
                        )
                        _kv = torch.cat(
                            [_kv_h, _kv_l.to(_kv_h.dtype)], dim=0,
                        )                           # [|bank|, Lh, C_H]
                        with record_function(f"flash-a2h-tranblock {i}"):
                            f_heavy_irreps, _ = blk(
                                node_pos=f_heavy_pos,
                                node_irreps=f_heavy_irreps,
                                sys_node_embedding=_sys_emb_heavy,
                                edge_dis=a2h_flash["f_dist"],
                                edge_vec=a2h_flash["f_edge_vec"],
                                attn_weight=_rbf_heavy,
                                atomic_numbers=f_heavy_an,
                                attn_mask=None,
                                poly_dist=a2h_flash["f_poly_dist"],
                                batch=f_heavy_batch,
                                batched_data=a2h_flash,
                                kv_source=_kv,
                                kv_atomic_numbers=_flash_kv_an_h,
                                kv_sys_embedding=_flash_kv_sys_h,
                            )
                    else:  # "A2L-FLASH"
                        # Light queries over KV = [down(H) ; L], in the
                        # LIGHT space. Including L is what gives light-light
                        # interaction and what keeps the N_H == 0 case
                        # non-degenerate.
                        _slot = self._a2l_flash_slot[i]
                        with record_function(f"flash-a2l-downproj {i}"):
                            _kv_h = self.flash_down_proj[_slot](
                                f_heavy_irreps[:, :_Ll]
                            )                       # [N_H, Ll, C_L]
                        _kv = torch.cat(
                            [_kv_h, f_light_irreps.to(_kv_h.dtype)], dim=0,
                        )                           # [N_H+N_L, Ll, C_L]
                        with record_function(f"flash-a2l-tranblock {i}"):
                            f_light_irreps, _ = self.light_blocks[i](
                                node_pos=f_light_pos,
                                node_irreps=f_light_irreps,
                                sys_node_embedding=_sys_emb_light_q,
                                edge_dis=a2l_flash["f_dist"],
                                edge_vec=a2l_flash["f_edge_vec"],
                                attn_weight=_rbf_light,
                                atomic_numbers=f_light_an,
                                attn_mask=None,
                                poly_dist=a2l_flash["f_poly_dist"],
                                batch=f_light_batch,
                                batched_data=a2l_flash,
                                kv_source=_kv,
                                kv_atomic_numbers=_flash_kv_an,
                                kv_sys_embedding=_flash_kv_sys_l,
                            )

                # ---- tail ----------------------------------------------
                # The heavy-shaped return tensor is unchanged in shape and
                # meaning, with LIGHT ROWS LEFT AT ZERO (no lift, no cost).
                # The light stream is published alongside instead of being
                # padded up into it -- padding would hand back the entire
                # memory saving at the tail and re-introduce the "which
                # lift" question this layout just removed.
                #
                # CONSEQUENCE: anything that consumes the full-atom irreps
                # tensor as if it covered every atom is UNSUPPORTED under
                # flash and must raise rather than read zeros. A consumer that
                # is willing to be TWO-HEADED is fine, and both read-outs are:
                # energy_head="distinct" (enforced here and in head.py) and,
                # with AutoGradForce=False, head.py's force_heads_light,
                # which reads node_irreps_lightBxN[:, :, 1:4] for the light
                # atoms and selects on is_light_2d. Note the light stream is
                # live at the tail no matter which stream the LAST block
                # wrote: A2H-flash passes L through unchanged.
                with record_function("flash-tail"):
                    _Cout = self._node_scalar_dim
                    node_irreps = torch.zeros(
                        (B, L, _Lh, _Cout), device=device,
                    )
                    node_irreps.reshape(B * L, _Lh, _Cout)[_heavy_flat] = (
                        self.norm_final(f_heavy_irreps)
                    )
                    node_irreps_light = torch.zeros(
                        (B, L, _Ll, self.light_scalar_dim), device=device,
                    )
                    node_irreps_light.reshape(
                        B * L, _Ll, self.light_scalar_dim
                    )[_light_flat] = self.light_norm_final(f_light_irreps)

                    # Published on batched_data, which E2FormerBackbone
                    # already returns as "data" -- so no signature changes
                    # anywhere on the way to head.py.
                    _is_light_bl = torch.zeros(
                        (B, L), dtype=torch.bool, device=device,
                    )
                    _is_light_bl.reshape(-1)[_light_flat] = True
                    batched_data["node_irreps_lightBxN"] = node_irreps_light
                    batched_data["is_light_2d"] = _is_light_bl
                return node_irreps[:, :, 0], node_irreps[:, :, 1:4], node_irreps

            # Initialise ALL atoms over the full graph (baseline init).
            with record_function("aligned_node_initialisation"):
                f_node_irreps = self.edge_deg_embed_dense(
                    node_input=f_atom_embedding,
                    node_pos=f_node_pos,
                    edge_dis=f_dist,
                    atomic_numbers=f_atomic_numbers,
                    edge_vec=f_edge_vec,
                    batched_data=batched_data,
                    attn_mask=None,
                    poly_dist=f_poly_dist,
                    edge_scalars=_rbf_full,
                )

            # cat_layer_output: keep a reference to the selected blocks' outputs
            # for the post-loop per-layer norm + channel concat (v1 tail only).
            _layer_outputs = [] if self.cat_layer_output else None

            for i, blk in enumerate(self.blocks):
                # Per-block role, precomputed in __init__ from the configured
                # cross_mode_list (X2Y = KV is X, query is Y).
                _role = self._block_schedule[i]
                if _role == "A2A":
                    with record_function(f"aligned-all2all-tranblock {i}"):
                        f_node_irreps, _ = blk(
                            node_pos=f_node_pos,
                            node_irreps=f_node_irreps,
                            sys_node_embedding=_sys_emb,
                            edge_dis=batched_data["f_dist"],
                            edge_vec=batched_data["f_edge_vec"],
                            attn_weight=_rbf_full,
                            atomic_numbers=f_atomic_numbers,
                            attn_mask=None,
                            poly_dist=batched_data["f_poly_dist"],
                            batch=f_batch,
                            batched_data=batched_data,
                        )
                elif _role == "H2A":
                    # Heavy2All: ALL atoms are queries over a HEAVY-ONLY KV bank.
                    # The transpose of All2Heavy — light atoms are updated here,
                    # inside the loop, instead of only in the post-loop tail, and
                    # thereafter carry an evolving full irreps state (so the
                    # "light rows stay at their edge_deg init" invariant that the
                    # All2Heavy comment below describes no longer holds once one
                    # of these has run). Output replaces the whole node stream —
                    # no index_copy.
                    #
                    # kv_source is read from f_node_irreps BEFORE the update: the
                    # binding below makes that ordering explicit rather than
                    # leaning on Python's argument-evaluation order.
                    _kv_heavy = f_node_irreps.index_select(0, f_heavy_idx)
                    with record_function(f"interleaved-heavy2all-tranblock {i}"):
                        f_node_irreps, _ = blk(
                            node_pos=f_node_pos,
                            node_irreps=f_node_irreps,
                            sys_node_embedding=_sys_emb,
                            # Edge geometry is passed explicitly, NOT read from
                            # batched_data: edge_vec feeds the spherical harmonics
                            # and poly_dist scales alpha, both per-edge over the
                            # h2a edge subset. Inheriting the full-graph versions
                            # would be a shape error — or, on an all-heavy system
                            # where E_full == E_h2a, silently wrong physics.
                            edge_dis=h2a["f_dist"],
                            edge_vec=h2a["f_edge_vec"],
                            attn_weight=_rbf_h2a,
                            atomic_numbers=f_atomic_numbers,
                            attn_mask=None,
                            poly_dist=h2a["f_poly_dist"],
                            batch=f_batch,
                            batched_data=h2a,
                            kv_source=_kv_heavy,
                            kv_atomic_numbers=f_heavy_an,
                            kv_sys_embedding=_sys_emb_heavy,
                        )
                elif _role == "A2L":
                    # All2Light, IN THE LIGHT SPACE. Light atoms are
                    # the queries and every atom is KV, but -- unlike H2A --
                    # the block runs at the light irreps/channel count, so a
                    # light atom is still light after it. The round trip is
                    #   down: SO3 linear C_H -> C_L, truncated to lmax_light
                    #   attn: light_blocks[i], light queries over all-atom KV
                    #   up:   edge tensor product back to the heavy irreps
                    # and only light rows are written, mirroring how A2H writes
                    # only heavy rows.
                    #
                    # The up-projection cannot be a linear map: SO3_Linear is a
                    # per-order channel mixer, so it would return light atoms
                    # whose l > lmax_light is identically zero at every layer.
                    # See A2LUpProjection.
                    #
                    # NOT the "v2" tail's All2Light block: that one runs once,
                    # after the loop, on a reduced scalar summary.
                    _slot = self._a2l_slot[i]
                    with record_function(f"interleaved-a2l-downproj {i}"):
                        _kv_light = self.a2l_down_proj[_slot](
                            f_node_irreps[:, :(self.lmax_light + 1) ** 2]
                        )                                    # [N_all, Ll, C_L]
                    with record_function(f"interleaved-all2light-tranblock {i}"):
                        _out_light, _ = self.light_blocks[i](
                            node_pos=f_light_pos,
                            node_irreps=_kv_light.index_select(0, f_light_idx),
                            sys_node_embedding=_sys_emb_light_q,
                            # Explicit edge geometry over the light-query edge
                            # subset, for the same reason as H2A above.
                            edge_dis=a2l["f_dist"],
                            edge_vec=a2l["f_edge_vec"],
                            attn_weight=_rbf_light,
                            atomic_numbers=f_light_an,
                            attn_mask=None,
                            poly_dist=a2l["f_poly_dist"],
                            batch=f_light_batch,
                            batched_data=a2l,
                            kv_source=_kv_light,
                            kv_atomic_numbers=f_atomic_numbers,
                            kv_sys_embedding=_sys_emb_light_kv,
                        )
                    with record_function(f"interleaved-a2l-upproj {i}"):
                        _delta_light = self.a2l_up_proj[_slot](
                            x_light=_out_light,
                            query_idx=a2l["query_idx"],
                            edge_vec=a2l["f_edge_vec"],
                            poly_dist=a2l["f_poly_dist"],
                            edge_scalars=_rbf_light,
                            metadata=a2l["metadata"],
                            num_queries=f_N_light,
                        )
                    f_node_irreps = f_node_irreps.index_copy(
                        0, f_light_idx,
                        f_node_irreps.index_select(0, f_light_idx)
                        + _delta_light.to(f_node_irreps.dtype),
                    )
                else:
                    # All2Heavy: heavy atoms query the full-node KV. kv_source is
                    # the CURRENT f_node_irreps read BEFORE the update, so heavy
                    # queries attend to current-heavy + frozen-init-light (matches
                    # the mpl merged-kv semantics). Light rows are never written,
                    # so they stay at their edge_deg init — no torch.where revert.
                    # With a2h_kv_union the KV is sliced to the heavy-neighbour
                    # union; the slice must stay INSIDE the loop and before the
                    # index_copy below, since heavy rows advance per block.
                    _kv_full = (
                        f_node_irreps if _kv_union is None
                        else f_node_irreps.index_select(0, _kv_union["U_pri"])
                    )
                    with record_function(f"aligned-all2heavy-tranblock {i}"):
                        _out_heavy, _ = blk(
                            node_pos=f_heavy_pos,
                            node_irreps=f_node_irreps.index_select(0, f_heavy_idx),
                            sys_node_embedding=_sys_emb_heavy,
                            edge_dis=a2h_heavy["f_dist"],
                            edge_vec=a2h_heavy["f_edge_vec"],
                            attn_weight=_rbf_heavy,
                            atomic_numbers=f_heavy_an,
                            attn_mask=None,
                            poly_dist=a2h_heavy["f_poly_dist"],
                            batch=f_heavy_batch,
                            batched_data=a2h_heavy,
                            kv_source=_kv_full,
                            kv_atomic_numbers=_kv_an_U,
                            # kv == the full node stream (or its heavy-neighbour
                            # union), so it must carry the per-atom sys embedding
                            # on l=0 to match the old self-attention (query==kv)
                            # path.
                            kv_sys_embedding=_sys_emb_U,
                        )
                    f_node_irreps = f_node_irreps.index_copy(
                        0, f_heavy_idx, _out_heavy,
                    )

                if self.cat_layer_output and i in self._cat_layer_slots:
                    # After the role branch, so this is the block's OUTPUT for
                    # every role (A2H/A2L write only their own rows and leave the
                    # rest carrying the incoming state).
                    _layer_outputs.append(f_node_irreps)

            if self._block_tail == "v1":
                # aligned: final norm + scatter-back (mirror the non-MR tail).
                if self.cat_layer_output:
                    # One norm per slot, then concat along the channel axis:
                    # [N, Lh, C] * n_slots -> [N, Lh, n_slots * C]. norm_final is
                    # bypassed but kept for checkpoint compatibility.
                    f_node_irreps = torch.cat(
                        [
                            self.cat_layer_norms[_s](_t)
                            for _s, _t in enumerate(_layer_outputs)
                        ],
                        dim=-1,
                    ).contiguous()
                else:
                    f_node_irreps = self.norm_final(f_node_irreps).contiguous()
                _Lh = (self.lmax + 1) ** 2
                _Cout = self._out_scalar_dim
                node_irreps = torch.zeros(
                    (B, L, _Lh, _Cout),
                    device=device,
                )
                # Integer scatter (was boolean node_irreps[node_mask] = ...,
                # which syncs); _valid_flat are the flat B*L indices of valid
                # atoms, in the same ascending order as the boolean mask.
                node_irreps.reshape(B * L, _Lh, _Cout)[
                    _valid_flat
                ] = f_node_irreps
                return node_irreps[:, :, 0], node_irreps[:, :, 1:4], node_irreps
        with record_function("node_initiliation"):
            if not self._mr_enabled:
                f_node_irreps = self.edge_deg_embed_dense(
                    node_input=f_atom_embedding,
                    node_pos=f_node_pos,
                    edge_dis=f_dist,
                    atomic_numbers=f_atomic_numbers,
                    edge_vec=f_edge_vec,
                    batched_data=batched_data,
                    attn_mask=f_attn_mask,
                    poly_dist=f_poly_dist,
                    edge_scalars=f_dist_embedding,
                )


        # Non-mixed-resolution path (E2Former without the MR dispatch).
        # The mixed-resolution branch above always returns, so this is the
        # only route that reaches here.
        _shared_kv = f_node_irreps if self.layerwise_shared_KV_cache else None
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
                    batched_data=batched_data,
                    kv_source=_shared_kv,
                    kv_atomic_numbers=f_atomic_numbers if _shared_kv is not None else None,
                )

        # POC: restrict H atoms to lmax=0 (zero out l>0 irreps for H)
        if self.zero_H_high_order:
            is_H = (f_atomic_numbers == 1)
            if is_H.any():
                f_node_irreps = f_node_irreps.clone()
                f_node_irreps[is_H, 1:, :] = 0.0

        # Cluster / long-range path (only for original non-MR mode)
        if "node" in self.with_cluster:
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
                        batched_data=batched_data,
                    )
            f_node_irreps = self.norm_cluster(f_node_irreps)
            f_node_irreps = self.final_linear(torch.cat([f_node_irreps_short,f_node_irreps],dim=-1))

        elif self.with_cluster!="False":
            f_node_irreps = self.long_range(batched_data,f_node_irreps,cluster_neighbor_info=cluster_neighbor_info)
        else:
            f_node_irreps = self.norm_final(f_node_irreps)

        f_node_irreps = f_node_irreps.contiguous()
        node_irreps = torch.zeros(
            (B, L, (self.lmax+1)**2, self._node_scalar_dim), device=device
        )
        node_irreps[node_mask] = f_node_irreps  # the part of order 0

        node_attr = node_irreps[:,:,0]
        node_vec = node_irreps[:,:,1:4]

        # # output shape: [B, L, (lmax+1)^2, hidden]
        # out_dim1 = (self.lmax + 1) ** 2
        # out_dim2 = self._node_scalar_dim
        
        # node_irreps_flat = torch.zeros(
        #     (B * L, out_dim1 * out_dim2), 
        #     device=device, 
        #     dtype=f_node_irreps.dtype
        # )
        
        # mask_flat = node_mask.view(-1).unsqueeze(-1)
        # source_flat = f_node_irreps.view(f_node_irreps.shape[0], -1)
        # node_irreps_flat.masked_scatter_(mask_flat, source_flat)
        # node_irreps = node_irreps_flat.view(B, L, out_dim1, out_dim2)
        # node_attr = node_irreps[:,:,0]
        # node_vec = node_irreps[:,:,1:4]
        
        return node_attr, node_vec, node_irreps


# coeffs length is lmax+1, help normalize or refactor of spherical hamonics
def get_powers(vec,coeffs,lmax):
    out_powers = [
        coeffs[0] * torch.ones_like(vec.narrow(-1, 0, 1).unsqueeze(dim=-1))
    ]
    # Y is pos. Precompute spherical harmonics for all orders
    for i in range(1, lmax + 1):
        out_powers.append(
            coeffs[i]
            * e3nn.o3.spherical_harmonics(
                i, vec, normalize=False, normalization="integral"
            ).unsqueeze(-1)
        )

    return out_powers
