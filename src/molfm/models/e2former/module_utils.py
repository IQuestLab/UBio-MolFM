# -*- coding: utf-8 -*-
import logging
import math
from multiprocessing.dummy import current_process
from typing import List
import warnings

import e3nn
import numpy as np
import scipy.special as sp
import torch


from torch import nn
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch.profiler import record_function


from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.o3 import FromS2Grid, ToS2Grid


from molfm.models.e2former.maceblocks import NonLinearDipoleReadoutBlock
from .triton_dr.triton_sparse_qk_autograd import prepare_sparse_qk_edge_index_metadata


from .wigner6j.base_tensor_product import Simple_TensorProduct

from loguru import logger
import torch
from torch import logical_not, nn

import torch, torch.nn as nn, re
from torch.optim import AdamW, Adam

import torch.nn as nn
from typing import Optional

def build_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    custom_factors: Optional[list[tuple[str, float]]] = None,
):
    """
    custom_factors: 
        e.g. [("svp", 10.0), ("tzvpd", 30.0)]

    """
    model = model.module if hasattr(model, "module") else model

    no_decay_types = (
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        EquivariantRMSNormArraySphericalHarmonicsV2_BL,
    )
    no_decay_name_substr = (
        "bias",
        "embedding",
        "pos_embed",
        "relative_position_bias",
    )

    def is_no_decay(name, module, param):
        if param.ndim == 1:  
            return True
        if isinstance(module, no_decay_types):
            return True
        if any(k in name for k in no_decay_name_substr):
            return True
        return False

    # key: ("decay" or "no_decay", lr_mult) -> [params...]
    groups: dict[tuple[str, float], list] = {}

    for mod_name, module in model.named_modules():
        for p_name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue

            full = f"{mod_name}.{p_name}" if mod_name else p_name
            nd = is_no_decay(full, module, p)

            
            lr_mult = 1.0
            
            if custom_factors is not None:
                for key_substr, factor in custom_factors:
                    if key_substr in full:
                        lr_mult = float(factor)
                        break

            kind = "no_decay" if nd else "decay"
            gkey = (kind, lr_mult)
            groups.setdefault(gkey, []).append(p)

    
    param_groups = []
    for (kind, lr_mult), params in groups.items():
        if not params:
            continue
        wd = 0.0 if kind == "no_decay" else weight_decay
        param_groups.append(
            {
                "params": params,
                "lr": base_lr * lr_mult,
                "weight_decay": wd,
            }
        )

    return param_groups


def scaled_sigmoid(x, N, k=1.0):
    """
    Map the input x (an integer between 0 and N) to the interval [0, 1], approaching 0 near 0 and approaching 1 near N, similar to a sigmoid function. 
    Parameters:
        x: Tensor, input value, should be within the range [0, N]
        N: float or int, maximum value
        k: float, controls the steepness of the curve, the larger the k, the steeper the curve (default 1.0) 
    Return:
        Tensor with the same shape as x, and values within the range [0, 1]
    """
    
    x_norm = x / N  
    
    
    return torch.sigmoid(k * (x_norm - 0.5))  


class sph_fromxyz(torch.nn.Module):
    def __init__(self, lmax = 2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lmax = lmax

        rt5, rt15 = math.sqrt(5.0), math.sqrt(15.0)
        W = torch.zeros(5, 9)
        # sh_2_0
        W[0, 2] =  rt15             # xz
        # sh_2_1
        W[1, 1] =  rt15             # xy
        # sh_2_2
        W[2, 4] =  rt5              # + y^2
        W[2, 0] = -0.5 * rt5        # - 0.5*x^2
        W[2, 8] = -0.5 * rt5        # - 0.5*z^2
        # sh_2_3
        W[3, 5] =  rt15             # yz
        # sh_2_4
        W[4, 8] =  0.5 * rt15       # + 0.5*z^2
        W[4, 0] = -0.5 * rt15       # - 0.5*x^2
        self.W = torch.nn.Parameter(W.t(),requires_grad=False)
    def forward(self,vec):
        """
        vec: (..., 3)   x,y,z
        return: (..., 5)   [sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4]
        """
        assert vec.shape[-1] == 3
        if self.lmax == 0:
            return 
        elif self.lmax == 1:
            return torch.cat([torch.ones_like(vec[...,:1]), math.sqrt(3)*vec],dim = -1)
        elif self.lmax == 2:
            
            
            outer = vec[..., :, None] * vec[..., None, :]        # (...,3,3)
            feat  = outer.reshape(*vec.shape[:-1], 9)            # (...,9)

            sh_l2 = feat @ self.W
            return torch.cat([torch.ones_like(vec[...,:1]), math.sqrt(3)*vec,sh_l2],dim = -1)
        else:
            raise ValueError


# follow fairchem 2.4.0 and e3nn 0.4.0
# quick fix: for some special case e.g. [0,0,0],[0,1,0],[0,x,0]
@torch.compiler.disable
def init_edge_rot_euler_angles(edge_distance_vec,training = True,ts = 0.00001):
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.norm(edge_vec_0,dim = 1) #torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    # Make sure the atoms are far enough apart
    # assert torch.min(edge_vec_0_distance) < 0.0001
    mask = edge_vec_0_distance < ts
    if len(edge_vec_0_distance) > 0 and torch.min(edge_vec_0_distance) < ts:
        logger.error(f"Error edge_vec_0_distance: {torch.min(edge_vec_0_distance)} - > reset to 1")
        edge_vec_0_distance = torch.where(
            edge_vec_0_distance < ts,
            1,
            edge_vec_0_distance
        )
    #     mask = 
    # else:
    #     # are we standing at the north pole
    #     mask = xyz[:, 1].abs().isclose(xyz.new_ones(1))

    # make unit vectors
    xyz = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))
    mask = mask + xyz[:, 1].abs().isclose(xyz.new_ones(1))

    # compute alpha and beta

    # latitude (beta)
    beta = xyz.new_zeros(xyz.shape[0])
    beta[~mask] = torch.acos(xyz[~mask, 1])
    beta[mask] = torch.acos(xyz[mask, 1]).detach()

    # longitude (alpha)
    alpha = torch.zeros_like(beta)
    alpha[~mask] = torch.atan2(xyz[~mask, 0], xyz[~mask, 2])
    alpha[mask] = torch.atan2(xyz[mask, 0], xyz[mask, 2]).detach()
    if training:
        # random gamma (roll)
        gamma = torch.rand_like(alpha) * 2 * torch.pi
        # gamma = torch.zeros_like(alpha)
    else:
        gamma = torch.zeros_like(alpha) #* 2 * torch.pi

    # intrinsic to extrinsic swap
    return -gamma, -beta, -alpha

# # follow fairchem 2.4.0 and e3nn 0.4.0
# # quick fix: for some special case e.g. [0,0,0],[0,1,0],[0,x,0]
# def init_edge_rot_euler_angles(edge_distance_vec,ts = 0.000001):
#     edge_vec_0 = edge_distance_vec
#     edge_vec_0_distance = torch.norm(edge_vec_0,dim = 1) #torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

#     # Handle small distances using torch.where to maintain graph integrity
#     # Instead of boolean indexing, we conditionally replace values
#     edge_vec_0_distance = torch.where(
#         edge_vec_0_distance < ts,
#         torch.ones_like(edge_vec_0_distance), # Avoid division by zero
#         edge_vec_0_distance
#     )
    
#     # Calculate mask logic fully (without boolean indexing for flow control)
#     # Original mask logic combined with pole check
#     xyz = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))
    
#     # Combine distance check and pole check
#     orig_dist_mask = torch.norm(edge_distance_vec, dim=1) < ts 
#     pole_mask = xyz[:, 1].abs().isclose(xyz.new_ones(1))
#     mask = orig_dist_mask | pole_mask

#     safe_y = torch.clamp(xyz[:, 1], -1-1, 1+1)
#     beta_full = torch.acos(safe_y)
#     beta = torch.where(mask, beta_full.detach(), beta_full)
#     alpha_full = torch.atan2(xyz[:, 0], xyz[:, 2])
#     alpha = torch.where(mask, alpha_full.detach(), alpha_full)

#     # random gamma (roll)
#     gamma = torch.rand_like(alpha) * 2 * torch.pi

#     # intrinsic to extrinsic swap
#     return -gamma, -beta, -alpha


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L37
# In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
# https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92
def wigner_D(
    lv: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    _Jd: list[torch.Tensor],
):
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[lv]
    Xa = _z_rot_mat(alpha, lv)
    Xb = _z_rot_mat(beta, lv)
    Xc = _z_rot_mat(gamma, lv)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle: torch.Tensor, lv: int):
    M = angle.new_zeros((*angle.shape, 2 * lv + 1, 2 * lv + 1))

    # The following code needs to replaced for a for loop because
    # torch.export barfs on outer product like operations
    # ie: torch.outer(frequences, angle) (same as frequencies * angle[..., None])
    # will place a non-sense Guard on the dimensions of angle when attempting to export setting
    # angle (edge dimensions) as dynamic. This may be fixed in torch2.4.

    # inds = torch.arange(0, 2 * lv + 1, 1, device=device)
    # reversed_inds = torch.arange(2 * lv, -1, -1, device=device)
    # frequencies = torch.arange(lv, -lv - 1, -1, dtype=dtype, device=device)
    # M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    # M[..., inds, inds] = torch.cos(frequencies * angle[..., None])

    inds = list(range(0, 2 * lv + 1, 1))
    reversed_inds = list(range(2 * lv, -1, -1))
    frequencies = list(range(lv, -lv - 1, -1))
    for i in range(len(frequencies)):
        M[..., inds[i], reversed_inds[i]] = torch.sin(frequencies[i] * angle)
        M[..., inds[i], inds[i]] = torch.cos(frequencies[i] * angle)
    return M


def eulers_to_wigner(
    eulers: torch.Tensor,
    start_lmax: int,
    end_lmax: int,
    Jd: list[torch.Tensor],
    l3_sequential = None
):
    """
    set <rot_clip=True> to handle gradient instability when using gradient-based force/stress prediction.
    """
    alpha, beta, gamma = eulers

    size = int((end_lmax + 1) ** 2) - int((start_lmax) ** 2)
    wigner = torch.zeros(len(alpha), size, size, device=alpha.device, dtype=alpha.dtype)
    start = 0
    for lmax in range(start_lmax, end_lmax + 1):
        block = wigner_D(lmax, alpha, beta, gamma, Jd)
        end = start + block.size()[1]
        wigner[:, start:end, start:end] = block
        start = end
    
    if l3_sequential is not None:
        s = sum([(2*tmp_l3+1)*tmp_l3_cnt for tmp_l3,tmp_l3_cnt in l3_sequential])
        start = 0
        wigner_inv = torch.zeros(len(alpha), s, s, device=alpha.device, dtype=alpha.dtype)
        for idx,(tmp_l3,tmp_l3_cnt) in enumerate(l3_sequential):
            block = wigner_D(tmp_l3, alpha, beta, gamma, Jd)
            for _ in range(tmp_l3_cnt):
                wigner_inv[:, start:start+tmp_l3*2+1, start:start+tmp_l3*2+1] = block
                start += tmp_l3*2+1
    else:
        wigner_inv = wigner
    return wigner,torch.transpose(wigner_inv, 1, 2).contiguous()


class SO3_Embedding:
    """
    Helper functions for performing operations on irreps embedding

    Args:
        length (int):           Batch size
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
        num_channels (int):     Number of channels
        device:                 Device of the output
        dtype:                  type of the output tensors
    """

    def __init__(
        self,
        length,
        lmax_list,
        num_channels,
        device,
        dtype,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.device = device
        self.dtype = dtype
        self.num_resolutions = len(lmax_list)

        self.num_coefficients = 0
        for i in range(self.num_resolutions):
            self.num_coefficients = self.num_coefficients + int((lmax_list[i] + 1) ** 2)

        embedding = torch.zeros(
            length,
            self.num_coefficients,
            self.num_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.set_embedding(embedding)
        self.set_lmax_mmax(lmax_list, lmax_list.copy())

    # Clone an embedding of irreps
    def clone(self):
        clone = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.num_channels,
            self.device,
            self.dtype,
        )
        clone.set_embedding(self.embedding.clone())
        return clone

    # Initialize an embedding of irreps
    def set_embedding(self, embedding):
        self.length = len(embedding)
        self.embedding = embedding

    # Set the maximum order to be the maximum degree
    def set_lmax_mmax(self, lmax_list, mmax_list):
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list

    # Expand the node embeddings to the number of edges
    def _expand_edge(self, edge_index):
        embedding = self.embedding[edge_index]
        self.set_embedding(embedding)

    # Initialize an embedding of irreps of a neighborhood
    def expand_edge(self, edge_index):
        x_expand = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.num_channels,
            self.device,
            self.dtype,
        )
        x_expand.set_embedding(self.embedding[edge_index])
        return x_expand

    # Compute the sum of the embeddings of the neighborhood
    def _reduce_edge(self, edge_index, num_nodes):
        new_embedding = torch.zeros(
            num_nodes,
            self.num_coefficients,
            self.num_channels,
            device=self.embedding.device,
            dtype=self.embedding.dtype,
        )
        new_embedding.index_add_(0, edge_index, self.embedding)
        self.set_embedding(new_embedding)

    # Reshape the embedding l -> m
    def _m_primary(self, mapping):
        self.embedding = torch.einsum("nac, ba -> nbc", self.embedding, mapping.to_m)

    # Reshape the embedding m -> l
    def _l_primary(self, mapping):
        # print(self.embedding.dtype,self.dtype)
        self.embedding = torch.einsum("nac, ab -> nbc", self.embedding, mapping.to_m)

    # Rotate the embedding
    def _rotate(self, SO3_rotation, lmax_list, mmax_list):
        if self.num_resolutions == 1:
            embedding_rotate = SO3_rotation[0].rotate(
                self.embedding, lmax_list[0], mmax_list[0]
            )
        else:
            offset = 0
            embedding_rotate = torch.tensor([], device=self.device, dtype=self.dtype)
            for i in range(self.num_resolutions):
                num_coefficients = int((self.lmax_list[i] + 1) ** 2)
                embedding_i = self.embedding[:, offset : offset + num_coefficients]
                embedding_rotate = torch.cat(
                    [
                        embedding_rotate,
                        SO3_rotation[i].rotate(embedding_i, lmax_list[i], mmax_list[i]),
                    ],
                    dim=1,
                )
                offset = offset + num_coefficients

        self.embedding = embedding_rotate
        self.set_lmax_mmax(lmax_list.copy(), mmax_list.copy())

    # Rotate the embedding by the inverse of the rotation matrix
    def _rotate_inv(self, SO3_rotation, mappingReduced):
        if self.num_resolutions == 1:
            embedding_rotate = SO3_rotation[0].rotate_inv(
                self.embedding, self.lmax_list[0], self.mmax_list[0]
            )
        else:
            offset = 0
            embedding_rotate = torch.tensor([], device=self.device, dtype=self.dtype)
            for i in range(self.num_resolutions):
                num_coefficients = mappingReduced.res_size[i]
                embedding_i = self.embedding[:, offset : offset + num_coefficients]
                embedding_rotate = torch.cat(
                    [
                        embedding_rotate,
                        SO3_rotation[i].rotate_inv(
                            embedding_i, self.lmax_list[i], self.mmax_list[i]
                        ),
                    ],
                    dim=1,
                )
                offset = offset + num_coefficients
        self.embedding = embedding_rotate

        # Assume mmax = lmax when rotating back
        for i in range(self.num_resolutions):
            self.mmax_list[i] = int(self.lmax_list[i])
        self.set_lmax_mmax(self.lmax_list, self.mmax_list)

    # Compute point-wise spherical non-linearity
    def _grid_act(self, SO3_grid, act, mappingReduced):
        offset = 0
        for i in range(self.num_resolutions):
            num_coefficients = mappingReduced.res_size[i]

            if self.num_resolutions == 1:
                x_res = self.embedding
            else:
                x_res = self.embedding[
                    :, offset : offset + num_coefficients
                ].contiguous()
            to_grid_mat = SO3_grid[self.lmax_list[i]][
                self.mmax_list[i]
            ].get_to_grid_mat(self.device)
            from_grid_mat = SO3_grid[self.lmax_list[i]][
                self.mmax_list[i]
            ].get_from_grid_mat(self.device)

            x_grid = torch.einsum("bai, zic -> zbac", to_grid_mat, x_res)
            x_grid = act(x_grid)
            x_res = torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)
            if self.num_resolutions == 1:
                self.embedding = x_res
            else:
                self.embedding[:, offset : offset + num_coefficients] = x_res
            offset = offset + num_coefficients

    # Compute a sample of the grid
    def to_grid(self, SO3_grid, lmax=-1):
        if lmax == -1:
            lmax = max(self.lmax_list)

        to_grid_mat_lmax = SO3_grid[lmax][lmax].get_to_grid_mat(self.device)
        grid_mapping = SO3_grid[lmax][lmax].mapping

        offset = 0
        x_grid = torch.tensor([], device=self.device)

        for i in range(self.num_resolutions):
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)
            if self.num_resolutions == 1:
                x_res = self.embedding
            else:
                x_res = self.embedding[
                    :, offset : offset + num_coefficients
                ].contiguous()
            to_grid_mat = to_grid_mat_lmax[
                :, :, grid_mapping.coefficient_idx(self.lmax_list[i], self.lmax_list[i])
            ]
            x_grid = torch.cat(
                [x_grid, torch.einsum("bai, zic -> zbac", to_grid_mat, x_res)], dim=3
            )
            offset = offset + num_coefficients

        return x_grid

    # Compute irreps from grid representation
    def _from_grid(self, x_grid, SO3_grid, lmax=-1):
        if lmax == -1:
            lmax = max(self.lmax_list)

        from_grid_mat_lmax = SO3_grid[lmax][lmax].get_from_grid_mat(self.device)
        grid_mapping = SO3_grid[lmax][lmax].mapping

        offset = 0
        offset_channel = 0
        for i in range(self.num_resolutions):
            from_grid_mat = from_grid_mat_lmax[
                :, :, grid_mapping.coefficient_idx(self.lmax_list[i], self.lmax_list[i])
            ]
            if self.num_resolutions == 1:
                temp = x_grid
            else:
                temp = x_grid[
                    :, :, :, offset_channel : offset_channel + self.num_channels
                ]
            x_res = torch.einsum("bai, zbac -> zic", from_grid_mat, temp)
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)

            if self.num_resolutions == 1:
                self.embedding = x_res
            else:
                self.embedding[:, offset : offset + num_coefficients] = x_res

            offset = offset + num_coefficients
            offset_channel = offset_channel + self.num_channels

    def to_e3nn_embeddings(self):
        from e3nn.io import SphericalTensor
        from e3nn.o3 import Irreps

        embedding = self.embedding.reshape(self.length, -1)

        l = o3.Irreps(
            str(SphericalTensor(self.lmax_list[-1], 1, -1)).replace(
                "1x", f"{self.num_channels}x"
            )
        )
        # multiple channels
        return l, embedding


class CoefficientMappingModule(torch.nn.Module):
    """
    Helper module for coefficients used to reshape l <--> m and to get coefficients of specific degree or order

    Args:
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
        mmax_list (list:int):   List of maximum order of the spherical harmonics
    """

    def __init__(
        self,
        lmax_list,
        mmax_list,
    ):
        super().__init__()

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)

        # Temporarily use `cpu` as device and this will be overwritten.
        self.device = "cpu"

        # Compute the degree (l) and order (m) for each entry of the embedding
        l_harmonic = torch.tensor([], device=self.device).long()
        m_harmonic = torch.tensor([], device=self.device).long()
        m_complex = torch.tensor([], device=self.device).long()

        res_size = torch.zeros([self.num_resolutions], device=self.device).long()

        offset = 0
        for i in range(self.num_resolutions):
            for l in range(0, self.lmax_list[i] + 1):
                mmax = min(self.mmax_list[i], l)
                m = torch.arange(-mmax, mmax + 1, device=self.device).long()
                m_complex = torch.cat([m_complex, m], dim=0)
                m_harmonic = torch.cat([m_harmonic, torch.abs(m).long()], dim=0)
                l_harmonic = torch.cat([l_harmonic, m.fill_(l).long()], dim=0)
            res_size[i] = len(l_harmonic) - offset
            offset = len(l_harmonic)

        num_coefficients = len(l_harmonic)
        # `self.to_m` moves m components from different L to contiguous index
        to_m = torch.zeros([num_coefficients, num_coefficients], device=self.device)
        m_size = torch.zeros([max(self.mmax_list) + 1], device=self.device).long()

        # The following is implemented poorly - very slow. It only gets called
        # a few times so haven't optimized.
        offset = 0
        for m in range(max(self.mmax_list) + 1):
            idx_r, idx_i = self.complex_idx(m, -1, m_complex, l_harmonic)

            for idx_out, idx_in in enumerate(idx_r):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)

            m_size[m] = int(len(idx_r))

            for idx_out, idx_in in enumerate(idx_i):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

        to_m = to_m.detach()

        # save tensors and they will be moved to GPU
        self.register_buffer("l_harmonic", l_harmonic)
        self.register_buffer("m_harmonic", m_harmonic)
        self.register_buffer("m_complex", m_complex)
        self.register_buffer("res_size", res_size)
        self.register_buffer("to_m", to_m)
        self.register_buffer("m_size", m_size)

        # for caching the output of `coefficient_idx`
        self.lmax_cache, self.mmax_cache = None, None
        self.mask_indices_cache = None
        self.rotate_inv_rescale_cache = None

    # Return mask containing coefficients of order m (real and imaginary parts)
    def complex_idx(self, m, lmax, m_complex, l_harmonic):
        """
        Add `m_complex` and `l_harmonic` to the input arguments
        since we cannot use `self.m_complex`.
        """
        if lmax == -1:
            lmax = max(self.lmax_list)

        indices = torch.arange(len(l_harmonic), device=self.device)
        # Real part
        mask_r = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(m))
        mask_idx_r = torch.masked_select(indices, mask_r)

        mask_idx_i = torch.tensor([], device=self.device).long()
        # Imaginary part
        if m != 0:
            mask_i = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(-m))
            mask_idx_i = torch.masked_select(indices, mask_i)

        return mask_idx_r, mask_idx_i

    # Return mask containing coefficients less than or equal to degree (l) and order (m)
    def coefficient_idx(self, lmax, mmax):
        if (self.lmax_cache is not None) and (self.mmax_cache is not None):
            if (self.lmax_cache == lmax) and (self.mmax_cache == mmax):
                if self.mask_indices_cache is not None:
                    return self.mask_indices_cache

        mask = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_harmonic.le(mmax))
        self.device = mask.device
        indices = torch.arange(len(mask), device=self.device)
        mask_indices = torch.masked_select(indices, mask)
        self.lmax_cache, self.mmax_cache = lmax, mmax
        self.mask_indices_cache = mask_indices
        return self.mask_indices_cache

    # Return the re-scaling for rotating back to original frame
    # this is required since we only use a subset of m components for SO(2) convolution
    def get_rotate_inv_rescale(self, lmax, mmax):
        if (self.lmax_cache is not None) and (self.mmax_cache is not None):
            if (self.lmax_cache == lmax) and (self.mmax_cache == mmax):
                if self.rotate_inv_rescale_cache is not None:
                    return self.rotate_inv_rescale_cache

        if self.mask_indices_cache is None:
            self.coefficient_idx(lmax, mmax)

        rotate_inv_rescale = torch.ones(
            (1, (lmax + 1) ** 2, (lmax + 1) ** 2), device=self.device
        )
        for l in range(lmax + 1):
            if l <= mmax:
                continue
            start_idx = l**2
            length = 2 * l + 1
            rescale_factor = math.sqrt(length / (2 * mmax + 1))
            rotate_inv_rescale[
                :, start_idx : (start_idx + length), start_idx : (start_idx + length)
            ] = rescale_factor
        rotate_inv_rescale = rotate_inv_rescale[:, :, self.mask_indices_cache]
        self.rotate_inv_rescale_cache = rotate_inv_rescale
        return self.rotate_inv_rescale_cache

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax_list={self.lmax_list}, mmax_list={self.mmax_list})"




class SO3_Grid(torch.nn.Module):
    """
    Helper functions for grid representation of the irreps

    Args:
        lmax (int):   Maximum degree of the spherical harmonics
        mmax (int):   Maximum order of the spherical harmonics
    """

    def __init__(
        self,
        lmax,
        mmax,
        normalization="integral",
        resolution=None,
    ):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.lat_resolution = 2 * (self.lmax + 1)
        if lmax == mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * (self.mmax) + 1
        if resolution is not None:
            self.lat_resolution = resolution
            self.long_resolution = resolution

        self.mapping = CoefficientMappingModule([self.lmax], [self.lmax])

        device = "cpu"

        to_grid = ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization,  # normalization="integral",
            device=device,
        )
        to_grid_mat = torch.einsum("mbi, am -> bai", to_grid.shb, to_grid.sha).detach()
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l**2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx : (start_idx + length)] = (
                    to_grid_mat[:, :, start_idx : (start_idx + length)] * rescale_factor
                )
        to_grid_mat = to_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization,  # normalization="integral",
            device=device,
        )
        from_grid_mat = torch.einsum(
            "am, mbi -> bai", from_grid.sha, from_grid.shb
        ).detach()
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l**2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                from_grid_mat[:, :, start_idx : (start_idx + length)] = (
                    from_grid_mat[:, :, start_idx : (start_idx + length)]
                    * rescale_factor
                )
        from_grid_mat = from_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        # save tensors and they will be moved to GPU
        self.register_buffer("to_grid_mat", to_grid_mat)
        self.register_buffer("from_grid_mat", from_grid_mat)

    # Compute matrices to transform irreps to grid
    def get_to_grid_mat(self, device):
        return self.to_grid_mat

    # Compute matrices to transform grid to irreps
    def get_from_grid_mat(self, device):
        return self.from_grid_mat

    # Compute grid from irreps representation
    def to_grid(self, embedding, lmax, mmax):
        to_grid_mat = self.to_grid_mat[:, :, self.mapping.coefficient_idx(lmax, mmax)]
        grid = torch.einsum("bai, zic -> zbac", to_grid_mat, embedding)
        return grid

    # Compute irreps from grid representation
    def from_grid(self, grid, lmax, mmax):
        from_grid_mat = self.from_grid_mat[
            :, :, self.mapping.coefficient_idx(lmax, mmax)
        ]
        embedding = torch.einsum("bai, zbac -> zic", from_grid_mat, grid)
        return embedding


# -*- coding: utf-8 -*-
"""
    1. Normalize features of shape (N, sphere_basis, C),
    with sphere_basis = (lmax + 1) ** 2.

    2. The difference from `layer_norm.py` is that all type-L vectors have
    the same number of channels and input features are of shape (N, sphere_basis, C).
"""


@torch.jit.script
def mask_after_k_persample(n_sample: int, n_len: int, persample_k: torch.Tensor):
    assert persample_k.shape[0] == n_sample
    assert persample_k.max() <= n_len
    device = persample_k.device
    mask = torch.zeros([n_sample, n_len + 1], device=device)
    mask[torch.arange(n_sample, device=device), persample_k] = 1
    mask = mask.cumsum(dim=1)[:, :-1]
    return mask.type(torch.bool)


class CellExpander:
    def __init__(
        self,
        cutoff=10.0,
        expanded_token_cutoff=512,
        pbc_expanded_num_cell_per_direction=10,
    ):
        self.cells = []
        for i in range(
            -pbc_expanded_num_cell_per_direction,
            pbc_expanded_num_cell_per_direction + 1,
        ):
            for j in range(
                -pbc_expanded_num_cell_per_direction,
                pbc_expanded_num_cell_per_direction + 1,
            ):
                for k in range(
                    -pbc_expanded_num_cell_per_direction,
                    pbc_expanded_num_cell_per_direction + 1,
                ):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    self.cells.append([i, j, k])

        self.cells = torch.tensor(self.cells)

        self.cell_mask_for_pbc = self.cells != 0

        self.candidate_cells = torch.tensor(
            [
                [i, j, k]
                for i in range(
                    -pbc_expanded_num_cell_per_direction,
                    pbc_expanded_num_cell_per_direction + 1,
                )
                for j in range(
                    -pbc_expanded_num_cell_per_direction,
                    pbc_expanded_num_cell_per_direction + 1,
                )
                for k in range(
                    -pbc_expanded_num_cell_per_direction,
                    pbc_expanded_num_cell_per_direction + 1,
                )
            ]
        )

        self.cutoff = cutoff

        self.expanded_token_cutoff = expanded_token_cutoff


        self.pbc_expanded_num_cell_per_direction = pbc_expanded_num_cell_per_direction

        self.conflict_cell_offsets = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i != 0 or j != 0 or k != 0:
                        self.conflict_cell_offsets.append([i, j, k])
        self.conflict_cell_offsets = torch.tensor(self.conflict_cell_offsets)  # 26 x 3

        conflict_to_consider = self.cells.unsqueeze(
            1
        ) - self.conflict_cell_offsets.unsqueeze(
            0
        )  # num_expand_cell x 26 x 3
        conflict_to_consider_mask = (
            ((conflict_to_consider * self.cells.unsqueeze(1)) >= 0)
            & (torch.abs(conflict_to_consider) <= self.cells.unsqueeze(1).abs())
        ).all(
            dim=-1
        )  # num_expand_cell x 26
        conflict_to_consider_mask &= (
            (conflict_to_consider <= pbc_expanded_num_cell_per_direction)
            & (conflict_to_consider >= -pbc_expanded_num_cell_per_direction)
        ).all(
            dim=-1
        )  # num_expand_cell x 26
        self.conflict_to_consider_mask = conflict_to_consider_mask

    

    def _get_cell_tensors(self, cell, use_local_attention=None):
        # fitler impossible offsets according to cell size and cutoff
        def _get_max_offset_for_dim(cell, dim):
            lattice_vec_0 = cell[:, dim, :]
            lattice_vec_1_2 = cell[
                :, torch.arange(3, dtype=torch.long, device=cell.device) != dim, :
            ]
            normal_vec = torch.cross(
                lattice_vec_1_2[:, 0, :], lattice_vec_1_2[:, 1, :], dim=-1
            )
            normal_vec = normal_vec / normal_vec.norm(dim=-1, keepdim=True)

            max_offset = int(
                torch.max(
                    torch.ceil(
                        self.cutoff
                        / torch.abs(torch.sum(normal_vec * lattice_vec_0, dim=-1))
                    )
                )
            )
            return max_offset

        max_offsets = []
        for i in range(3):
            try:
                max_offset = _get_max_offset_for_dim(cell, i)
            except Exception as e:
                logging.warning(f"{e} with cell {cell}")
                max_offset = self.pbc_expanded_num_cell_per_direction
            max_offsets.append(max_offset)
        max_offsets = torch.tensor(max_offsets, device=cell.device)
        self.cells = self.cells.to(device=cell.device)
        self.cell_mask_for_pbc = self.cell_mask_for_pbc.to(device=cell.device)
        mask = (self.cells.abs() <= max_offsets).all(dim=-1)
        selected_cell = self.cells[mask, :]
        return selected_cell, self.cell_mask_for_pbc[mask, :], mask

    def _get_conflict_mask(self, cell, pos, atoms):
        batch_size, max_num_atoms = pos.size()[:2]
        self.conflict_cell_offsets = self.conflict_cell_offsets.to(device=pos.device)
        self.conflict_to_consider_mask = self.conflict_to_consider_mask.to(
            device=pos.device
        )
        offset = torch.bmm(
            self.conflict_cell_offsets.unsqueeze(0)
            .repeat(batch_size, 1, 1)
            .to(dtype=cell.dtype),
            cell,
        )  # batch_size x 26 x 3
        expand_pos = (pos.unsqueeze(1) + offset.unsqueeze(2)).reshape(
            batch_size, -1, 3
        )  # batch_size x max_num_atoms x 3, batch_size x 26 x 3 -> batch_size x (26 x max_num_atoms) x 3
        expand_dist = (pos.unsqueeze(2) - expand_pos.unsqueeze(1)).norm(
            p=2, dim=-1
        )  # batch_size x max_num_atoms x (26 x max_num_atoms)

        expand_atoms = atoms.repeat(
            1, self.conflict_cell_offsets.size()[0]
        )  # batch_size x (26 x max_num_atoms)
        atoms_identical_mask = atoms.unsqueeze(-1) == expand_atoms.unsqueeze(
            1
        )  # batch_size x max_num_atoms x (26 x max_num_atoms)

        conflict_mask = (
            ((expand_dist < 1e-5) & atoms_identical_mask)
            .any(dim=1)
            .reshape(batch_size, -1, max_num_atoms)
        )  # batch_size x 26 x max_num_atoms
        all_conflict_mask = (
            torch.bmm(
                self.conflict_to_consider_mask.unsqueeze(0)
                .to(dtype=cell.dtype)
                .repeat(batch_size, 1, 1),
                conflict_mask.to(dtype=cell.dtype),
            )
            .long()
            .bool()
        )  # batch_size x num_expand_cell x 26, batch_size x 26 x max_num_atoms -> batch_size x num_expand_cell x max_num_atoms
        return all_conflict_mask

    def check_conflict(self, pos, atoms, pbc_expand_batched):
        # ensure that there's no conflict in the expanded atoms
        # a conflict means that two atoms (or special tokens) share both the same position and token type
        expand_pos = pbc_expand_batched["expand_pos"]
        all_pos = torch.cat([pos, expand_pos], dim=1)
        num_expanded_atoms = all_pos.size()[1]
        all_dist = (all_pos.unsqueeze(1) - all_pos.unsqueeze(2)).norm(p=2, dim=-1)
        outcell_index = pbc_expand_batched[
            "outcell_index"
        ]  # batch_size x expanded_max_num_atoms
        all_atoms = torch.cat(
            [atoms, torch.gather(atoms, dim=-1, index=outcell_index)], dim=-1
        )
        atom_identical_mask = all_atoms.unsqueeze(1) == all_atoms.unsqueeze(-1)
        full_mask = torch.cat([atoms.eq(0), pbc_expand_batched["expand_mask"]], dim=-1)
        atom_identical_mask = atom_identical_mask.masked_fill(
            full_mask.unsqueeze(-1), False
        )
        atom_identical_mask = atom_identical_mask.masked_fill(
            full_mask.unsqueeze(1), False
        )
        conflict_mask = (all_dist < 1e-5) & atom_identical_mask
        conflict_mask[
            :,
            torch.arange(num_expanded_atoms, device=all_pos.device),
            torch.arange(num_expanded_atoms, device=all_pos.device),
        ] = False
        assert ~(
            conflict_mask.any()
        ), f"{all_dist[conflict_mask]} {all_atoms[conflict_mask.any(dim=-2)]}"


    def expand_includeself(
        self,
        pos,
        pbc,
        # num_atoms,
        atomic_numbers,
        cell,
        neighbors_radius,
        # use_local_attention=True,
        use_grad=False,
        padding_mask=None,
    ):
        with torch.set_grad_enabled(use_grad):
            # pos = pos.float()
            cell = cell.float()
            batch_size, max_num_atoms = pos.size()[:2]
            if padding_mask is None:
                padding_mask = torch.zeros((batch_size,max_num_atoms)).bool()
            pos = torch.where(
                padding_mask.unsqueeze(dim=-1).repeat(1, 1, 3), 999.0, pos.float()
            )

            cell_tensor, cell_mask, _ = self._get_cell_tensors(
                cell,
            )

            cell_tensor = torch.cat(
                [torch.zeros((1, 3), device=cell_tensor.device), cell_tensor], dim=0
            )
            # self.cell_mask_for_pbc = self.cells != 0
            cell_mask = torch.cat(
                [torch.zeros((1, 3), device=cell_mask.device).bool(), cell_mask], dim=0
            )

            cell_tensor = (
                cell_tensor.unsqueeze(0).repeat(batch_size, 1, 1).to(dtype=cell.dtype)
            )
            num_expanded_cell = cell_tensor.size()[1]
            offset = torch.bmm(cell_tensor, cell)  # B x num_expand_cell x 3
            expand_pos = pos.unsqueeze(1) + offset.unsqueeze(
                2
            )  # B x num_expand_cell x T x 3
            expand_pos = expand_pos.view(
                batch_size, -1, 3
            )  # B x (num_expand_cell x T) x 3

            # eliminate duplicate atoms of expanded atoms, comparing with the original unit cell
            expand_dist = torch.norm(
                pos.unsqueeze(2) - expand_pos.unsqueeze(1), p=2, dim=-1
            )  # B x T x (num_expand_cell x T)
            # expand_atoms = atoms.repeat(1, num_expanded_cell)
            # expand_atom_identical = atoms.unsqueeze(-1) == expand_atoms.unsqueeze(1)

            if neighbors_radius[0] is None or neighbors_radius[0]>=expand_pos.shape[1]:
                expand_mask = (expand_dist < neighbors_radius[1])
            else:
                values, _ = torch.topk(
                    expand_dist, neighbors_radius[0] + 1, dim=-1, largest=False
                )
                expand_mask = (
                    expand_dist <= (values[:, :, neighbors_radius[0]].unsqueeze(dim=-1))
                ) & (expand_dist < neighbors_radius[1])
                # & (expand_dist > 1e-5)
                #     (
                #     (expand_dist > 1e-5) | ~expand_atom_identical
                # )  # B x T x (num_expand_cell x T)
            
            expand_mask = (
                expand_mask
                & (~padding_mask.repeat(1, num_expanded_cell).unsqueeze(1))
                & (~(atomic_numbers.eq(0).unsqueeze(-1)))
            )

            expand_mask = torch.sum(expand_mask, dim=1) > 0
            # if not use_local_attention:
            #     expand_mask = expand_mask & (~all_conflict_mask)
            expand_mask = expand_mask & (
                ~(atomic_numbers.eq(0).repeat(1, num_expanded_cell))
            )  # B x (num_expand_cell x T)

            cell_mask = (
                torch.all(pbc.unsqueeze(1) >= cell_mask.unsqueeze(0), dim=-1)
                .unsqueeze(-1)
                .repeat(1, 1, max_num_atoms)
                .reshape(expand_mask.size())
            )  # B x (num_expand_cell x T)
            expand_mask &= cell_mask
            expand_len = torch.sum(expand_mask, dim=-1)

            # threshold_num_expanded_token = torch.zeros(batch_size,device=cell_tensor.device).int()+self.expanded_token_cutoff
            # torch.clamp(
            #     self.expanded_token_cutoff - num_atoms*0, min=0
            # )

            max_expand_len = torch.max(expand_len)

            # # cutoff within expanded_token_cutoff tokens
            # need_threshold = expand_len > threshold_num_expanded_token
            # if need_threshold.any():
            #     min_expand_dist = expand_dist.masked_fill(expand_dist <= 1e-5, np.inf)
            #     expand_dist_mask = (
            #         atomic_numbers.eq(0).unsqueeze(-1) | atomic_numbers.eq(0).unsqueeze(1)
            #     ).repeat(1, 1, num_expanded_cell)
            #     min_expand_dist = min_expand_dist.masked_fill_(expand_dist_mask, np.inf)
            #     min_expand_dist = min_expand_dist.masked_fill_(
            #         ~cell_mask.unsqueeze(1), np.inf
            #     )
            #     min_expand_dist = torch.min(min_expand_dist, dim=1)[0]

            #     need_threshold_distances = min_expand_dist[
            #         need_threshold
            #     ]  # B x (num_expand_cell x T)
            #     threshold_num_expanded_token = threshold_num_expanded_token[
            #         need_threshold
            #     ]
            #     threshold_dist = torch.sort(
            #         need_threshold_distances, dim=-1, descending=False
            #     )[0]

            #     threshold_dist = torch.gather(
            #         threshold_dist, 1, threshold_num_expanded_token.unsqueeze(-1).long()
            #     )

            #     new_expand_mask = min_expand_dist[need_threshold] < threshold_dist
            #     expand_mask[need_threshold] &= new_expand_mask
            #     expand_len = torch.sum(expand_mask, dim=-1)
            #     max_expand_len = torch.max(expand_len)

            outcell_index = torch.zeros(
                [batch_size, max_expand_len], dtype=torch.long, device=pos.device
            )
            expand_pos_compressed = torch.zeros(
                [batch_size, max_expand_len, 3], dtype=pos.dtype, device=pos.device
            )
            outcell_all_index = torch.arange(
                max_num_atoms, dtype=torch.long, device=pos.device
            ).repeat(num_expanded_cell)
            for i in range(batch_size):
                outcell_index[i, : expand_len[i]] = outcell_all_index[expand_mask[i]]
                # assert torch.all(outcell_index[i, :expand_len[i]] < natoms[i])
                expand_pos_compressed[i, : expand_len[i], :] = expand_pos[
                    i, expand_mask[i], :
                ]

            # if use_local_attention:
            #     expand_dist_compress = (
            #         pos.unsqueeze(2) - expand_pos_compressed.unsqueeze(1)
            #     ).norm(p=2, dim=-1)
            #     local_attention_weight = self.polynomial(
            #         expand_dist_compress,
            #         cutoff=self.cutoff,
            #     )
            #     is_periodic = pbc.any(dim=-1)
            #     local_attention_weight = local_attention_weight.masked_fill(
            #         ~is_periodic.unsqueeze(-1).unsqueeze(-1), 1.0
            #     )
            #     local_attention_weight = local_attention_weight.masked_fill(
            #         atoms.eq(0).unsqueeze(-1), 1.0
            #     )
            #     expand_mask = mask_after_k_persample(
            #         batch_size, max_expand_len, expand_len
            #     )
            #     local_attention_weight = local_attention_weight.masked_fill(
            #         atoms.eq(0).unsqueeze(-1), 1.0
            #     )
            #     local_attention_weight = local_attention_weight.masked_fill(
            #         expand_mask.unsqueeze(1), 0.0
            #     )
            #     pbc_expand_batched = {
            #         "expand_pos": expand_pos_compressed,
            #         "outcell_index": outcell_index,
            #         "expand_mask": expand_mask,
            #         "local_attention_weight": local_attention_weight,
            #         "expand_node_type_edge": expand_node_type_edge,
            #     }
            # else:
            pbc_expand_batched = {
                "expand_node_pos": expand_pos_compressed.float(),
                "outcell_index": outcell_index,
                "expand_node_mask": logical_not(mask_after_k_persample(
                    batch_size, max_expand_len, expand_len
                )),
                # "local_attention_weight": None,
                # "expand_node_type_edge": expand_node_type_edge,
            }
            # print(pbc_expand_batched["expand_mask"],
            #       torch.sum(local_attention_weight==0,dim = 1)!=local_attention_weight.shape[1])

            # expand_pos_no_offset = torch.gather(
            #     pos, dim=1, index=outcell_index.unsqueeze(-1)
            # )
            # offset = expand_pos_compressed - expand_pos_no_offset
            # init_expand_pos_no_offset = torch.gather(
            #     init_pos, dim=1, index=outcell_index.unsqueeze(-1)
            # )
            # init_expand_pos = init_expand_pos_no_offset + offset
            # init_expand_pos = init_expand_pos.masked_fill(
            #     pbc_expand_batched["expand_mask"].unsqueeze(-1),
            #     0.0,
            # )

            # pbc_expand_batched["init_expand_pos"] = init_expand_pos

            # # self.check_conflict(pos, atoms, pbc_expand_batched)
            # print(f"local attention weight {local_attention_weight.numel()} zero:{torch.sum(local_attention_weight==0)}")
            # # print(torch.sum(local_attention_weight==0,dim = 1)==(local_attention_weight.shape[1]))
            # print("N1+N2, ",local_attention_weight.shape[2],torch.sum(
            #     torch.sum(local_attention_weight==0,dim = 1)==local_attention_weight.shape[1])/(local_attention_weight.shape[0]*1.0))
            # pbc_expand_batched["local_attention_weight"] = None
            return pbc_expand_batched


def get_normalization_layer(
    norm_type, lmax, num_channels, eps=1e-5, affine=True, normalization="component"
):
    assert norm_type in [
        "layer_norm",
        "layer_norm_sh",
        "rms_norm_sh",
        "rms_norm_sh_BL",
        "identity",
    ]
    # if norm_type == "layer_norm":
    #     norm_class = EquivariantLayerNormArray
    # elif norm_type == "layer_norm_sh" or norm_type == "layer_norm_sh_BL":
    #     norm_class = EquivariantLayerNormArraySphericalHarmonics
    if norm_type == "rms_norm_sh" or norm_type == "rms_norm_sh_BL":
        norm_class = EquivariantRMSNormArraySphericalHarmonicsV2_BL
    elif norm_type == "identity":
        norm_class = nn.Identity
    else:
        raise ValueError
    return norm_class(lmax, num_channels, eps, affine, normalization)


def get_l_to_all_m_expand_index(lmax):
    expand_index = torch.zeros([(lmax + 1) ** 2]).long()
    for l in range(lmax + 1):
        start_idx = l**2
        length = 2 * l + 1
        expand_index[start_idx : (start_idx + length)] = l
    return expand_index


class EquivariantLayerNormArray(nn.Module):
    def __init__(
        self, lmax, num_channels, eps=1e-5, affine=True, normalization="component"
    ):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(lmax + 1, num_channels))
            self.affine_bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"

    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        out = []

        for l in range(self.lmax + 1):
            start_idx = l**2
            length = 2 * l + 1

            feature = node_input.narrow(1, start_idx, length)

            # For scalars, first compute and subtract the mean
            if l == 0:
                feature_mean = torch.mean(feature, dim=2, keepdim=True)
                feature = feature - feature_mean

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
            elif self.normalization == "component":
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)

            if self.affine:
                weight = self.affine_weight.narrow(0, l, 1)  # [1, C]
                weight = weight.view(1, 1, -1)  # [1, 1, C]
                feature_norm = feature_norm * weight  # [N, 1, C]

            feature = feature * feature_norm

            if self.affine and l == 0:
                bias = self.affine_bias
                bias = bias.view(1, 1, -1)
                feature = feature + bias

            out.append(feature)

        out = torch.cat(out, dim=1)

        return out


class EquivariantLayerNormArraySphericalHarmonics(nn.Module):
    """
    1. Normalize over L = 0.
    2. Normalize across all m components from degrees L > 0.
    3. Do not normalize separately for different L (L > 0).
    """

    def __init__(
        self,
        lmax,
        num_channels,
        eps=1e-5,
        affine=True,
        normalization="component",
        std_balance_degrees=True,
    ):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees

        # for L = 0
        self.norm_l0 = torch.nn.LayerNorm(
            self.num_channels, eps=self.eps, elementwise_affine=self.affine
        )

        # for L > 0
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.lmax, self.num_channels))
        else:
            self.register_parameter("affine_weight", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2 - 1, 1)
            for l in range(1, self.lmax + 1):
                start_idx = l**2 - 1
                length = 2 * l + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (
                    1.0 / length
                )
            balance_degree_weight = balance_degree_weight / self.lmax
            self.register_buffer("balance_degree_weight", balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees})"

    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """
        out_shape = node_input.shape[:-2]
        node_input = node_input.reshape(
            out_shape.numel(), (self.lmax + 1) ** 2, self.num_channels
        )

        out = []

        # for L = 0
        feature = node_input.narrow(1, 0, 1)
        feature = self.norm_l0(feature)
        out.append(feature)

        # for L > 0
        if self.lmax > 0:
            num_m_components = (self.lmax + 1) ** 2
            feature = node_input.narrow(1, 1, num_m_components - 1)

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
            elif self.normalization == "component":
                if self.std_balance_degrees:
                    feature_norm = feature.pow(
                        2
                    )  # [N, (L_max + 1)**2 - 1, C], without L = 0
                    feature_norm = torch.einsum(
                        "nic, ia -> nac", feature_norm, self.balance_degree_weight
                    )  # [N, 1, C]
                else:
                    feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)

            for l in range(1, self.lmax + 1):
                start_idx = l**2
                length = 2 * l + 1
                feature = node_input.narrow(1, start_idx, length)  # [N, (2L + 1), C]
                if self.affine:
                    weight = self.affine_weight.narrow(0, (l - 1), 1)  # [1, C]
                    weight = weight.view(1, 1, -1)  # [1, 1, C]
                    feature_scale = feature_norm * weight  # [N, 1, C]
                else:
                    feature_scale = feature_norm
                feature = feature * feature_scale
                out.append(feature)

        out = torch.cat(out, dim=1)
        return out.reshape(out_shape + ((self.lmax + 1) ** 2, self.num_channels))


class EquivariantRMSNormArraySphericalHarmonics(nn.Module):
    """
    1. Normalize across all m components from degrees L >= 0.
    """

    def __init__(
        self, lmax, num_channels, eps=1e-5, affine=True, normalization="component"
    ):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        # for L >= 0
        if self.affine:
            self.affine_weight = nn.Parameter(
                torch.ones((self.lmax + 1), self.num_channels)
            )
        else:
            self.register_parameter("affine_weight", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"

    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        out = []

        # for L >= 0
        feature = node_input
        if self.normalization == "norm":
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
        elif self.normalization == "component":
            feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        for l in range(0, self.lmax + 1):
            start_idx = l**2
            length = 2 * l + 1
            feature = node_input.narrow(1, start_idx, length)  # [N, (2L + 1), C]
            if self.affine:
                weight = self.affine_weight.narrow(0, l, 1)  # [1, C]
                weight = weight.view(1, 1, -1)  # [1, 1, C]
                feature_scale = feature_norm * weight  # [N, 1, C]
            else:
                feature_scale = feature_norm
            feature = feature * feature_scale
            out.append(feature)

        out = torch.cat(out, dim=1)
        return out


class EquivariantRMSNormArraySphericalHarmonicsV2(nn.Module):
    """
    1. Normalize across all m components from degrees L >= 0.
    2. Expand weights and multiply with normalized feature to prevent slicing and concatenation.
    """

    def __init__(
        self,
        lmax,
        num_channels,
        eps=1e-5,
        affine=True,
        normalization="component",
        centering=True,
        std_balance_degrees=True,
    ):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees

        # for L >= 0
        if self.affine:
            self.affine_weight = nn.Parameter(
                torch.ones((self.lmax + 1), self.num_channels)
            )
            if self.centering:
                self.affine_bias = nn.Parameter(torch.zeros(self.num_channels))
            else:
                self.register_parameter("affine_bias", None)
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

        expand_index = get_l_to_all_m_expand_index(self.lmax)
        self.register_buffer("expand_index", expand_index)

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1)
            for l in range(self.lmax + 1):
                start_idx = l**2
                length = 2 * l + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (
                    1.0 / length
                )
            balance_degree_weight = balance_degree_weight / (self.lmax + 1)
            self.register_buffer("balance_degree_weight", balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, centering={self.centering}, std_balance_degrees={self.std_balance_degrees})"

    def forward(self, node_input, batch=None):
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        feature = node_input

        if self.centering:
            feature_l0 = feature.narrow(1, 0, 1)
            feature_l0_mean = feature_l0.mean(dim=2, keepdim=True)  # [N, 1, 1]
            feature_l0 = feature_l0 - feature_l0_mean
            feature = torch.cat(
                (feature_l0, feature.narrow(1, 1, feature.shape[1] - 1)), dim=1
            )

        # for L >= 0
        if self.normalization == "norm":
            assert not self.std_balance_degrees
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
        elif self.normalization == "component":
            if self.std_balance_degrees:
                feature_norm = feature.pow(2)  # [N, (L_max + 1)**2, C]
                feature_norm = torch.einsum(
                    "nic, ia -> nac", feature_norm, self.balance_degree_weight
                )  # [N, 1, C]
            else:
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        if self.affine:
            weight = self.affine_weight.view(
                1, (self.lmax + 1), self.num_channels
            )  # [1, L_max + 1, C]
            weight = torch.index_select(
                weight, dim=1, index=self.expand_index
            )  # [1, (L_max + 1)**2, C]
            feature_norm = feature_norm * weight  # [N, (L_max + 1)**2, C]

        out = feature * feature_norm

        if self.affine and self.centering:
            out[:, 0:1, :] = out.narrow(1, 0, 1) + self.affine_bias.view(
                1, 1, self.num_channels
            )

        return out


class EquivariantRMSNormArraySphericalHarmonicsV2_BL(nn.Module):
    """
    1. Normalize across all m components from degrees L >= 0.
    2. Expand weights and multiply with normalized feature to prevent slicing and concatenation.
    """

    def __init__(
        self,
        lmax,
        num_channels,
        eps=1e-5,
        affine=True,
        normalization="component",
        centering=True,
        std_balance_degrees=True,
    ):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees

        # for L >= 0
        if self.affine:
            self.affine_weight = nn.Parameter(
                torch.ones((self.lmax + 1), self.num_channels)
            )
            if self.centering:
                self.affine_bias = nn.Parameter(torch.zeros(self.num_channels))
            else:
                self.register_parameter("affine_bias", None)
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

        expand_index = get_l_to_all_m_expand_index(self.lmax)
        self.register_buffer("expand_index", expand_index)

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1)
            for l in range(self.lmax + 1):
                start_idx = l**2
                length = 2 * l + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (
                    1.0 / length
                )
            balance_degree_weight = balance_degree_weight / (self.lmax + 1)
            self.register_buffer("balance_degree_weight", balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, centering={self.centering}, std_balance_degrees={self.std_balance_degrees})"

    def forward(self, node_input, batch=None):
        """
        Assume input is of shape [N, sphere_basis, C]
        """
        out_shape = node_input.shape[:-2]
        feature = node_input.reshape(
            out_shape.numel(), (self.lmax + 1) ** 2, self.num_channels
        )

        if self.centering:
            feature_l0 = feature.narrow(1, 0, 1)
            feature_l0_mean = feature_l0.mean(dim=2, keepdim=True)  # [N, 1, 1]
            feature_l0 = feature_l0 - feature_l0_mean
            feature = torch.cat(
                (feature_l0, feature.narrow(1, 1, feature.shape[1] - 1)), dim=1
            )

        # for L >= 0
        if self.normalization == "norm":
            assert not self.std_balance_degrees
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
        elif self.normalization == "component":
            if self.std_balance_degrees:
                feature_norm = feature.pow(2)  # [N, (L_max + 1)**2, C]
                feature_norm = torch.einsum(
                    "nic, ia -> nac", feature_norm, self.balance_degree_weight
                )  # [N, 1, C]
            else:
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        if self.affine:
            weight = self.affine_weight.view(
                1, (self.lmax + 1), self.num_channels
            )  # [1, L_max + 1, C]
            weight = torch.index_select(
                weight, dim=1, index=self.expand_index
            )  # [1, (L_max + 1)**2, C]
            feature_norm = feature_norm * weight  # [N, (L_max + 1)**2, C]

        out = feature * feature_norm

        if self.affine and self.centering:
            out[:, 0:1, :] = out.narrow(1, 0, 1) + self.affine_bias.view(
                1, 1, self.num_channels
            )

        return out.reshape(out_shape + ((self.lmax + 1) ** 2, self.num_channels))


class EquivariantDegreeLayerScale(nn.Module):
    """
    1. Similar to Layer Scale used in CaiT (Going Deeper With Image Transformers (ICCV'21)), we scale the output of both attention and FFN.
    2. For degree L > 0, we scale down the square root of 2 * L, which is to emulate halving the number of channels when using higher L.
    """

    def __init__(self, lmax, num_channels, scale_factor=2.0):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.scale_factor = scale_factor

        self.affine_weight = nn.Parameter(
            torch.ones(1, (self.lmax + 1), self.num_channels)
        )
        for l in range(1, self.lmax + 1):
            self.affine_weight.data[0, l, :].mul_(
                1.0 / math.sqrt(self.scale_factor * l)
            )
        expand_index = get_l_to_all_m_expand_index(self.lmax)
        self.register_buffer("expand_index", expand_index)

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, scale_factor={self.scale_factor})"

    def forward(self, node_input):
        weight = torch.index_select(
            self.affine_weight, dim=1, index=self.expand_index
        )  # [1, (L_max + 1)**2, C]
        node_input = node_input * weight  # [N, (L_max + 1)**2, C]
        return node_input


@torch.jit.script
def gaussian(x, mean, std):
    pi = torch.pi
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)



# From Graphormer
class GaussianRadialBasisLayer(torch.nn.Module):
    def __init__(self, num_basis, cutoff):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff + 0.0
        self.mean = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.std = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.weight = torch.nn.Parameter(torch.ones(1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))

        self.std_init_max = 1.0
        self.std_init_min = 1.0 / self.num_basis
        self.mean_init_max = 1.0
        self.mean_init_min = 0
        torch.nn.init.uniform_(self.mean, self.mean_init_min, self.mean_init_max)
        torch.nn.init.uniform_(self.std, self.std_init_min, self.std_init_max)
        torch.nn.init.constant_(self.weight, 1)
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, dist, node_atom=None, edge_src=None, edge_dst=None):
        x = dist / self.cutoff
        x = x.unsqueeze(-1)
        x = self.weight * x + self.bias
        x = x.expand(-1, self.num_basis)
        mean = self.mean
        std = self.std.abs() + 1e-5
        x = gaussian(x, mean, std)
        return x

    def extra_repr(self):
        return "mean_init_max={}, mean_init_min={}, std_init_max={}, std_init_min={}".format(
            self.mean_init_max, self.mean_init_min, self.std_init_max, self.std_init_min
        )


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        num_basis,
        cutoff: float = 5.0,
        basis_width_scalar: float = 2.0,
    ) -> None:
        super().__init__()
        offset = torch.linspace(0, cutoff, num_basis)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        shape = dist.shape
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        dist = torch.exp(self.coeff * torch.pow(dist, 2))
        return dist.reshape(*shape, -1)


# gaussian layer with edge type (i,j)
class GaussianLayer_Edgetype(nn.Module):
    def __init__(self, K=128, edge_types=512 * 3):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        '''
        x:*,*,*
        edge_types:*,*,*,2
        '''
        out_shape = x.shape
        x = x.view(-1)
        edge_types = edge_types.view(out_shape.numel(),2)
        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        # print(x.shape)
        # x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(1,-1)
        std = self.stds.weight.float().view(1,-1).abs() + 1e-2
        x_rbf = gaussian(x.float(), mean, std).type_as(self.means.weight)
        return x_rbf.reshape(out_shape+(-1,))




def construct_o3irrps(dim, order):
    string = []
    for l in range(order + 1):
        string.append(f"{dim}x{l}e" if l % 2 == 0 else f"{dim}x{l}o")
    return "+".join(string)


def to_torchgeometric_Data(data: dict):
    torchgeometric_data = Data()
    for key in data.keys():
        torchgeometric_data[key] = data[key]
    return torchgeometric_data


def construct_o3irrps_base(dim, order):
    string = []
    for l in range(order + 1):
        string.append(f"{dim}x{l}e")
    return "+".join(string)


# def polynomial(dist: torch.Tensor, cutoff: float):
#     """
#     Polynomial cutoff function,ref: https://arxiv.org/abs/2204.13639
#     Args:
#         dist (tf.Tensor): distance tensor
#         cutoff (float): cutoff distance
#     Returns: polynomial cutoff functions
#     """
#     ratio = torch.div(dist, cutoff)
#     result = (
#         1
#         - 6 * torch.pow(ratio, 5)
#         + 15 * torch.pow(ratio, 4)
#         - 10 * torch.pow(ratio, 3)
#     )
#     return torch.clamp(result, min=0.0)

        
def polynomial(dist: torch.Tensor, cutoff: float, exponent: int = 5) -> torch.Tensor:
    a: float = -(exponent + 1) * (exponent + 2) / 2
    b: float = exponent * (exponent + 2)
    c: float = -exponent * (exponent + 1) / 2
    ratio = dist/cutoff
    env_val = 1 + ratio**exponent * (
        a + ratio * (b + c * ratio)
    )
    return torch.where(ratio<1, env_val, 0)

def smooth_polynomial_bell(dist, cutoff_min, cutoff_max, exponent=2):
    """
    Construct a smooth bell-shaped polynomial function on the interval [x, y]: f(x) = 0, f(y) = 0, f(mid) = 1
    Use: f(t) = (1 - t^2)^exponent, where t is normalized to [-1, 1] 
    Parameters:
        z (Tensor): Input tensor of any shape
        x (float or Tensor): Left endpoint of the interval
        y (float or Tensor): Right endpoint of the interval
        exponent (int): Controls the steepness of the peak, recommended values are 2 (default), 3, 4 
    Tensor: shape is the same as z, values are within [0, 1]
    """
    
    if not isinstance(cutoff_min, torch.Tensor):
        cutoff_min = torch.tensor(cutoff_min, dtype=dist.dtype, device=dist.device)
    if not isinstance(cutoff_max, torch.Tensor):
        cutoff_max = torch.tensor(cutoff_max, dtype=dist.dtype, device=dist.device)

    
    eps = 1e-8
    cutoff_max = torch.maximum(cutoff_max, cutoff_min + eps)

    
    t = (2 * dist - (cutoff_min + cutoff_max)) / (cutoff_max - cutoff_min)  

    
    t = torch.clamp(t, -1.0, 1.0)

    
    f_t = (1.0 - t**2) ** exponent

    return f_t

def SmoothSoftmax(input, edge_dis, max_dist=5.0, dim=2, eps=1e-5, batched_data=None):
    local_attn_weight = polynomial(edge_dis, max_dist)
    input = input.to(torch.float64)
    local_attn_weight = local_attn_weight.to(input.dtype)

    max_value = input.max(dim=dim, keepdim=True).values
    input = input - max_value
    e_ij = torch.exp(input) * local_attn_weight.unsqueeze(-1)
    # e_ij = input * local_attn_weight.unsqueeze(-1)

    if torch.isnan(e_ij).any() or torch.isinf(e_ij).any():
        logger.warning("e_ij has nan or inf: {}", e_ij)
    # Compute softmax along the last dimension
    softmax = e_ij / (torch.sum(e_ij, dim=dim, keepdim=True) + eps)
    # softmax = torch.nn.functional.softmax(e_ij, dim=dim)

    softmax = softmax.to(torch.float32)

    return softmax



class SO3_Linear_e2former(torch.nn.Module):
    def __init__(self, in_features, out_features, lmax, bias=True):
        """
        1. Use `torch.einsum` to prevent slicing and concatenation
        2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        self.weight = torch.nn.Parameter(
            torch.randn((self.lmax + 1), out_features, in_features)
        )
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for l in range(lmax + 1):
            start_idx = l**2
            length = 2 * l + 1
            expand_index[start_idx : (start_idx + length)] = l
        self.register_buffer("expand_index", expand_index,persistent=False)

    def forward(self, input_embedding):
        output_shape = input_embedding.shape[:-2]
        l_sum, hidden = input_embedding.shape[-2:]
        input_embedding = input_embedding.reshape(
            [output_shape.numel()] + [l_sum, hidden]
        )
        weight = torch.index_select(
            self.weight, dim=0, index=self.expand_index
        )  # [(L_max + 1) ** 2, C_out, C_in]
        out = torch.einsum(
            "bmi, moi -> bmo", input_embedding, weight
        )  # [N, (L_max + 1) ** 2, C_out]
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias

        out = out.reshape(output_shape + (l_sum, self.out_features))

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"


# class Learn_PolynomialDistance(torch.nn.Module):
#     def __init__(self, degree, highest_degree=3):
#         """
#         Constructs a polynomial model with learnable coefficients.

#         P(d) = c_0 + c_1 * d + c_2 * d^2 + ... + c_n * d^n

#         :param degree: The highest degree of the polynomial.
#         """
#         super().__init__()
#         self.coefficients = 0.01 * torch.randn(highest_degree + 1)
#         self.coefficients[degree] = 1

#         self.coefficients = torch.nn.Parameter(self.coefficients)
#         self.act = torch.nn.ReLU()

#     def forward(self, distance):
#         """
#         Computes the polynomial value for a given distance.

#         :param distance: The distance value (torch.Tensor)
#         :return: The computed polynomial value.
#         """
#         powers = torch.stack(
#             [distance**i for i in range(len(self.coefficients))], dim=-1
#         )
#         return self.act(torch.sum(self.coefficients * powers, dim=-1))


def drop_path_BL(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], x.shape[1]) + (1,) * (
        x.ndim - 2
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath_BL(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath_BL, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, batch):
        batch_size = batch.max() + 1
        shape = (batch_size,) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        ones = torch.ones(shape, dtype=x.dtype, device=x.device)

        if len(x.shape) == 4:
            drop = drop_path_BL(ones, self.drop_prob, self.training)
        elif len(x.shape) == 3:
            drop = drop_path(ones, self.drop_prob, self.training)
        return x * drop[batch]

    def extra_repr(self):
        return "drop_prob={}".format(self.drop_prob)


class RadialProfile(nn.Module):
    def __init__(self, ch_list, use_layer_norm=True, use_offset=True):
        super().__init__()
        modules = []
        input_channels = ch_list[0]
        for i in range(len(ch_list)):
            if i == 0:
                continue
            modules.append(nn.Linear(input_channels, ch_list[i], bias=use_offset))
            input_channels = ch_list[i]

            if i == len(ch_list) - 1:
                break

            if use_layer_norm:
                modules.append(nn.LayerNorm(ch_list[i]))
            # modules.append(nn.ReLU())
            # modules.append(Activation(o3.Irreps('{}x0e'.format(ch_list[i])),
            #    acts=[torch.nn.functional.silu]))
            # modules.append(Activation(o3.Irreps('{}x0e'.format(ch_list[i])),
            #    acts=[ShiftedSoftplus()]))
            modules.append(torch.nn.SiLU())

        self.net = nn.Sequential(*modules)

    def forward(self, f_in):
        f_out = self.net(f_in)
        return f_out


class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x):
        ## x could be any dimension.
        return (1 - self.alpha) * x * torch.sigmoid(x) + self.alpha * x

    def extra_repr(self):
        return "negative_slope={}".format(self.alpha)






class SO3_Linear2Scalar_e2former(torch.nn.Module):
    def __init__(self, in_features, out_features, lmax, hidden_features = None,bias=True):
        """
        1. Use `torch.einsum` to prevent slicing and concatenation
        2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax
        hidden_features = out_features // 2 if hidden_features is None else hidden_features
        self.weight = torch.nn.Parameter(
            torch.randn((self.lmax + 1), hidden_features, in_features)
        )
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(hidden_features))

        self.weight2 = torch.nn.Parameter(
            torch.randn((self.lmax + 1), hidden_features, in_features)
        )
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight2, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, hidden_features))

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for l in range(lmax + 1):
            start_idx = l**2
            length = 2 * l + 1
            expand_index[start_idx : (start_idx + length)] = l
        self.register_buffer("expand_index", expand_index)

        self.final_linear = nn.Sequential(
            nn.Linear(hidden_features * (lmax + 1), hidden_features),
            nn.LayerNorm(hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, input_embedding):
        output_shape = input_embedding.shape[:-2]
        l_sum, hidden = input_embedding.shape[-2:]
        input_embedding = input_embedding.reshape(
            [output_shape.numel()] + [l_sum, hidden]
        )
        weight = torch.index_select(
            self.weight, dim=0, index=self.expand_index
        )  # [(L_max + 1) ** 2, C_out, C_in]
        out = torch.einsum(
            "bmi, moi -> bmo", input_embedding, weight
        )  # [N, (L_max + 1) ** 2, C_out]
        out[:, 0:1, :] = out.narrow(1, 0, 1) + self.bias

        weight2 = torch.index_select(
            self.weight2, dim=0, index=self.expand_index
        )  # [(L_max + 1) ** 2, C_out, C_in]
        out2 = torch.einsum(
            "bmi, moi -> bmo", input_embedding, weight2
        )  # [N, (L_max + 1) ** 2, C_out]
        out2[:, 0:1, :] = out2.narrow(1, 0, 1)

        tmp_out = []
        for l in range(self.lmax + 1):
            tmp_out.append(
                torch.sum(
                    out[:, l**2 : (l + 1) ** 2] * out2[:, l**2 : (l + 1) ** 2],
                    dim=1,
                )
            )

        tmp_out = self.final_linear(torch.cat(tmp_out, dim=-1))

        tmp_out = tmp_out.reshape(output_shape + (self.out_features,))

        return tmp_out




class EquivariantDropout(nn.Module):
    def __init__(self, dim, lmax, drop_prob):
        """
        equivariant for irreps: [..., irreps]
        """

        super(EquivariantDropout, self).__init__()
        self.lmax = lmax
        self.scalar_dim = dim
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)

    def forward(self, x):
        """
        x: [..., irreps]

        t1 = o3.Irreps("5x0e+4x1e+3x2e")
        func = EquivariantDropout(t1, 0.5)
        out = func(t1.randn(2,3,-1))
        """
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = x.shape
        N = x.shape[:-2].numel()
        x = x.reshape(N, (self.lmax + 1) ** 2, -1)

        mask = torch.ones(
            (N, self.lmax + 1, self.scalar_dim), dtype=x.dtype, device=x.device
        )
        mask = self.drop(mask)
        out = []
        for l in range(self.lmax + 1):
            out.append(x[:, l**2 : (l + 1) ** 2] * mask[:, l : l + 1])
        out = torch.cat(out, dim=1)
        return out.reshape(shape)


class TensorProductRescale(torch.nn.Module):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
        mode="default",
    ):
        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.rescale = rescale
        self.use_bias = bias

        # e3nn.__version__ == 0.4.4
        # Use `path_normalization` == 'none' to remove normalization factor
        if mode == "simple":
            self.tp = Simple_TensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                instructions=instructions,
                rescale=rescale,
                # normalization=normalization,
                # internal_weights=internal_weights,
                # shared_weights=shared_weights,
                # path_normalization="none",
            )
        else:
            self.tp = o3.TensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                instructions=instructions,
                normalization=normalization,
                internal_weights=internal_weights,
                shared_weights=shared_weights,
                path_normalization="none",
            )

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
        # Determine the order for each output tensor and their dims
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
                            self.irreps_bias_dims[slice_idx], dtype=self.tp.weight.dtype
                        )
                    )
                    self.bias += [out_bias]
                    self.bias_slices += [out_slice]
                    self.bias_slice_idx += [slice_idx]
        self.bias = torch.nn.ParameterList(self.bias)

        self.slices_sqrt_k = {}
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {}  # fan_in per slice
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                fan_in = self.calculate_fan_in(instr)
                slices_fan_in[slice_idx] = (
                    slices_fan_in[slice_idx] + fan_in
                    if slice_idx in slices_fan_in.keys()
                    else fan_in
                )
            for instr in self.tp.instructions:
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
            if self.tp.internal_weights:
                for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                    # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                    # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                    # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                    slice_idx = instr[2]
                    if self.rescale:
                        sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                        weight.data.mul_(sqrt_k)
                    # else:
                    #    sqrt_k = 1.
                    #
                    # if self.rescale:
                    # weight.data.uniform_(-sqrt_k, sqrt_k)
                    #    weight.data.mul_(sqrt_k)
                    # self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Initialize the biases
            # for (out_slice_idx, out_slice, out_bias) in zip(self.bias_slice_idx, self.bias_slices, self.bias):
            #    sqrt_k = 1 / slices_fan_in[out_slice_idx] ** 0.5
            #    out_bias.uniform_(-sqrt_k, sqrt_k)

    def forward_tp_rescale_bias(self, x, y, weight=None):
        out = self.tp(x, y, weight)
        # if self.rescale and self.tp.internal_weights:
        #    for (slice, slice_sqrt_k) in self.slices_sqrt_k.values():
        #        out[:, slice] /= slice_sqrt_k
        if self.use_bias:
            for _, slice, bias in zip(self.bias_slice_idx, self.bias_slices, self.bias):
                # out[:, slice] += bias
                out.narrow(-1, slice.start, slice.stop - slice.start).add_(bias)
        return out

    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        return out



class CosineCutoff(torch.nn.Module):
    r"""Appies a cosine cutoff to the input distances.

    .. math::
        \text{cutoffs} =
        \begin{cases}
        0.5 * (\cos(\frac{\text{distances} * \pi}{\text{cutoff}}) + 1.0),
        & \text{if } \text{distances} < \text{cutoff} \\
        0, & \text{otherwise}
        \end{cases}

    Args:
        cutoff (float): A scalar that determines the point at which the cutoff
            is applied.
    """

    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances):
        r"""Applies a cosine cutoff to the input distances.

        Args:
            distances (torch.Tensor): A tensor of distances.

        Returns:
            cutoffs (torch.Tensor): A tensor where the cosine function
                has been applied to the distances,
                but any values that exceed the cutoff are set to 0.
        """
        cutoffs = 0.5 * ((distances * math.pi / self.cutoff).cos() + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


def get_mul_0(irreps):
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0


@compile_mode("trace")
class Activation(torch.nn.Module):
    """
    Directly apply activation when irreps is type-0.
    """

    def __init__(self, irreps_in, acts):
        super().__init__()
        if isinstance(irreps_in, str):
            irreps_in = o3.Irreps(irreps_in)
        assert len(irreps_in) == len(acts), (irreps_in, acts)

        # normalize the second moment
        acts = [
            e3nn.math.normalize2mom(act) if act is not None else None for act in acts
        ]

        from e3nn.util._argtools import _get_device

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError(
                        "Activation: cannot apply an activation function to a non-scalar input."
                    )

                x = torch.linspace(0, 10, 256, device=_get_device(act))

                a1, a2 = act(x), act(-x)
                if (a1 - a2).abs().max() < 1e-5:
                    p_act = 1
                elif (a1 + a2).abs().max() < 1e-5:
                    p_act = -1
                else:
                    p_act = 0

                p_out = p_act if p_in == -1 else p_in
                irreps_out.append((mul, (0, p_out)))

                if p_out == 0:
                    raise ValueError(
                        "Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd."
                    )
            else:
                irreps_out.append((mul, (l_in, p_in)))

        self.irreps_in = irreps_in
        self.irreps_out = o3.Irreps(irreps_out)
        self.acts = torch.nn.ModuleList(acts)
        assert len(self.irreps_in) == len(self.acts)

    # def __repr__(self):
    #    acts = "".join(["x" if a is not None else " " for a in self.acts])
    #    return f"{self.__class__.__name__} [{self.acts}] ({self.irreps_in} -> {self.irreps_out})"
    def extra_repr(self):
        output_str = super(Activation, self).extra_repr()
        output_str = output_str + "{} -> {}, ".format(self.irreps_in, self.irreps_out)
        return output_str

    def forward(self, features, dim=-1):
        # directly apply activation without narrow
        if len(self.acts) == 1:
            return self.acts[0](features)

        output = []
        index = 0
        for (mul, ir), act in zip(self.irreps_in, self.acts):
            if act is not None:
                output.append(act(features.narrow(dim, index, mul)))
            else:
                output.append(features.narrow(dim, index, mul * ir.dim))
            index += mul * ir.dim

        if len(output) > 1:
            return torch.cat(output, dim=dim)
        elif len(output) == 1:
            return output[0]
        else:
            return torch.zeros_like(features)


@compile_mode("script")
class Gate(torch.nn.Module):
    """
    TODO: to be optimized.  Toooooo ugly
    1. Use `narrow` to split tensor.
    2. Use `Activation` in this file.
    """

    def __init__(
        self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated
    ):
        super().__init__()
        irreps_scalars = o3.Irreps(irreps_scalars)
        irreps_gates = o3.Irreps(irreps_gates)
        irreps_gated = o3.Irreps(irreps_gated)

        if len(irreps_gates) > 0 and irreps_gates.lmax > 0:
            raise ValueError(
                f"Gate scalars must be scalars, instead got irreps_gates = {irreps_gates}"
            )
        if len(irreps_scalars) > 0 and irreps_scalars.lmax > 0:
            raise ValueError(
                f"Scalars must be scalars, instead got irreps_scalars = {irreps_scalars}"
            )
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(
                f"There are {irreps_gated.num_irreps} irreps in irreps_gated, but a different number ({irreps_gates.num_irreps}) of gate scalars in irreps_gates"
            )
        # assert len(irreps_scalars) == 1
        # assert len(irreps_gates) == 1

        self.irreps_scalars = irreps_scalars
        self.irreps_gates = irreps_gates
        self.irreps_gated = irreps_gated
        self._irreps_in = (irreps_scalars + irreps_gates + irreps_gated).simplify()

        self.act_scalars = Activation(irreps_scalars, act_scalars)
        irreps_scalars = self.act_scalars.irreps_out

        self.act_gates = Activation(irreps_gates, act_gates)
        irreps_gates = self.act_gates.irreps_out

        self.mul = o3.ElementwiseTensorProduct(irreps_gated, irreps_gates)
        irreps_gated = self.mul.irreps_out

        self._irreps_out = irreps_scalars + irreps_gated

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features):
        scalars_dim = self.irreps_scalars.dim
        gates_dim = self.irreps_gates.dim
        input_dim = self.irreps_in.dim

        scalars = features.narrow(-1, 0, scalars_dim)
        gates = features.narrow(-1, scalars_dim, gates_dim)
        gated = features.narrow(
            -1, (scalars_dim + gates_dim), (input_dim - scalars_dim - gates_dim)
        )

        scalars = self.act_scalars(scalars)
        if gates.shape[-1]:
            gates = self.act_gates(gates)
            gated = self.mul(gated, gates)
            features = torch.cat([scalars, gated], dim=-1)
        else:
            features = scalars
        return features

    @property
    def irreps_in(self):
        """Input representations."""
        return self._irreps_in

    @property
    def irreps_out(self):
        """Output representations."""
        return self._irreps_out


@compile_mode("script")
class Gate_s3(torch.nn.Module):
    """
    TODO: to be optimized.  Toooooo ugly
    1. Use `narrow` to split tensor.
    2. Use `Activation` in this file.
    """

    def __init__(self, sphere_channels, lmax, act_scalars="silu", act_vector="sigmoid"):
        super().__init__()

        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.gates = torch.nn.Linear(sphere_channels, sphere_channels * (lmax + 1))
        bound = 1 / math.sqrt(sphere_channels)
        torch.nn.init.uniform_(self.gates.weight, -bound, bound)

        if act_scalars == "silu":
            self.act_scalars = e3nn.math.normalize2mom(torch.nn.SiLU())
        else:
            raise ValueError("in Gate, only support silu")

        if act_vector == "sigmoid":
            self.act_vector = e3nn.math.normalize2mom(torch.nn.Sigmoid())
        else:
            raise ValueError("in Gate, only support sigmoid for vector")

    def __repr__(self):
        return f"{self.__class__.__name__} sph ({self.sphere_channels} lmax {self.lmax}"

    def forward(self, features):
        input_shape = features.shape
        features = features.reshape(input_shape[:-2].numel(), -1, input_shape[-1])

        scalars = self.gates(features[:, 0:1])
        out = [self.act_scalars(scalars[:, :, : self.sphere_channels])]

        start = 1
        for l in range(1, self.lmax + 1):
            out.append(
                self.act_vector(
                    scalars[
                        :,
                        :,
                        l * self.sphere_channels : l * self.sphere_channels
                        + self.sphere_channels,
                    ]
                )  # __ * 1 * hidden_dim
                * features[:, start : start + 2 * l + 1, :]  # __ * (2l+1) * hidden_dim
            )
            start += 2 * l + 1

        out = torch.cat(out, dim=1)
        return out.reshape(input_shape)

    @property
    def irreps_in(self):
        """Input representations."""
        return self.out


@compile_mode("script")
class FeedForwardNetwork_s3(torch.nn.Module):
    """
    Use two (FCTP + Gate)
    """

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        output_channels,
        lmax,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.slinear_1 = SO3_Linear_e2former(
            self.sphere_channels, self.hidden_channels, lmax=lmax, bias=True
        )

        self.gate = Gate_s3(
            self.hidden_channels, lmax=lmax, act_scalars="silu", act_vector="sigmoid"
        )

        self.slinear_2 = SO3_Linear_e2former(
            self.hidden_channels, self.output_channels, lmax=lmax, bias=True
        )

    def forward(self, node_input, **kwargs):
        """
        irreps_in = o3.Irreps("128x0e+32x1e")
        func =  FeedForwardNetwork(
                irreps_in,
                irreps_in,
                proj_drop=0.1,
            )
        out = func(irreps_in.randn(10,20,-1))
        """
        node_output = self.slinear_1(node_input)
        node_output = self.gate(node_output)
        node_output = self.slinear_2(node_output)
        return node_output


class S2Activation(torch.nn.Module):
    """
    Assume we only have one resolution
    """

    def __init__(self, lmax, mmax):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.act = torch.nn.SiLU()

    def forward(self, inputs, SO3_grid):
        to_grid_mat = SO3_grid.get_to_grid_mat(
            device=None
        )  # `device` is not used
        from_grid_mat = SO3_grid.get_from_grid_mat(device=None)
        x_grid = torch.einsum("bai, zic -> zbac", to_grid_mat, inputs)
        x_grid = self.act(x_grid)
        outputs = torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)
        return outputs


class SeparableS2Activation(torch.nn.Module):
    def __init__(self, lmax, mmax):
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax

        self.scalar_act = torch.nn.SiLU()
        self.s2_act = S2Activation(self.lmax, self.mmax)

    def forward(self, input_scalars, input_tensors, SO3_grid):
        output_scalars = self.scalar_act(input_scalars)
        output_scalars = output_scalars.reshape(
            output_scalars.shape[0], 1, output_scalars.shape[-1]
        )
        output_tensors = self.s2_act(input_tensors, SO3_grid)
        outputs = torch.cat(
            (output_scalars, output_tensors.narrow(1, 1, output_tensors.shape[1] - 1)),
            dim=1,
        )
        return outputs


# follow eSCN
class FeedForwardNetwork_escn(torch.nn.Module):
    """
    FeedForwardNetwork: Perform feedforward network with S2 activation or gate activation

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        activation (str):           Type of activation function
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs.
        use_sep_s2_act (bool):      If `True`, use separable grid MLP when `use_grid_mlp` is True.
    """

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        output_channels,
        lmax,
        grid_resolution=18,
    ):
        super(FeedForwardNetwork_escn, self).__init__()
        self.sphere_channels = sphere_channels
        # self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.so3_grid = torch.nn.ModuleList()
        self.lmax = lmax
        for l in range(lmax + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(lmax + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, m, resolution=grid_resolution  # , normalization="component"
                    )
                )
            self.so3_grid.append(SO3_m_grid)

        self.act = nn.SiLU()
        # Non-linear point-wise comvolution for the aggregated messages
        self.fc1_sphere = nn.Linear(
            2 * self.sphere_channels, self.sphere_channels, bias=False
        )

        self.fc2_sphere = nn.Linear(
            self.sphere_channels, self.sphere_channels, bias=False
        )

        self.fc3_sphere = nn.Linear(
            self.sphere_channels, self.sphere_channels, bias=False
        )

    def forward(self, node_irreps, nore_irreps_his, **kwargs):
        """_summary_
            model = FeedForwardNetwork_grid_nonlinear(
                    sphere_channels = 128,
                    hidden_channels = 128,
                    output_channels = 128,
                    lmax = 4,
                    grid_resolution = 18,
                )
            node_irreps = torch.randn(100,3,25,128)
            node_irreps_his = torch.randn(100,3,25,128)
            model(node_irreps,node_irreps_his).shape
        Args:
            node_irreps (_type_): _description_
            nore_irreps_his (_type_): _description_

        Returns:
            _type_: _description_
        """

        out_shape = node_irreps.shape[:-2]

        node_irreps = node_irreps.reshape(
            out_shape.numel(), (self.lmax + 1) ** 2, self.sphere_channels
        )
        nore_irreps_his = nore_irreps_his.reshape(
            out_shape.numel(), (self.lmax + 1) ** 2, self.sphere_channels
        )

        to_grid_mat = self.so3_grid[self.lmax][self.lmax].get_to_grid_mat(
            device=None
        )  # `device` is not used
        from_grid_mat = self.so3_grid[self.lmax][self.lmax].get_from_grid_mat(
            device=None
        )

        # Compute point-wise spherical non-linearity on aggregated messages
        # Project to grid
        x_grid = torch.einsum(
            "bai, zic -> zbac", to_grid_mat, node_irreps
        )  # input_embedding.to_grid(self.SO3_grid, lmax=max_lmax)
        x_grid_his = torch.einsum("bai, zic -> zbac", to_grid_mat, nore_irreps_his)
        x_grid = torch.cat([x_grid, x_grid_his], dim=3)

        # Perform point-wise convolution
        x_grid = self.act(self.fc1_sphere(x_grid))
        x_grid = self.act(self.fc2_sphere(x_grid))
        x_grid = self.fc3_sphere(x_grid)

        node_irreps = torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)
        return node_irreps.reshape(out_shape + (-1, self.output_channels))



def fibonacci_sphere(samples=100):
    """
    Generate uniform grid points on a unit sphere using the Fibonacci lattice.

    Args:
        samples (int): Number of points.

    Returns:
        torch.Tensor: Shape (samples, 3), unit sphere points.
    """
    indices = torch.arange(0, samples, dtype=torch.float32) + 0.5
    phi = torch.acos(1 - 2 * indices / samples)  # Latitude
    theta = torch.pi * (1 + 5**0.5) * indices  # Longitude

    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)

    return torch.stack([x, y, z], dim=-1)  # Shape (samples, 3)


def gaussian_function(r, gaussian_center, sigma=1, co=1):
    """
    Compute Gaussian function centered at gaussian_center.

    Args:
        r (torch.Tensor): Shape (N,sph_grid, 3), points in space.
        gaussian_center (torch.Tensor): Shape (N,topK or N,uniform point, 3), uniform point between atoms.
        sigma (float): (N,topK or N,uniform point,channel),  Standard deviation of Gaussian .
        coefficient (float): (N,topK or N,uniform point,channel),coefficient of Gaussian.

    Returns:
        torch.Tensor: Shape (N, M), Gaussian values for each point and midpoint.
    """
    N, sph_grid = r.shape[:2]
    gaussian_center = gaussian_center.unsqueeze(dim=3)
    if isinstance(sigma, torch.Tensor):
        sigma = torch.abs(sigma.unsqueeze(dim=3))
        co = co.unsqueeze(dim=3)

    dist = torch.norm(
        r.reshape(N, 1, 1, sph_grid, 3) - gaussian_center, dim=-1, keepdim=True
    )  # Compute Euclidean distances
    # the our put shape is (N,topK or N,sph_grid,uniform point,channel)
    return co * torch.exp(-(dist**2) * sigma)


# uniform_center_count means how many gaussian center between any atom pair.
# channels means in each gaussian, the function count or dimension or channel.


def cartesian_to_spherical(points):
    """
    Convert 3D Cartesian coordinates to spherical coordinates (r, theta, phi).

    Args:
        points (torch.Tensor): Shape (N, 3), 3D Cartesian coordinates.

    Returns:
        tuple: (theta, phi) in radians.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)  # Elevation angle
    phi = torch.atan2(y, x)  # Azimuthal angle
    return theta, phi


# Compute Gaussian function values
import torch


class Electron_Density_Descriptor(torch.nn.Module):
    def __init__(
        self,
        uniform_center_count=10,
        num_sphere_points=100,
        channel=8,
        lmax=3,
        output_channel=None,
        distribution="uniform",
    ):
        super().__init__()
        self.lmax = lmax
        self.uniform_center_count = uniform_center_count
        self.channel = channel
        self.output_channel = output_channel if output_channel is not None else channel
        self.proj = SO3_Linear_e2former(
            self.channel,
            self.output_channel,
            lmax=self.lmax,
        )
        self.gama = torch.nn.Parameter(
            torch.arange(0, uniform_center_count).reshape(1, 1, -1, 1)
            * 1.0
            / uniform_center_count,
            requires_grad=False,
        )
        # Example Usage
        self.sphere_grid = torch.nn.Parameter(
            fibonacci_sphere(num_sphere_points), requires_grad=False
        )

        logger.debug("gama shape={} sphere_grid shape={}", self.gama.shape, self.sphere_grid.shape)
        theta, phi = cartesian_to_spherical(self.sphere_grid)
        self.Y_lm_conj = []
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                # Compute spherical harmonics Y_{l,m} at each grid point
                Y_lm = sp.sph_harm(m, l, phi.numpy(), theta.numpy())  # Shape (N,)
                self.Y_lm_conj.append(
                    torch.tensor(Y_lm.conj(), dtype=torch.float32)
                )  # Take conjugate
        self.Y_lm_conj = torch.nn.Parameter(
            torch.stack(self.Y_lm_conj, dim=0), requires_grad=False
        )

    def forward(self, atom_positions, rji, sigma, co, neighbor_mask):
        # atom_positions = torch.randn((N, 3))  # Random atomic coordinates
        # rji = torch.randn((N,topk or N,1, 3))  # Random atomic coordinates
        # sigma = torch.randn(N,N,uniform_center_count,channel)
        # co = torch.randn(N,N,uniform_center_count,channel)
        output_shape = atom_positions.shape[:-1]
        atom_positions = atom_positions.reshape(-1, 3)
        N = atom_positions.shape[0]
        rji = rji.reshape(N, -1, 3)
        topK = rji.shape[1]

        sigma = torch.abs(sigma).reshape(
            N, topK, self.uniform_center_count, self.channel
        )
        co = co.reshape(N, topK, self.uniform_center_count, self.channel)
        gaussian_center = atom_positions.reshape(N, 1, 1, 3) + self.gama * rji.reshape(
            N, -1, 1, 3
        )
        gaussians = gaussian_function(
            atom_positions.reshape(-1, 1, 3) + self.sphere_grid.reshape(1, -1, 3),
            gaussian_center,
            sigma,
            co,
        )
        atom_center_sphgrid = torch.sum(
            gaussians * neighbor_mask.reshape(N, -1, 1, 1, 1), dim=(1, 2)
        )
        projection = (
            torch.sum(
                atom_center_sphgrid.unsqueeze(dim=1)
                * self.Y_lm_conj.reshape(1, (self.lmax + 1) ** 2, -1, 1),
                dim=2,
            )
            / self.Y_lm_conj.shape[1]
        )  # Normalize by N
        # print(prjection.shape)  # Output: ((lmax+1)^2,) → (16,)
        projection = self.proj(projection)
        return projection.reshape(
            output_shape + ((self.lmax + 1) ** 2, self.output_channel)
        )



def construct_radius_neighbor(node_pos,node_mask,
                              expand_node_pos,expand_node_mask,
                              max_dist,
                              include_mask = None,
                              min_dist = -1,
                              max_neighbors = None,
                              error_check = False,
                              poly = "poly",
                              toy_config = None,
                              ):
    '''
    node_pos: B*L1*3
    node_mask: B*L1  1 means nodes, 0 means padding
    expand_node_pos: B*L2*3
    expand_node_mask: B*L2  1 means nodes, 0 means padding
    radius: float
    outcell_index: B*L2  ranged from [0,L1), 
    max_neighbors: int
    
    poly: "poly" or "poly_bell"
    
    '''

    B,L = node_pos.shape[:2]
    L2  = expand_node_pos.shape[1]
    
    ptr = torch.cat(
            [
                torch.zeros(1,dtype = torch.int32,device=node_pos.device),
                torch.cumsum(torch.sum(node_mask, dim=-1), dim=-1),
            ],
            dim=0,
        )
    expand_ptr = torch.cat(
            [
                torch.zeros(1,dtype = torch.int32,device=node_pos.device),
                torch.cumsum(torch.sum(expand_node_mask, dim=-1), dim=-1),
            ],
            dim=0,
        )
    
    
    edge_vec = node_pos.unsqueeze(2) - expand_node_pos.unsqueeze(1)
    dist = torch.linalg.norm(
                edge_vec, dim=-1, keepdim=False
            )
    # dist = torch.norm(edge_vec, dim=-1)  # B*L*L Attention: ego-connection is 0 here
    dist = torch.where(dist >= min_dist,dist,max_dist+1000)
    if include_mask is not None:
        include_mask = include_mask.unsqueeze(1)
        dist = torch.where(include_mask,dist,max_dist+1000)
    neighbor_withincut = torch.max(torch.sum(dist <= max_dist,dim = -1)[node_mask])
    _, neighbor_indices = dist.sort(dim=-1,descending=False)
    if max_neighbors is not None:
        topK = max(min(expand_node_pos.shape[1], max_neighbors,neighbor_withincut),1)
    else:
        topK = max(min(expand_node_pos.shape[1],neighbor_withincut),1)
    # print("max_neighbors,neighbor_withincut",max_neighbors,neighbor_withincut,edge_vec.shape)
    neighbor_indices = neighbor_indices[:, :, :topK]  # Shape: B*L*K
    dist = torch.gather(dist, dim=-1, index=neighbor_indices)  # Shape: B*L*topK
    f_dist = dist[node_mask]  # flattn_N* topK*

    f_attn_mask = (f_dist > max_dist) .unsqueeze(dim=-1)
    if poly == "poly":
        f_poly_dist = polynomial(
            f_dist, max_dist
        )
        f_poly_dist = torch.where(f_attn_mask.squeeze(dim=-1),0,f_poly_dist)
    elif poly == "poly_bell":
        f_poly_dist = smooth_polynomial_bell(f_dist,min_dist,max_dist,exponent=2)
        f_poly_dist = torch.where(f_attn_mask.squeeze(dim=-1),0,f_poly_dist)
    else: # poly_bell
        raise ValueError("sorry, you must set poly or poly_bell")
    # if outcell_index is None:
    #     f_sparse_idx_node = (neighbor_indices + ptr[:B,None,None])[node_mask]
    # else:
    #     f_sparse_idx_node = (
    #         torch.gather(
    #             outcell_index.unsqueeze(1).repeat(1, L, 1), 2, neighbor_indice
    #         )
    #         + ptr[:B, None, None]
    #     )[node_mask]
    # f_sparse_idx_node = torch.clamp(f_sparse_idx_node, max=ptr[B] - 1)
    f_sparse_idx_expnode = (neighbor_indices + expand_ptr[:B, None, None])[
        node_mask
    ]
    f_sparse_idx_expnode = torch.clamp(f_sparse_idx_expnode, max=expand_ptr[B] - 1)
    f_edge_vec = node_pos[node_mask].unsqueeze(dim=1) - expand_node_pos[expand_node_mask][f_sparse_idx_expnode]

    if error_check:
        f_attn_mask_tmp = f_attn_mask.squeeze(dim=-1)
        a1 = torch.norm(f_edge_vec,dim = -1)[~f_attn_mask_tmp]-f_dist[~f_attn_mask_tmp]
        if torch.max(a1)!=0:
            logger.error("Please verify your code, neighbor selection error happened")
        if torch.sum(node_pos[~node_mask]==0):
            logger.warning("node padding part is zero, will lead to some neighbor dist sort error")
        if torch.sum(expand_node_pos[~expand_node_mask]==0):
            logger.warning("expand_node_pos padding part is zero, will lead to some neighbor dist sort error")
    N, K = f_sparse_idx_expnode.shape
    device = f_sparse_idx_expnode.device

    if toy_config is not None:
        f_attn_mask |= (torch.rand_like(f_attn_mask.float()) < toy_config.get("prob",0))

    if f_attn_mask.dim() == 3: # f_attn_mask may be [N, K, 1]
        valid_mask = ~f_attn_mask.squeeze(-1) # [N, K]
    else:
        valid_mask = ~f_attn_mask
    

    query_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, K)[valid_mask].contiguous()
    neighbor_idx = f_sparse_idx_expnode[valid_mask].contiguous()
    

    return {
        "query_idx": query_idx,
        "neighbor_idx": neighbor_idx,
        "metadata": prepare_sparse_qk_edge_index_metadata(
            query_idx,neighbor_idx,N,expand_ptr[B]
        ),
        "f_edge_vec": f_edge_vec[valid_mask].contiguous(),
        "f_dist": f_dist[valid_mask].contiguous(),
        "f_poly_dist": f_poly_dist[valid_mask].contiguous(),
    }
