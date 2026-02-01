
# -*- coding: utf-8 -*-
import math
import warnings
from typing import Dict, Optional

import e3nn
import torch

from sklearn.cluster import KMeans
from torch_scatter import scatter_mean
from e3nn import o3
from e3nn.util.jit import compile_mode

    
from torch import logical_not, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from .module_utils import (  # ,\; EquivariantInstanceNorm,EquivariantGraphNorm; EquivariantRMSNormArraySphericalHarmonicsV2,; GaussianLayer,; irreps2gate,; sort_irreps_even_first,
    RadialProfile,
    SmoothLeakyReLU,
    SO3_Grid,
    SO3_Linear_e2former,
    SO3_Embedding,
    CoefficientMappingModule
)
from .so2 import _init_edge_rot_mat

def wigner_D(l, alpha, beta, gamma):
    r"""Wigner D matrix representation of :math:`SO(3)`.

    It satisfies the following properties:

    * :math:`D(\text{identity rotation}) = \text{identity matrix}`
    * :math:`D(R_1 \circ R_2) = D(R_1) \circ D(R_2)`
    * :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`
    * :math:`D(\text{rotation around Y axis})` has some property that allows us to use FFT in `ToS2Grid`

    Parameters
    ----------
    l : int
        :math:`l`

    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\alpha` around Y axis, applied third.

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\beta` around X axis, applied second.

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\gamma` around Y axis, applied first.

    Returns
    -------
    `torch.Tensor`
        tensor :math:`D^l(\alpha, \beta, \gamma)` of shape :math:`(2l+1, 2l+1)`
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)
    X = so3_generators(l).to(alpha.device)
    return (
        torch.matrix_exp(alpha * X[1])
        @ torch.matrix_exp(beta * X[0])
        @ torch.matrix_exp(gamma * X[1])
    )


class SO3_Rotation(torch.nn.Module):
    """
    Helper functions for Wigner-D rotations

    Args:
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
    """

    def __init__(self, lmax, irreps="128x1e+64x1e"):
        super().__init__()
        self.lmax = lmax
        self.mapping = CoefficientMappingModule([self.lmax], [self.lmax])

        self.irreps = o3.Irreps(irreps) if isinstance(irreps, str) else irreps

    def set_wigner(self, rot_mat3x3):
        self.device, self.dtype = rot_mat3x3.device, rot_mat3x3.dtype

        start_lmax, end_lmax = 0, self.lmax
        x = rot_mat3x3 @ rot_mat3x3.new_tensor([0.0, 1.0, 0.0])
        alpha, beta = o3.xyz_to_angles(x)
        R = (
            o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
            @ rot_mat3x3
        )
        gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

        (end_lmax + 1) ** 2 - (start_lmax) ** 2
        start = 0
        self.wigner = []
        self.wigner_inv = []

        for lmax in range(start_lmax, end_lmax + 1):
            block = wigner_D(lmax, alpha, beta, gamma)
            end = start + block.size()[1]
            self.wigner.append(block.detach())
            self.wigner_inv.append(torch.transpose(block.detach(), 1, 2).contiguous())
            start = end

    @staticmethod
    def init_edge_rot_mat(self, edge_distance_vec):
        return _init_edge_rot_mat(edge_distance_vec=edge_distance_vec).detach()

    # Rotate the embedding
    def rotate(self, embedding, irreps=None):
        shape = list(embedding.shape[:-2])
        num = embedding.shape[:-2].numel()
        embedding = embedding.reshape(num, (self.lmax + 1) ** 2, -1)

        irreps = self.irreps if irreps is None else irreps
        out = []
        for i in range(len(irreps)):
            l = irreps[i][1].l
            cur = self.wigner[l] @ embedding[:, l**2 : (l + 1) ** 2, :]
            out.append(cur)

        embedding = torch.cat(out, dim=-2).reshape(shape + [(self.lmax + 1) ** 2, -1])
        return embedding

    # Rotate the embedding by the inverse of the rotation matrix
    def rotate_inv(self, embedding, irreps=None):
        shape = list(embedding.shape[:-2])
        num = embedding.shape[:-2].numel()
        embedding = embedding.reshape(num, (self.lmax + 1) ** 2, -1)

        irreps = self.irreps if irreps is None else irreps
        out = []
        for i in range(len(irreps)):
            l = irreps[i][1].l
            irreps[i][0]
            cur = self.wigner_inv[l] @ embedding[:, l**2 : (l + 1) ** 2, :]
            out.append(cur)

        embedding = torch.cat(out, dim=-2).reshape(shape + [(self.lmax + 1) ** 2, -1])
        return embedding
class EdgeDegreeEmbeddingNetwork_eqv2(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding,
        avg_aggregate_num=10,
        number_of_basis=32,
        cutoff=15,
        lmax=2,
        time_embed=False,
        avg_nodes=22,
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

        self.lmax = self.irreps_node_embedding[-1][1].l
        self.sph_ch = self.irreps_node_embedding[0][0]

        # # Statistics of IS2RE 100K
        self.sphere_channels = self.sph_ch
        self.lmax_list = [self.lmax]
        self.mmax_list = [2]
        self.num_resolutions = len(self.lmax_list)

        self.SO3_grid = torch.nn.ModuleList()
        for lval in range(max(self.lmax_list) + 1):
            so3_m_grid = torch.nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                so3_m_grid.append(
                    SO3_Grid(
                        lval,
                        m,
                        resolution=18,
                        normalization="component",
                    )
                )

            self.SO3_grid.append(so3_m_grid)
        self.mappingReduced = CoefficientMapping([self.lmax], [2])

        self.m_0_num_coefficients = self.mappingReduced.m_size[0]
        self.m_all_num_coefficents = len(self.mappingReduced.l_harmonic)

        # Embedding function of the atomic numbers
        self.max_num_elements = 256
        self.edge_channels_list = copy.deepcopy([number_of_basis, 128, 128])
        self.use_atom_edge_embedding = True

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        # Embedding function of distance
        self.edge_channels_list.append(self.m_0_num_coefficients * self.sphere_channels)
        self.rad_func = RadialProfile(self.edge_channels_list)

        self.rescale_factor = avg_nodes

    def _forward(
        self,
        atomic_numbers,
        edge_distance,
        edge_index,
        SO3_edge_rot=None,
        mappingReduced=None,
        attn_mask=None,
    ):
        f_N1, topK = edge_distance.shape[:2]
        edge_distance = edge_distance.reshape(f_N1 * topK, -1)

        if self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )
        else:
            x_edge = edge_distance

        x_edge_m_0 = self.rad_func(x_edge)
        x_edge_m_0 = x_edge_m_0.reshape(
            -1, self.m_0_num_coefficients, self.sphere_channels
        )
        x_edge_m_pad = torch.zeros(
            (
                x_edge_m_0.shape[0],
                (self.m_all_num_coefficents - self.m_0_num_coefficients),
                self.sphere_channels,
            ),
            device=x_edge_m_0.device,
        )
        x_edge_m_all = torch.cat((x_edge_m_0, x_edge_m_pad), dim=1)

        x_edge_embedding = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.sphere_channels,
            device=x_edge_m_all.device,
            dtype=x_edge_m_all.dtype,
        )
        x_edge_embedding.set_embedding(x_edge_m_all)
        x_edge_embedding.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        x_edge_embedding._l_primary(mappingReduced)

        # Rotate back the irreps
        x_edge_embedding._rotate_inv(SO3_edge_rot, mappingReduced)

        out = x_edge_embedding.embedding.reshape(f_N1, topK, -1)
        out = out.masked_fill(attn_mask, 0)
        out = out.reshape(f_N1, topK, (self.lmax + 1) ** 2, -1)

        out = torch.sum(out, dim=1) / self.rescale_factor

        return out

    def forward(
        self,
        node_input,
        node_pos,
        edge_dis,
        atomic_numbers,
        edge_vec,
        batched_data,
        attn_mask,
        edge_scalars,
        time_embed=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        node_pos : postiion
            tensor of shape ``(B, L, 3)``

        edge_scalars : rbf of distance
            tensor of shape ``(B, L, L, number_of_basis)``

        edge_dis : distances of all node pairs
            tensor of shape ``(B, L, L)``


        self__irreps_node_embedding = e3nn.o3.Irreps("128x0e+128x1e+128x2e")
        self__number_of_basis = 64
        self__edge_deg_embed_dense = EdgeDegreeEmbeddingNetwork_eqv2(
            self__irreps_node_embedding,
            23.555,
            cutoff=5,
            number_of_basis=self__number_of_basis,
            time_embed=False,
        )
        B = 2
        L = 10
        basis = 64
        pos = torch.randn(B,L,3)
        dist = torch.norm(pos.unsqueeze(dim = 2)-pos.unsqueeze(dim = 1),dim = -1)
        edge_vec = pos.unsqueeze(dim = 2)-pos.unsqueeze(dim = 1)
        atomic_numbers = torch.randint(0,10,(B,L))
        dist_embedding = torch.randn(B,L,L,basis)
        attn_mask = torch.randn(B,L,L,1)>0
        out = self__edge_deg_embed_dense(
        None,
        pos,
        dist,
        batch=None,
        attn_mask=attn_mask,
        atomic_numbers=atomic_numbers,
        edge_vec=edge_vec,
        batched_data=None,
        time_embed=None,
        edge_scalars=dist_embedding,
        )
        print(out.shape)
        """
        f_N1, topK = attn_mask.shape[:2]
        self.SO3_edge_rot = torch.nn.ModuleList()
        for i in range(1):
            self.SO3_edge_rot.append(
                SO3_Rotation(
                    _init_edge_rot_mat(edge_vec.reshape(f_N1 * topK, 3)),
                    self.lmax_list[i],
                )
            )

        x = SO3_Embedding(
            f_N1,
            self.lmax_list,
            self.sphere_channels,
            node_input.device,
            node_input.dtype,
        )
        x.embedding = node_input
        x_embedding = self._forward(
            atomic_numbers,
            edge_scalars,
            edge_index=(
                batched_data["f_sparse_idx_node"].reshape(-1),
                torch.arange(f_N1).reshape(f_N1, -1).repeat(1, topK).reshape(-1),
            ),
            SO3_edge_rot=self.SO3_edge_rot,
            mappingReduced=self.mappingReduced,
            attn_mask=attn_mask,
        )

        return x_embedding


class BOOEmbedding(torch.nn.Module):
    """Bond Orientational Order embedding module.

    Computes rotationally invariant features from the spherical distribution of bonds
    around each node using spherical harmonics, following Steinhardt et al. [1983].
    """

    def __init__(self, max_l=4, hidden_dim=512):
        """
        Args:
            max_l (int): Maximum order of spherical harmonics to use (L in the paper)
        """
        super().__init__()
        self.max_l = max_l
        self.linear = nn.Linear(max_l + 1, hidden_dim)

    def forward(self, edge_vec, edge_mask):
        """
        Args:
            edge_vec (torch.Tensor): Bond vectors of shape (B, N, K, 3) where:
                B is batch size
                N is number of nodes
                K is max number of neighbors
            edge_mask (torch.Tensor): Mask for valid edges of shape (B, N, K, 1)
            batch (torch.Tensor, optional): Batch indices for each node

        Returns:
            torch.Tensor: BOO features of shape (B, N, max_l+1)
        """
        B, N, K, _ = edge_vec.shape

        # Normalize bond vectors
        edge_vec_norm = torch.norm(edge_vec, dim=-1, keepdim=True)
        edge_vec_normalized = edge_vec / (edge_vec_norm + 1e-10)

        # Count valid neighbors per node
        n_neighbors = edge_mask.squeeze(-1).float().sum(dim=-1)  # Shape: (B, N)
        n_neighbors = n_neighbors.clamp(min=1)  # Avoid division by zero

        # Initialize BOO features
        boo_features = []

        for l in range(self.max_l + 1):
            Y_lm = e3nn.o3.spherical_harmonics(
                l, edge_vec_normalized, normalize=True, normalization="component"
            )

            # Apply mask and normalize by number of neighbors
            Y_lm = Y_lm * edge_mask
            Y_lm = Y_lm / n_neighbors.view(B, N, 1, 1)

            q_lm = torch.sum(Y_lm, dim=2)

            norm_factor = math.sqrt(4 * math.pi / (2 * l + 1))
            q_lm = q_lm * norm_factor
            boo_l = torch.sum(
                torch.abs(q_lm) ** 2, dim=-1, keepdim=True
            )  # Shape: (B, N, 1)

            boo_features.append(boo_l)

        # Concatenate all BOO features
        boo = torch.cat(boo_features, dim=-1)
        boo = self.linear(boo)

        return boo


class CoefficientMapping(torch.nn.Module):
    """
    Helper functions for coefficients used to reshape l<-->m and to get coefficients of specific degree or order

    Args:
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
        mmax_list (list:int):   List of maximum order of the spherical harmonics
        device:                 Device of the output
    """

    def __init__(
        self,
        lmax_list: list[int],
        mmax_list: list[int],
    ) -> None:
        super().__init__()

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)

        # entry of the embedding

        self.l_harmonic = torch.tensor([]).long()
        self.m_harmonic = torch.tensor([]).long()
        self.m_complex = torch.tensor([]).long()

        self.res_size = torch.zeros([self.num_resolutions]).long()
        offset = 0
        for i in range(self.num_resolutions):
            for lval in range(self.lmax_list[i] + 1):
                mmax = min(self.mmax_list[i], lval)
                m = torch.arange(-mmax, mmax + 1).long()
                self.m_complex = torch.cat([self.m_complex, m], dim=0)
                self.m_harmonic = torch.cat(
                    [self.m_harmonic, torch.abs(m).long()], dim=0
                )
                self.l_harmonic = torch.cat(
                    [self.l_harmonic, m.fill_(lval).long()], dim=0
                )
            self.res_size[i] = len(self.l_harmonic) - offset
            offset = len(self.l_harmonic)

        num_coefficients = len(self.l_harmonic)
        self.to_m = torch.nn.Parameter(
            torch.zeros([num_coefficients, num_coefficients]), requires_grad=False
        )
        self.m_size = torch.zeros([max(self.mmax_list) + 1]).long()

        offset = 0
        for m in range(max(self.mmax_list) + 1):
            idx_r, idx_i = self.complex_idx(m)

            for idx_out, idx_in in enumerate(idx_r):
                self.to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)
            self.m_size[m] = int(len(idx_r))

            for idx_out, idx_in in enumerate(idx_i):
                self.to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

    def complex_idx(self, m, lmax: int = -1):
        if lmax == -1:
            lmax = max(self.lmax_list)

        indices = torch.arange(len(self.l_harmonic))
        # Real part
        mask_r = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_complex.eq(m))
        mask_idx_r = torch.masked_select(indices, mask_r)

        mask_idx_i = torch.tensor([]).long()
        # Imaginary part
        if m != 0:
            mask_i = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_complex.eq(-m))
            mask_idx_i = torch.masked_select(indices, mask_i)

        return mask_idx_r, mask_idx_i

    def coefficient_idx(self, lmax: int, mmax: int) -> torch.Tensor:
        mask = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_harmonic.le(mmax))
        indices = torch.arange(len(mask))

        return torch.masked_select(indices, mask)



import copy


class MessageBlock_eqv2(torch.nn.Module):
    """
    SO2EquivariantGraphAttention: Perform MLP attention + non-linear message passing
        SO(2) Convolution with radial function -> S2 Activation -> SO(2) Convolution -> attention weights and non-linear messages
        attention weights * non-linear messages -> Linear

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        output_channels (int):      Number of output channels
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        activation (str):           Type of activation function
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
    """

    def __init__(
        self,
        irreps_node_input="256x0e+256x1e+256x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+32x1e+32x2e",
        alpha_drop=0.1,
        rescale_degree=False,
        nonlinear_message=False,
        proj_drop=0.1,
        tp_type="v1",
        attn_type="first-order",  ## second-order
        add_rope=True,
        layer_id=0,
        irreps_origin="1x0e+1x1e+1x2e",
        neighbor_weight=None,
        atom_type_cnt=256,
        use_atom_edge_embedding: bool = True,
        use_m_share_rad: bool = False,
        use_s2_act_attn: bool = False,
        use_gate_act: bool = False,
        use_attn_renorm: bool = True,
        use_sep_s2_act: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.scalar_dim = self.irreps_node_input[0][0]  # scalar_dim x 0e

        self.sphere_channels = self.scalar_dim
        self.hidden_channels = self.scalar_dim // 2
        self.num_heads = 8
        self.attn_alpha_channels = self.scalar_dim // 2
        self.attn_value_channels = self.scalar_dim // self.num_heads
        self.output_channels = self.scalar_dim
        self.lmax = len(self.irreps_node_input) - 1
        self.lmax_list = [self.lmax]
        self.mmax_list = [2]
        self.num_resolutions = len(self.lmax_list)

        self.SO3_grid = torch.nn.ModuleList()
        for lval in range(max(self.lmax_list) + 1):
            so3_m_grid = torch.nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                so3_m_grid.append(
                    SO3_Grid(
                        lval,
                        m,
                        resolution=18,
                        normalization="component",
                    )
                )

            self.SO3_grid.append(so3_m_grid)
        self.mappingReduced = CoefficientMapping([self.lmax], [2])

        # Embedding function of the atomic numbers
        self.max_num_elements = 256
        self.edge_channels_list = copy.deepcopy(
            [
                attn_weight_input_dim,
                min(128, attn_weight_input_dim),
                min(128, attn_weight_input_dim),
            ]
        )
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.use_gate_act = use_gate_act
        self.use_sep_s2_act = use_sep_s2_act

        assert not self.use_s2_act_attn  # since this is not used

        extra_m0_output_channels = None
        if not self.use_s2_act_attn:
            extra_m0_output_channels = self.num_heads * self.attn_alpha_channels
            if self.use_gate_act:
                extra_m0_output_channels = (
                    extra_m0_output_channels
                    + max(self.lmax_list) * self.hidden_channels
                )
            else:
                if self.use_sep_s2_act:
                    extra_m0_output_channels = (
                        extra_m0_output_channels + self.hidden_channels
                    )

        if self.use_m_share_rad:
            self.edge_channels_list = [
                *self.edge_channels_list,
                2 * self.sphere_channels * (max(self.lmax_list) + 1),
            ]
            self.rad_func = RadialProfile(self.edge_channels_list)
            expand_index = torch.zeros([(max(self.lmax_list) + 1) ** 2]).long()
            for lval in range(max(self.lmax_list) + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                expand_index[start_idx : (start_idx + length)] = lval
            self.register_buffer("expand_index", expand_index)
        #     GateActivation,
        #     S2Activation,
        #     SeparableS2Activation,

        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=(bool(self.use_m_share_rad)),
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad else None
            ),
            extra_m0_output_channels=extra_m0_output_channels,  # for attention weights and/or gate activation
        )

        if self.use_s2_act_attn:
            self.alpha_norm = None
            self.alpha_act = None
            self.alpha_dot = None
        else:
            if self.use_attn_renorm:
                self.alpha_norm = torch.nn.LayerNorm(self.attn_alpha_channels)
            else:
                self.alpha_norm = torch.nn.Identity()
            self.alpha_act = SmoothLeakyReLU()
            self.alpha_dot = torch.nn.Parameter(
                torch.randn(self.num_heads, self.attn_alpha_channels)
            )
            std = 1.0 / math.sqrt(self.attn_alpha_channels)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        if self.use_gate_act:
            self.gate_act = GateActivation(
                lmax=max(self.lmax_list),
                mmax=max(self.mmax_list),
                num_channels=self.hidden_channels,
            )
        else:
            if self.use_sep_s2_act:
                # separable S2 activation
                self.s2_act = SeparableS2Activation(
                    lmax=max(self.lmax_list), mmax=max(self.mmax_list)
                )
            else:
                # S2 activation
                self.s2_act = S2Activation(
                    lmax=max(self.lmax_list), mmax=max(self.mmax_list)
                )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.num_heads * self.attn_value_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=(
                self.num_heads if self.use_s2_act_attn else None
            ),  # for attention weights
        )

        self.proj = SO3_Linear_e2former(
            self.num_heads * self.attn_value_channels,
            self.output_channels,
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
        f_N1, topK = attn_weight.shape[:2]
        num_atoms = node_irreps_input.shape[0]

        self.SO3_edge_rot = torch.nn.ModuleList()
        for i in range(1):
            self.SO3_edge_rot.append(
                SO3_Rotation(
                    _init_edge_rot_mat(edge_vec.reshape(f_N1 * topK, 3)),
                    self.lmax_list[i],
                )
            )

        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            node_irreps_input.device,
            node_irreps_input.dtype,
        )
        x.embedding = node_irreps_input
        x_embedding = self._forward(
            x,
            atomic_numbers,
            edge_distance=attn_weight,
            edge_index=(
                batched_data["f_sparse_idx_node"].reshape(-1),
                torch.arange(f_N1).reshape(f_N1, -1).repeat(1, topK).reshape(-1),
            ),
            SO3_edge_rot=self.SO3_edge_rot,
            mappingReduced=self.mappingReduced,
            attn_mask=attn_mask,
        )

        return x_embedding, attn_weight

    def _forward(
        self,
        x: torch.Tensor,
        atomic_numbers,
        edge_distance: torch.Tensor,
        edge_index,
        SO3_edge_rot,
        mappingReduced,
        attn_mask,
        node_offset: int = 0,
    ):
        f_N1, topK = edge_distance.shape[:2]
        edge_distance = edge_distance.reshape(f_N1 * topK, -1)

        # Uses atomic numbers and edge distance as inputs
        if self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )
        else:
            x_edge = edge_distance

        x_source = x.clone()
        x_target = x.clone()
        x_source._expand_edge(edge_index[0])
        x_target._expand_edge(edge_index[1])

        x_message_data = torch.cat((x_source.embedding, x_target.embedding), dim=2)
        x_message = SO3_Embedding(
            0,
            x_target.lmax_list.copy(),
            x_target.num_channels * 2,
            device=x_target.device,
            dtype=x_target.dtype,
        )
        x_message.set_embedding(x_message_data)
        x_message.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        if self.use_m_share_rad:
            x_edge_weight = self.rad_func(x_edge)
            x_edge_weight = x_edge_weight.reshape(
                -1, (max(self.lmax_list) + 1), 2 * self.sphere_channels
            )
            x_edge_weight = torch.index_select(
                x_edge_weight, dim=1, index=self.expand_index
            )  # [E, (L_max + 1) ** 2, C]
            x_message.embedding = x_message.embedding * x_edge_weight

        x_message._rotate(SO3_edge_rot, self.lmax_list, self.mmax_list)
        if self.use_s2_act_attn:
            x_message = self.so2_conv_1(x_message, x_edge)
        else:
            x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)

        # Activation
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
        if self.use_gate_act:
            # Gate activation
            x_0_gating = x_0_extra.narrow(
                1,
                x_alpha_num_channels,
                x_0_extra.shape[1] - x_alpha_num_channels,
            )  # for activation
            x_0_alpha = x_0_extra.narrow(
                1, 0, x_alpha_num_channels
            )  # for attention weights
            x_message.embedding = self.gate_act(x_0_gating, x_message.embedding)
        else:
            if self.use_sep_s2_act:
                x_0_gating = x_0_extra.narrow(
                    1,
                    x_alpha_num_channels,
                    x_0_extra.shape[1] - x_alpha_num_channels,
                )  # for activation
                x_0_alpha = x_0_extra.narrow(
                    1, 0, x_alpha_num_channels
                )  # for attention weights
                x_message.embedding = self.s2_act(
                    x_0_gating, x_message.embedding, self.SO3_grid
                )
            else:
                x_0_alpha = x_0_extra
                x_message.embedding = self.s2_act(x_message.embedding, self.SO3_grid)

        if self.use_s2_act_attn:
            x_message, x_0_extra = self.so2_conv_2(x_message, x_edge)
        else:
            x_message = self.so2_conv_2(x_message, x_edge)

        # Attention weights
        if self.use_s2_act_attn:
            alpha = x_0_extra
        else:
            x_0_alpha = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels)
            x_0_alpha = self.alpha_norm(x_0_alpha)
            x_0_alpha = self.alpha_act(x_0_alpha)
            alpha = torch.einsum("bik, ik -> bi", x_0_alpha, self.alpha_dot)

        alpha = alpha.reshape(f_N1, topK, self.num_heads)
        alpha = alpha.masked_fill(attn_mask, -1e6)
        alpha = torch.nn.functional.softmax(alpha, 1)
        alpha = alpha.masked_fill(attn_mask, 0)

        alpha = alpha.reshape(-1, 1, self.num_heads, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

        # Attention weights * non-linear messages
        attn = x_message.embedding
        attn = attn.reshape(
            attn.shape[0],
            attn.shape[1],
            self.num_heads,
            self.attn_value_channels,
        )
        attn = attn * alpha
        attn = attn.reshape(
            attn.shape[0],
            attn.shape[1],
            self.num_heads * self.attn_value_channels,
        )
        x_message.embedding = attn

        # Rotate back the irreps
        x_message._rotate_inv(SO3_edge_rot, self.mappingReduced)

        out = torch.sum(
            x_message.embedding.reshape(f_N1, topK, (self.lmax + 1) ** 2, -1), dim=1
        )
        # Project
        return self.proj(out)

