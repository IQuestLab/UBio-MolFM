# -*- coding: utf-8 -*-

import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch

from molfm.models.e2former.module_utils import (
    polynomial,
    smooth_polynomial_bell,
)
from molfm.models.e2former.triton_dr.triton_sparse_qk_autograd import (
    prepare_sparse_qk_edge_index_metadata,
)


def aabb_metrics(points, bmin, bmax):
    dxdydx = torch.max(
        torch.stack([bmin.view(1, 3) - points, points - bmax.view(1, 3)], dim=1),
        dim=1,
    )[0]
    dxdydx = torch.clamp(dxdydx, min=0.0)
    surface_dist = torch.sqrt(torch.sum(dxdydx**2, dim=-1))
    return surface_dist


def distributed_cell_expand(
    pos,
    cell_size,
    cutoff,
    cell_boundMIN,
    cell_boundMAX,
    num_cell_perdirection=1,
    pbc=None,
):
    device = pos.device
    if len(pos.shape) == 3:
        if pos.shape[0] != 1:
            raise ValueError("sorry, we did not support pos batch size >1 in distributed version")
        pos = pos.squeeze(dim=0)
    else:
        raise ValueError("please check your pos shape: 1*N*3")

    cell_per_direction_xyz = [
        np.arange(-num_cell_perdirection, num_cell_perdirection + 1),
        np.arange(-num_cell_perdirection, num_cell_perdirection + 1),
        np.arange(-num_cell_perdirection, num_cell_perdirection + 1),
    ]
    if pbc is not None:
        for idx, direction in enumerate(pbc.view(-1)[:3]):
            if not direction:
                cell_per_direction_xyz[idx] = [0]
    else:
        return {
            "expand_node_pos": pos.unsqueeze(dim=0),
            "outcell_index": torch.arange(pos.shape[0], dtype=torch.long, device=pos.device).unsqueeze(dim=0),
            "expand_node_mask": torch.ones(pos.shape[0], device=pos.device).bool().unsqueeze(dim=0),
        }

    candidate_cells = torch.tensor(
        [
            [i, j, k]
            for i in cell_per_direction_xyz[0]
            for j in cell_per_direction_xyz[1]
            for k in cell_per_direction_xyz[2]
        ]
    ).float().to(device)
    offset = torch.matmul(candidate_cells, cell_size)
    outcell_all_index = torch.arange(pos.shape[0], dtype=torch.long, device=pos.device).repeat(
        candidate_cells.shape[0]
    )

    expand_pos = pos.unsqueeze(dim=0) + offset.unsqueeze(dim=1)
    expand_pos = expand_pos.reshape(-1, 3)

    aabb_dist = aabb_metrics(expand_pos, cell_boundMIN, cell_boundMAX)

    expand_pos = expand_pos[aabb_dist < cutoff]
    outcell_index = outcell_all_index[aabb_dist < cutoff]
    expand_mask = torch.ones(expand_pos.shape[0], device=pos.device).bool()

    return {
        "expand_node_pos": expand_pos.unsqueeze(dim=0),
        "outcell_index": outcell_index.unsqueeze(dim=0),
        "expand_node_mask": expand_mask.unsqueeze(dim=0),
    }


class FaissSearch:
    def __init__(self, rank=0) -> None:
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = rank
        self.index = faiss.GpuIndexFlatL2(res, 3, cfg)

    def _estimate_max_neighbors(self, X_query, X_candidate, max_dist, sample_size=64, min_floor=1):
        if X_candidate.numel() == 0:
            return 1

        num_candidates = int(X_candidate.shape[0])
        if num_candidates <= sample_size:
            return max(1, num_candidates)

        num_queries = int(X_query.shape[0])
        sampled_queries = min(num_queries, sample_size)
        if sampled_queries < num_queries:
            sample_idx = torch.randperm(num_queries, device=X_query.device)[:sampled_queries]
            query_sample = X_query[sample_idx]
        else:
            query_sample = X_query

        sampled_dist = torch.cdist(query_sample, X_candidate)
        sampled_counts = (sampled_dist <= max_dist).sum(dim=1)
        estimated = int(sampled_counts.max().item()) if sampled_counts.numel() else 1

        return max(min_floor, min(num_candidates, estimated))

    def _search_with_optional_unbounded_k(
        self,
        X_query,
        X_candidate,
        max_neighbors,
        max_dist,
        min_dist,
        buffer,
        reset,
    ):
        if X_candidate.shape[0] == 0:
            empty_dist = X_query.new_empty((X_query.shape[0], 0))
            empty_idx = torch.empty((X_query.shape[0], 0), dtype=torch.long, device=X_query.device)
            return empty_dist, empty_idx

        if max_neighbors is not None:
            return self.search(
                X_query,
                X_candidate,
                max_neighbors,
                min_dist=min_dist,
                max_dist=max_dist,
                buffer=buffer,
                reset=reset,
            )

        candidate_count = int(X_candidate.shape[0])
        estimated_k = int(buffer * self._estimate_max_neighbors(X_query, X_candidate, max_dist))
        current_k = min(candidate_count, estimated_k)
        while True:
            Dk, Ik = self.search(
                X_query,
                X_candidate,
                current_k,
                min_dist=min_dist,
                max_dist=max_dist,
                buffer=1,
                reset=reset,
            )
            if current_k >= candidate_count:
                return Dk, Ik
            if Dk.shape[1] < current_k:
                return Dk, Ik

            boundary_is_inside_cutoff = (Dk[:, -1] <= max_dist).any()
            if not boundary_is_inside_cutoff:
                return Dk, Ik

            next_k = min(candidate_count, max(current_k * 2, current_k + 1))
            if next_k == current_k:
                return Dk, Ik
            current_k = next_k

    def search(
        self,
        X_query,
        X_candidate,
        max_neighbors,
        max_dist=5,
        min_dist=0,
        buffer=1.5,
        reset=True,
    ):
        if X_query.dim() == 3:
            assert X_query.shape[0] == 1 and X_query.shape[2] == 3
            assert X_candidate.shape[0] == 1 and X_candidate.shape[2] == 3
            X_query = X_query.squeeze(0)
            X_candidate = X_candidate.squeeze(0)

        assert X_query.is_cuda
        if X_candidate.shape[0] == 0 or max_neighbors <= 0:
            empty_dist = X_query.new_empty((X_query.shape[0], 0))
            empty_idx = torch.empty((X_query.shape[0], 0), dtype=torch.long, device=X_query.device)
            return empty_dist, empty_idx
        if reset:
            self.index.reset()
        self.index.add(X_candidate)
        if min_dist > 0:
            k_raw = min(int(max_neighbors * buffer), X_candidate.shape[0])
        else:
            k_raw = min(max_neighbors, X_candidate.shape[0])
        Dk, Ik = self.index.search(X_query, k_raw)

        Dk = torch.norm(X_query.unsqueeze(dim=1) - X_candidate[Ik], dim=-1)
        Dk = torch.where((Dk >= min_dist) & (Dk <= max_dist), Dk, max_dist + 1000)
        D_sorted, idx_sorted = torch.sort(Dk, dim=1, stable=True, descending=False)

        valid_col = (D_sorted <= max_dist).any(dim=0)
        if valid_col.any():
            D_sorted = D_sorted[:, valid_col]
            idx_sorted = idx_sorted[:, valid_col]
        else:
            D_sorted = D_sorted[:, :0]
            idx_sorted = idx_sorted[:, :0]

        Ik_sorted = Ik.gather(1, idx_sorted)
        return D_sorted[:, :max_neighbors], Ik_sorted[:, :max_neighbors]

    @torch.compiler.disable
    def construct_radius_neighbor(
        self,
        X_query,
        query_mask,
        X_candidate,
        candidate_mask,
        include_mask=None,
        max_dist=5,
        min_dist=0,
        max_neighbors=32,
        poly="poly",
        buffer=1.5,
        reset=True,
        toy_config=None,
    ):
        if X_query.dim() == 3:
            assert X_query.shape[0] == 1 and X_query.shape[2] == 3
            assert X_candidate.shape[0] == 1 and X_candidate.shape[2] == 3

            X_query = X_query.squeeze(0)
            X_candidate = X_candidate.squeeze(0)
            if include_mask is not None:
                include_mask = include_mask.squeeze(0)
        M = X_candidate.shape[0]
        assert X_query.is_cuda
        basic_idx = torch.arange(X_candidate.shape[0], device=X_query.device)
        bound_MIN = torch.min(X_query, dim=0)[0]
        bound_MAX = torch.max(X_query, dim=0)[0]
        dis_can2cub = aabb_metrics(X_candidate, bound_MIN, bound_MAX)
        if include_mask is not None:
            X_candidate_filter = X_candidate[(dis_can2cub <= max_dist) & include_mask]
            basic_idx_filter = basic_idx[(dis_can2cub <= max_dist) & include_mask]
        else:
            X_candidate_filter = X_candidate[dis_can2cub <= max_dist]
            basic_idx_filter = basic_idx[dis_can2cub <= max_dist]

        _, Ik = self._search_with_optional_unbounded_k(
            X_query,
            X_candidate_filter,
            max_neighbors,
            min_dist=min_dist,
            max_dist=max_dist,
            buffer=buffer,
            reset=reset,
        )
        if Ik.shape[1] == 0:
            empty_long = torch.empty((0,), dtype=torch.long, device=X_query.device)
            empty_float = X_query.new_empty((0,))
            empty_vec = X_query.new_empty((0, 3))
            return {
                "query_idx": empty_long,
                "neighbor_idx": empty_long,
                "metadata": prepare_sparse_qk_edge_index_metadata(
                    empty_long, empty_long, X_query.shape[0], M
                ),
                "f_edge_vec": empty_vec,
                "f_dist": empty_float,
                "f_poly_dist": empty_float,
            }
        Ik = basic_idx_filter[Ik]

        f_edge_vec = X_query.unsqueeze(dim=1) - X_candidate[Ik]
        f_dist = torch.norm(f_edge_vec, dim=-1)
        f_dist = torch.where(f_dist < min_dist, 1000 + max_dist, f_dist)

        f_attn_mask = (f_dist > max_dist).unsqueeze(-1)

        if poly == "poly":
            f_poly_dist = polynomial(f_dist, max_dist)
        elif poly == "poly_bell":
            f_poly_dist = smooth_polynomial_bell(f_dist, min_dist, max_dist, exponent=2)
        else:
            raise ValueError("sorry, you must set poly or poly_bell")
        N, K = Ik.shape
        device = Ik.device
        if toy_config is not None:
            f_attn_mask |= torch.rand_like(f_attn_mask.float()) < toy_config.get("prob", 0)
        if f_attn_mask.dim() == 3:
            valid_mask = ~f_attn_mask.squeeze(-1)
        else:
            valid_mask = ~f_attn_mask

        query_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, K)[valid_mask].contiguous()
        neighbor_idx = Ik[valid_mask].contiguous()

        return {
            "query_idx": query_idx,
            "neighbor_idx": neighbor_idx,
            "metadata": prepare_sparse_qk_edge_index_metadata(query_idx, neighbor_idx, N, M),
            "f_edge_vec": f_edge_vec[valid_mask].contiguous(),
            "f_dist": f_dist[valid_mask].contiguous(),
            "f_poly_dist": f_poly_dist[valid_mask].contiguous(),
        }
