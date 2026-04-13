# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl

from .triton_sparse_qk_autograd import prepare_sparse_qk_edge_index_metadata


def _cfgs_outer_reduce():
    cfgs = []
    for bl in (1, 2, 4, 8):
        for bh in (16, 32, 64):
            for nw in (4, 8):
                cfgs.append(
                    triton.Config(
                        {"BLOCK_L": bl, "BLOCK_H": bh},
                        num_warps=nw,
                        num_stages=2,
                    )
                )
    return cfgs


@triton.autotune(configs=_cfgs_outer_reduce(), key=["L", "H"])
@triton.jit
def _edge_outer_reduce_kernel(
    EDGE_SH,
    EDGE_FEA,
    Q_ROW_PTR,
    Q_ALPHA_INDEX,
    OUT,
    N,
    L,
    H,
    s_se,
    s_sl,
    s_fe,
    s_fh,
    s_on,
    s_ol,
    s_oh,
    BLOCK_L: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_lb = tl.program_id(1)
    pid_hb = tl.program_id(2)
    if pid_n >= N:
        return

    l_offs = pid_lb * BLOCK_L + tl.arange(0, BLOCK_L)
    h_offs = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    l_mask = l_offs < L
    h_mask = h_offs < H

    ptr_start = tl.load(Q_ROW_PTR + pid_n)
    ptr_end = tl.load(Q_ROW_PTR + pid_n + 1)

    acc = tl.zeros((BLOCK_L, BLOCK_H), dtype=tl.float32)
    for ptr in range(ptr_start, ptr_end):
        edge_idx = tl.load(Q_ALPHA_INDEX + ptr)
        sh_ptr = EDGE_SH + edge_idx * s_se + l_offs * s_sl
        fea_ptr = EDGE_FEA + edge_idx * s_fe + h_offs * s_fh
        sh_vec = tl.load(sh_ptr, mask=l_mask, other=0.0).to(tl.float32)
        fea_vec = tl.load(fea_ptr, mask=h_mask, other=0.0).to(tl.float32)
        acc += sh_vec[:, None] * fea_vec[None, :]

    out_ptr = OUT + pid_n * s_on + l_offs[:, None] * s_ol + h_offs[None, :] * s_oh
    tl.store(out_ptr, acc, mask=l_mask[:, None] & h_mask[None, :])


def _cfgs_dsh():
    cfgs = []
    for bl in (1, 2, 4, 8):
        for bh in (16, 32, 64):
            cfgs.append(triton.Config({"BLOCK_L": bl, "BLOCK_H": bh}, num_warps=4, num_stages=2))
    return cfgs


@triton.autotune(configs=_cfgs_dsh(), key=["L", "H"])
@triton.jit
def _edge_dsh_kernel(
    GO,
    EDGE_FEA,
    QUERY_INDEX,
    D_SH,
    E,
    L,
    H,
    s_gn,
    s_gl,
    s_gh,
    s_fe,
    s_fh,
    s_qe,
    s_de,
    s_dl,
    BLOCK_L: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_lb = tl.program_id(1)
    if pid_e >= E:
        return

    q = tl.load(QUERY_INDEX + pid_e * s_qe, mask=True, other=0).to(tl.int32)
    l_offs = pid_lb * BLOCK_L + tl.arange(0, BLOCK_L)
    l_mask = l_offs < L

    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)
    h0 = 0
    while h0 < H:
        h_offs = h0 + tl.arange(0, BLOCK_H)
        h_mask = h_offs < H
        go_ptr = GO + q * s_gn + l_offs[:, None] * s_gl + h_offs[None, :] * s_gh
        fea_ptr = EDGE_FEA + pid_e * s_fe + h_offs * s_fh
        go_tile = tl.load(go_ptr, mask=l_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
        fea_vec = tl.load(fea_ptr, mask=h_mask, other=0.0).to(tl.float32)
        acc += tl.sum(go_tile * fea_vec[None, :], axis=1)
        h0 += BLOCK_H

    d_ptr = D_SH + pid_e * s_de + l_offs * s_dl
    tl.store(d_ptr, acc, mask=l_mask)


def _cfgs_dfea():
    cfgs = []
    for bl in (1, 2, 4, 8):
        for bh in (16, 32, 64):
            cfgs.append(triton.Config({"BLOCK_L": bl, "BLOCK_H": bh}, num_warps=4, num_stages=2))
    return cfgs


@triton.autotune(configs=_cfgs_dfea(), key=["L", "H"])
@triton.jit
def _edge_dfea_kernel(
    GO,
    EDGE_SH,
    QUERY_INDEX,
    D_FEA,
    E,
    L,
    H,
    s_gn,
    s_gl,
    s_gh,
    s_se,
    s_sl,
    s_qe,
    s_de,
    s_dh,
    BLOCK_L: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_hb = tl.program_id(1)
    if pid_e >= E:
        return

    q = tl.load(QUERY_INDEX + pid_e * s_qe, mask=True, other=0).to(tl.int32)
    h_offs = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    l0 = 0
    while l0 < L:
        l_offs = l0 + tl.arange(0, BLOCK_L)
        l_mask = l_offs < L
        go_ptr = GO + q * s_gn + l_offs[:, None] * s_gl + h_offs[None, :] * s_gh
        sh_ptr = EDGE_SH + pid_e * s_se + l_offs * s_sl
        go_tile = tl.load(go_ptr, mask=l_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
        sh_vec = tl.load(sh_ptr, mask=l_mask, other=0.0).to(tl.float32)
        acc += tl.sum(go_tile * sh_vec[:, None], axis=0)
        l0 += BLOCK_L

    d_ptr = D_FEA + pid_e * s_de + h_offs * s_dh
    tl.store(d_ptr, acc, mask=h_mask)


def _edge_outer_reduce(edge_sh, edge_fea, q_row_ptr, q_alpha_index, num_queries):
    n = num_queries
    l = edge_sh.shape[1]
    h = edge_fea.shape[1]
    out = torch.zeros((n, l, h), device=edge_sh.device, dtype=torch.float32)
    grid = lambda meta: (n, triton.cdiv(l, meta["BLOCK_L"]), triton.cdiv(h, meta["BLOCK_H"]))
    _edge_outer_reduce_kernel[grid](
        edge_sh,
        edge_fea,
        q_row_ptr,
        q_alpha_index,
        out,
        n,
        l,
        h,
        edge_sh.stride(0),
        edge_sh.stride(1),
        edge_fea.stride(0),
        edge_fea.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    return out


def _edge_dsh(go, edge_fea, query_index):
    e = edge_fea.shape[0]
    l = go.shape[1]
    h = go.shape[2]
    d_sh = torch.zeros((e, l), device=go.device, dtype=torch.float32)
    grid = lambda meta: (e, triton.cdiv(l, meta["BLOCK_L"]))
    _edge_dsh_kernel[grid](
        go,
        edge_fea,
        query_index,
        d_sh,
        e,
        l,
        h,
        go.stride(0),
        go.stride(1),
        go.stride(2),
        edge_fea.stride(0),
        edge_fea.stride(1),
        query_index.stride(0),
        d_sh.stride(0),
        d_sh.stride(1),
    )
    return d_sh


def _edge_dfea(go, edge_sh, query_index):
    e = edge_sh.shape[0]
    l = go.shape[1]
    h = go.shape[2]
    d_fea = torch.zeros((e, h), device=go.device, dtype=torch.float32)
    grid = lambda meta: (e, triton.cdiv(h, meta["BLOCK_H"]))
    _edge_dfea_kernel[grid](
        go,
        edge_sh,
        query_index,
        d_fea,
        e,
        l,
        h,
        go.stride(0),
        go.stride(1),
        go.stride(2),
        edge_sh.stride(0),
        edge_sh.stride(1),
        query_index.stride(0),
        d_fea.stride(0),
        d_fea.stride(1),
    )
    return d_fea


class EdgeDegreeOuterBackward_Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_out, edge_sh, edge_fea, query_index, q_row_ptr, q_alpha_index, num_queries):
        go = grad_out.contiguous().to(torch.float32)
        sh = edge_sh.contiguous()
        fea = edge_fea.contiguous()

        d_sh = _edge_dsh(go, fea, query_index)
        d_fea = _edge_dfea(go, sh, query_index)

        ctx.save_for_backward(go, sh, fea, query_index, q_row_ptr, q_alpha_index)
        ctx.num_queries = num_queries
        return d_sh.to(sh.dtype), d_fea.to(fea.dtype)

    @staticmethod
    def backward(ctx, gg_dsh, gg_dfea):
        go, sh, fea, query_index, q_row_ptr, q_alpha_index = ctx.saved_tensors
        num_queries = ctx.num_queries

        d_go = d_sh = d_fea = None
        gg_dsh_f = None if gg_dsh is None else gg_dsh.contiguous().to(torch.float32)
        gg_dfea_f = None if gg_dfea is None else gg_dfea.contiguous().to(torch.float32)

        if ctx.needs_input_grad[0]:
            d_go = torch.zeros_like(go, dtype=torch.float32)
            if gg_dsh_f is not None:
                d_go = d_go + _edge_outer_reduce(gg_dsh_f, fea, q_row_ptr, q_alpha_index, num_queries)
            if gg_dfea_f is not None:
                d_go = d_go + _edge_outer_reduce(sh, gg_dfea_f, q_row_ptr, q_alpha_index, num_queries)
            d_go = d_go.to(go.dtype)

        if ctx.needs_input_grad[1] and gg_dfea_f is not None:
            d_sh = _edge_dsh(go, gg_dfea_f, query_index).to(sh.dtype)

        if ctx.needs_input_grad[2] and gg_dsh_f is not None:
            d_fea = _edge_dfea(go, gg_dsh_f, query_index).to(fea.dtype)

        return d_go, d_sh, d_fea, None, None, None, None


class EdgeDegreeOuterEdgeIndex_Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_sh, edge_fea, query_index, q_row_ptr, q_alpha_index, num_queries):
        sh = edge_sh.contiguous()
        fea = edge_fea.contiguous()
        out = _edge_outer_reduce(sh, fea, q_row_ptr, q_alpha_index, num_queries)

        ctx.save_for_backward(sh, fea, query_index, q_row_ptr, q_alpha_index)
        ctx.num_queries = num_queries
        return out.to(fea.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        sh, fea, query_index, q_row_ptr, q_alpha_index = ctx.saved_tensors
        num_queries = ctx.num_queries

        d_sh = d_fea = None
        if torch.is_grad_enabled():
            d_sh, d_fea = EdgeDegreeOuterBackward_Fn.apply(
                grad_out,
                sh,
                fea,
                query_index,
                q_row_ptr,
                q_alpha_index,
                num_queries,
            )
        else:
            go = grad_out.contiguous().to(torch.float32)
            if ctx.needs_input_grad[0]:
                d_sh = _edge_dsh(go, fea, query_index).to(sh.dtype)
            if ctx.needs_input_grad[1]:
                d_fea = _edge_dfea(go, sh, query_index).to(fea.dtype)

        return d_sh, d_fea, None, None, None, None


def prepare_edge_degree_outer_metadata(query_index, num_queries):
    query_index_i32 = query_index.contiguous().to(torch.int32)
    edge_index_i32 = torch.arange(query_index_i32.numel(), device=query_index_i32.device, dtype=torch.int32)
    metadata = prepare_sparse_qk_edge_index_metadata(
        query_index_i32,
        edge_index_i32,
        num_queries,
        edge_index_i32.numel(),
    )
    return metadata 


def edge_degree_outer_triton(edge_sh, edge_fea, query_index, num_queries, metadata=None):
    """
    edge_sh:  [E, L]
    edge_fea: [E, H]
    query_index: [E]
    out: [N, L, H], out[n,l,h] = sum_{e:q[e]=n} edge_sh[e,l] * edge_fea[e,h]
    """
    query_index_i32 = query_index.contiguous().to(torch.int32)
    if metadata is None:
        q_row_ptr, _, q_alpha_index, _, _, _ = prepare_edge_degree_outer_metadata(query_index_i32, num_queries)
    else:
        if isinstance(metadata, (tuple, list)):
            if len(metadata) == 6:
                # Compatible with prepare_sparse_qk_edge_index_metadata.
                q_row_ptr, _, q_alpha_index, _, _, _ = metadata
            else:
                raise ValueError("metadata must be a 6-tuple")
        else:
            raise ValueError("metadata must be a tuple/list")

    return EdgeDegreeOuterEdgeIndex_Fn.apply(
        edge_sh,
        edge_fea,
        query_index_i32,
        q_row_ptr,
        q_alpha_index,
        num_queries,
    )
