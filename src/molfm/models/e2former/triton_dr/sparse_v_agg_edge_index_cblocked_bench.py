# -*- coding: utf-8 -*-
from time import perf_counter
import torch
import triton
import triton.language as tl
from loguru import logger

from .sparse_v_agg_lastdim_autograd import (
    prepare_sparse_v_agg_edge_index_metadata,
)


def _make_cfgs_cblocked():
    cfgs = []
    for bh in (16, 32, 64):
        for bc in (4,8,16,32,64):
            for nw in (4, 8):
                for ns in (2, 3):
                    cfgs.append(
                        triton.Config(
                            {"BLOCK_H": bh, "BLOCK_C": bc},
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    seen, out = set(), []
    for c in cfgs:
        key = (c.kwargs["BLOCK_H"], c.kwargs["BLOCK_C"], c.num_warps, c.num_stages)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out





@triton.autotune(configs=_make_cfgs_cblocked(), key=["H", "C"])
@triton.jit
def _edge_dot_cblocked_csr_kernel(
    GO, VALUE,
    ROW_PTR, COL_INDEX, ALPHA_INDEX,
    OUT,
    N, H, C,
    s_gn, s_gc, s_gh,
    s_vn, s_vc, s_vh,
    s_oe, s_oh,
    BLOCK_H: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_n >= N:
        return

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    ptr_start = tl.load(ROW_PTR + pid_n)
    ptr_end = tl.load(ROW_PTR + pid_n + 1)
    if ptr_start >= ptr_end:
        return

    c0 = 0
    while c0 < C:
        c_idx = c0 + c_offs
        cm = c_idx < C
        go_ptr = GO + pid_n * s_gn + c_idx[:, None] * s_gc + h_offs[None, :] * s_gh
        go_tile = tl.load(go_ptr, mask=cm[:, None] & h_mask[None, :], other=0.0).to(tl.float32)

        for ptr in range(ptr_start, ptr_end):
            value_row = tl.load(COL_INDEX + ptr)
            edge_idx = tl.load(ALPHA_INDEX + ptr)

            v_ptr = VALUE + value_row * s_vn + c_idx[:, None] * s_vc + h_offs[None, :] * s_vh
            v_tile = tl.load(v_ptr, mask=cm[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
            acc = tl.sum(go_tile * v_tile, axis=0)

            out_ptr = OUT + edge_idx * s_oe + h_offs * s_oh
            if c0 == 0:
                tl.store(out_ptr, acc, mask=h_mask)
            else:
                prev = tl.load(out_ptr, mask=h_mask, other=0.0).to(tl.float32)
                tl.store(out_ptr, prev + acc, mask=h_mask)
        c0 += BLOCK_C


@triton.autotune(configs=_make_cfgs_cblocked(), key=["H", "C"])
@triton.jit
def _edge_grouped_reduce_cblocked_kernel(
    ALPHA, X,
    ROW_PTR, COL_INDEX, ALPHA_INDEX,
    OUT,
    OUT_ROWS, H, C,
    s_ae, s_ah,
    s_xn, s_xc, s_xh,
    s_on, s_oc, s_oh,
    BLOCK_H: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    if pid_n >= OUT_ROWS:
        return

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    h_mask = h_offs < H
    c_mask = c_offs < C

    ptr_start = tl.load(ROW_PTR + pid_n)
    ptr_end = tl.load(ROW_PTR + pid_n + 1)

    acc = tl.zeros((BLOCK_C, BLOCK_H), dtype=tl.float32)
    for ptr in range(ptr_start, ptr_end):
        src_row = tl.load(COL_INDEX + ptr)
        edge_idx = tl.load(ALPHA_INDEX + ptr)

        a_ptr = ALPHA + edge_idx * s_ae + h_offs * s_ah
        alpha_vec = tl.load(a_ptr, mask=h_mask, other=0.0).to(tl.float32)

        x_ptr = X + src_row * s_xn + c_offs[:, None] * s_xc + h_offs[None, :] * s_xh
        x_tile = tl.load(x_ptr, mask=c_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
        acc += x_tile * alpha_vec[None, :]

    out_ptr = OUT + pid_n * s_on + c_offs[:, None] * s_oc + h_offs[None, :] * s_oh
    tl.store(out_ptr, acc, mask=c_mask[:, None] & h_mask[None, :])


def _edge_dot_cblocked(go, value, row_ptr, col_index, alpha_index):
    n = go.shape[0]
    h_dim = go.shape[2]
    e = col_index.numel()
    out = torch.zeros((e, h_dim), device=go.device, dtype=torch.float32)
    if e == 0:
        return out

    grid = lambda meta: (n, triton.cdiv(h_dim, meta["BLOCK_H"]))
    _edge_dot_cblocked_csr_kernel[grid](
        go,
        value,
        row_ptr,
        col_index,
        alpha_index,
        out,
        n,
        h_dim,
        go.shape[1],
        go.stride(0),
        go.stride(1),
        go.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        out.stride(0),
        out.stride(1),
    )
    return out


def _edge_grouped_reduce_cblocked(alpha, x, row_ptr, col_index, alpha_index):
    out_rows = row_ptr.numel() - 1
    out = torch.zeros((out_rows, x.shape[1], x.shape[2]), device=x.device, dtype=torch.float32)
    if col_index.numel() == 0:
        return out

    grid = lambda meta: (
        out_rows,
        triton.cdiv(x.shape[1], meta["BLOCK_C"]),
        triton.cdiv(x.shape[2], meta["BLOCK_H"]),
    )
    _edge_grouped_reduce_cblocked_kernel[grid](
        alpha,
        x,
        row_ptr,
        col_index,
        alpha_index,
        out,
        out_rows,
        x.shape[2],
        x.shape[1],
        alpha.stride(0),
        alpha.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    return out


@triton.autotune(configs=_make_cfgs_cblocked(), key=["H", "C"])
@triton.jit
def _edge_index_cblocked_kernel(
    VALUE, ALPHA,
    ROW_PTR, COL_INDEX, ALPHA_INDEX,
    OUT,
    N, H, C,
    s_vn, s_vc, s_vh,
    s_an, s_ah,
    s_on, s_oc, s_oh,
    BLOCK_H: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c = tl.program_id(2)

    if pid_n >= N:
        return

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    h_mask = h_offs < H
    c_mask = c_offs < C

    ptr_start = tl.load(ROW_PTR + pid_n)
    ptr_end = tl.load(ROW_PTR + pid_n + 1)

    acc = tl.zeros((BLOCK_H, BLOCK_C), dtype=tl.float32)

    for ptr in range(ptr_start, ptr_end):
        value_row = tl.load(COL_INDEX + ptr)
        edge_idx = tl.load(ALPHA_INDEX + ptr)

        a_ptr = ALPHA + edge_idx * s_an + h_offs * s_ah
        alpha_vec = tl.load(a_ptr, mask=h_mask, other=0.0).to(tl.float32)

        v_ptrs = VALUE + value_row * s_vn + c_offs[None, :] * s_vc + h_offs[:, None] * s_vh
        v_tile = tl.load(v_ptrs, mask=h_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)

        acc += alpha_vec[:, None] * v_tile

    out_ptrs = OUT + pid_n * s_on + c_offs[None, :] * s_oc + h_offs[:, None] * s_oh
    tl.store(out_ptrs, acc, mask=h_mask[:, None] & c_mask[None, :])


def _sparse_v_agg_edge_index_triton_cblocked_forward(
    value, alpha, query_index, value_index, num_queries, metadata=None
):
    """
    value: [M, C, H]
    alpha: [E, H]
    query_index: [E]
    value_index: [E]
    out: [N, C, H], with
      out[query_index[e], c, h] += alpha[e, h] * value[value_index[e], c, h]
    """
    if value.ndim != 3:
        raise ValueError("value must be [M, C, H]")
    if alpha.ndim != 2:
        raise ValueError("alpha must be [E, H]")

    query_index = query_index.contiguous().to(torch.int32)
    value_index = value_index.contiguous().to(torch.int32)

    if metadata is None:
        metadata = prepare_sparse_v_agg_edge_index_metadata(
            query_index, value_index, num_queries, value.shape[0]
        )
    elif len(metadata) != 6:
        raise ValueError("metadata must be a 6-tuple from prepare_sparse_v_agg_edge_index_metadata")

    q_row_ptr, q_col_index, q_alpha_index, _, _, _ = metadata

    v = value.contiguous()
    a = alpha.contiguous().to(torch.float32)

    c_dim = v.shape[1]
    h_dim = v.shape[2]
    out = torch.zeros((num_queries, c_dim, h_dim), device=v.device, dtype=torch.float32)
    if q_col_index.numel() == 0:
        return out.to(v.dtype)

    grid = lambda meta: (
        num_queries,
        triton.cdiv(h_dim, meta["BLOCK_H"]),
        triton.cdiv(c_dim, meta["BLOCK_C"]),
    )
    _edge_index_cblocked_kernel[grid](
        v, a,
        q_row_ptr, q_col_index, q_alpha_index,
        out,
        num_queries, h_dim, c_dim,
        v.stride(0), v.stride(1), v.stride(2),
        a.stride(0), a.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
    )
    return out.to(v.dtype)



class SparseVAggEdgeIndexCBlocked_Fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        value,
        alpha,
        query_index,
        value_index,
        q_row_ptr,
        q_col_index,
        q_alpha_index,
        v_row_ptr,
        v_col_index,
        v_alpha_index,
    ):
        # value: [M, C, H] -> out: [N, C, H]
        out = _sparse_v_agg_edge_index_triton_cblocked_forward(
            value, alpha, query_index, value_index, q_row_ptr.numel() - 1,
            metadata=(q_row_ptr, q_col_index, q_alpha_index, v_row_ptr, v_col_index, v_alpha_index),
        )
        ctx.save_for_backward(
            value, alpha, query_index, value_index,
            q_row_ptr, q_col_index, q_alpha_index,
            v_row_ptr, v_col_index, v_alpha_index,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            v, a, query_index, value_index,
            q_row_ptr, q_col_index, q_alpha_index,
            v_row_ptr, v_col_index, v_alpha_index,
        ) = ctx.saved_tensors

        d_value = d_alpha = None
        if torch.is_grad_enabled():
            d_value, d_alpha = SparseVAggEdgeIndexCBlocked_Backward_Fn.apply(
                grad_out,
                v,
                a,
                query_index,
                value_index,
                q_row_ptr,
                q_col_index,
                q_alpha_index,
                v_row_ptr,
                v_col_index,
                v_alpha_index,
            )
        else:
            go_f = grad_out.contiguous().to(torch.float32)
            if ctx.needs_input_grad[0]:
                d_value = _edge_grouped_reduce_cblocked(
                    a, go_f, v_row_ptr, v_col_index, v_alpha_index
                ).to(v.dtype)
            if ctx.needs_input_grad[1]:
                d_alpha = _edge_dot_cblocked(
                    go_f, v, q_row_ptr, q_col_index, q_alpha_index
                ).to(a.dtype)
        return d_value, d_alpha, None, None, None, None, None, None, None, None


class SparseVAggEdgeIndexCBlocked_Backward_Fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_out,
        value,
        alpha,
        query_index,
        value_index,
        q_row_ptr,
        q_col_index,
        q_alpha_index,
        v_row_ptr,
        v_col_index,
        v_alpha_index,
    ):
        go = grad_out.contiguous().to(torch.float32)
        v = value.contiguous()
        a = alpha.contiguous()

        d_alpha = _edge_dot_cblocked(go, v, q_row_ptr, q_col_index, q_alpha_index)
        d_value = _edge_grouped_reduce_cblocked(a, go, v_row_ptr, v_col_index, v_alpha_index)

        ctx.save_for_backward(
            go, v, a, query_index, value_index,
            q_row_ptr, q_col_index, q_alpha_index,
            v_row_ptr, v_col_index, v_alpha_index,
        )
        return d_value.to(v.dtype), d_alpha.to(a.dtype)

    @staticmethod
    def backward(ctx, gg_dvalue, gg_dalpha):
        (
            go, v, a, query_index, value_index,
            q_row_ptr, q_col_index, q_alpha_index,
            v_row_ptr, v_col_index, v_alpha_index,
        ) = ctx.saved_tensors

        d_go = d_value = d_alpha = None
        gg_dvalue_f = None if gg_dvalue is None else gg_dvalue.contiguous().to(torch.float32)
        gg_dalpha_f = None if gg_dalpha is None else gg_dalpha.contiguous().to(torch.float32)

        if ctx.needs_input_grad[0]:
            d_go = torch.zeros_like(go, dtype=torch.float32)
            if gg_dvalue_f is not None:
                d_go = d_go + _edge_grouped_reduce_cblocked(
                    a, gg_dvalue_f, q_row_ptr, q_col_index, q_alpha_index
                )
            if gg_dalpha_f is not None:
                d_go = d_go + _edge_grouped_reduce_cblocked(
                    gg_dalpha_f, v, q_row_ptr, q_col_index, q_alpha_index
                )
            d_go = d_go.to(go.dtype)

        if ctx.needs_input_grad[1] and gg_dalpha_f is not None:
            d_value = _edge_grouped_reduce_cblocked(
                gg_dalpha_f, go, v_row_ptr, v_col_index, v_alpha_index
            ).to(v.dtype)

        if ctx.needs_input_grad[2] and gg_dvalue_f is not None:
            d_alpha = _edge_dot_cblocked(
                go, gg_dvalue_f, q_row_ptr, q_col_index, q_alpha_index
            ).to(a.dtype)

        return d_go, d_value, d_alpha, None, None, None, None, None, None, None, None


def sparse_v_agg_edge_index_triton_cblocked(
    value, alpha, query_index, value_index, num_queries, metadata=None
):
    """
    Autograd-capable wrapper (1st + 2nd order).
    value: [M, C, H]
    alpha: [E, H]
    out: [N, C, H]
    """
    query_index = query_index.contiguous().to(torch.int32)
    value_index = value_index.contiguous().to(torch.int32)

    if metadata is None:
        metadata = prepare_sparse_v_agg_edge_index_metadata(
            query_index, value_index, num_queries, value.shape[0]
        )
    elif len(metadata) != 6:
        raise ValueError("metadata must be a 6-tuple from prepare_sparse_v_agg_edge_index_metadata")

    return SparseVAggEdgeIndexCBlocked_Fn.apply(
        value, alpha, query_index, value_index, *metadata
    )
