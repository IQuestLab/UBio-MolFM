# -*- coding: utf-8 -*-
import torch
import triton
import triton.language as tl

from .triton_sparse_qk_autograd import _prepare_segment_metadata


def _make_cfgs():
    cfgs = []
    for block_h in (32, 64, 128):
        for block_e in (8, 16, 32, 64):
            for nw in (4, 8):
                for ns in (2, 3):
                    cfgs.append(
                        triton.Config(
                            {"BLOCK_H": block_h, "BLOCK_E": block_e},
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return cfgs


AUTOTUNE_CFGS = _make_cfgs()


@triton.autotune(configs=AUTOTUNE_CFGS, key=["H"])
@triton.jit
def _fwd_alpha_sum_edge_index(
    ROW_PTR,
    ALPHA_INDEX,
    ALPHA,
    OUT,
    N,
    H,
    s_ae,
    s_ah,
    s_on,
    s_oh,
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_n >= N:
        return

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    row_start = tl.load(ROW_PTR + pid_n).to(tl.int32)
    row_end = tl.load(ROW_PTR + pid_n + 1).to(tl.int32)

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    e0 = row_start
    while e0 < row_end:
        e_offs = e0 + tl.arange(0, BLOCK_E)
        e_mask = e_offs < row_end
        alpha_ids = tl.load(ALPHA_INDEX + e_offs, mask=e_mask, other=0).to(tl.int32)
        a_ptr = ALPHA + alpha_ids[:, None] * s_ae + h_offs[None, :] * s_ah
        a_tile = tl.load(a_ptr, mask=e_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(a_tile, axis=0)
        e0 += BLOCK_E

    out_ptr = OUT + pid_n * s_on + h_offs * s_oh
    tl.store(out_ptr, acc, mask=h_mask)


@triton.autotune(configs=AUTOTUNE_CFGS, key=["H"])
@triton.jit
def _bwd_alpha_sum_edge_index(
    GRAD_OUT,
    QUERY_INDEX,
    GRAD_ALPHA,
    E,
    H,
    s_gn,
    s_gh,
    s_qe,
    s_ae,
    s_ah,
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_h = tl.program_id(1)

    e_offs = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    e_mask = e_offs < E
    h_mask = h_offs < H

    query_ids = tl.load(QUERY_INDEX + e_offs * s_qe, mask=e_mask, other=0).to(tl.int32)
    go_ptr = GRAD_OUT + query_ids[:, None] * s_gn + h_offs[None, :] * s_gh
    go_tile = tl.load(go_ptr, mask=e_mask[:, None] & h_mask[None, :], other=0.0)

    ga_ptr = GRAD_ALPHA + e_offs[:, None] * s_ae + h_offs[None, :] * s_ah
    tl.store(ga_ptr, go_tile, mask=e_mask[:, None] & h_mask[None, :])


def prepare_sparse_alpha_sum_edge_index_metadata(query_index, num_queries):
    query_index_i32 = query_index.contiguous().to(torch.int32)
    row_ptr, _, alpha_index = _prepare_segment_metadata(
        query_index_i32,
        query_index_i32,
        num_queries,
    )
    return row_ptr, alpha_index


def _alpha_sum_edge_index_forward_only(alpha, row_ptr, alpha_index, num_queries):
    alpha = alpha.contiguous()
    out = torch.zeros((num_queries, alpha.shape[1]), device=alpha.device, dtype=torch.float32)
    grid = lambda meta: (num_queries, triton.cdiv(alpha.shape[1], meta["BLOCK_H"]))
    _fwd_alpha_sum_edge_index[grid](
        row_ptr,
        alpha_index,
        alpha,
        out,
        num_queries,
        alpha.shape[1],
        *alpha.stride(),
        *out.stride(),
    )
    return out.to(alpha.dtype)


class SparseAlphaSumEdgeIndexFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, query_index, row_ptr, alpha_index, num_queries):
        out = _alpha_sum_edge_index_forward_only(alpha, row_ptr, alpha_index, num_queries)
        ctx.save_for_backward(
            query_index.contiguous().to(torch.int32),
            row_ptr,
            alpha_index,
        )
        ctx.alpha_shape = alpha.shape
        ctx.num_queries = num_queries
        return out

    @staticmethod
    def backward(ctx, grad_out):
        query_index, row_ptr, alpha_index = ctx.saved_tensors
        if torch.is_grad_enabled():
            grad_alpha = SparseAlphaSumEdgeIndexBackwardFn.apply(
                grad_out,
                query_index,
                row_ptr,
                alpha_index,
                ctx.alpha_shape[0],
                ctx.alpha_shape[1],
                ctx.num_queries,
            )
        else:
            grad_out = grad_out.contiguous()
            e_count, h_count = ctx.alpha_shape
            grad_alpha = torch.empty((e_count, h_count), device=grad_out.device, dtype=grad_out.dtype)
            grid = lambda meta: (triton.cdiv(e_count, meta["BLOCK_E"]), triton.cdiv(h_count, meta["BLOCK_H"]))
            _bwd_alpha_sum_edge_index[grid](
                grad_out,
                query_index,
                grad_alpha,
                e_count,
                h_count,
                *grad_out.stride(),
                query_index.stride(0),
                *grad_alpha.stride(),
            )
        return grad_alpha, None, None, None, None


class SparseAlphaSumEdgeIndexBackwardFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_out, query_index, row_ptr, alpha_index, e_count, h_count, num_queries):
        grad_out = grad_out.contiguous()
        grad_alpha = torch.empty((e_count, h_count), device=grad_out.device, dtype=grad_out.dtype)
        grid = lambda meta: (triton.cdiv(e_count, meta["BLOCK_E"]), triton.cdiv(h_count, meta["BLOCK_H"]))
        _bwd_alpha_sum_edge_index[grid](
            grad_out,
            query_index,
            grad_alpha,
            e_count,
            h_count,
            *grad_out.stride(),
            query_index.stride(0),
            *grad_alpha.stride(),
        )
        ctx.save_for_backward(row_ptr, alpha_index)
        ctx.num_queries = num_queries
        return grad_alpha

    @staticmethod
    def backward(ctx, gg_grad_alpha):
        row_ptr, alpha_index = ctx.saved_tensors
        d_grad_out = None
        if ctx.needs_input_grad[0]:
            d_grad_out = _alpha_sum_edge_index_forward_only(
                gg_grad_alpha,
                row_ptr,
                alpha_index,
                ctx.num_queries,
            )
        return d_grad_out, None, None, None, None, None, None


def sparse_alpha_sum_edge_index_triton(alpha, query_index, num_queries, metadata=None):
    if metadata is None:
        metadata = prepare_sparse_alpha_sum_edge_index_metadata(query_index, num_queries)
    row_ptr, alpha_index = metadata
    if torch.is_grad_enabled() and alpha.requires_grad:
        return SparseAlphaSumEdgeIndexFn.apply(alpha, query_index, row_ptr, alpha_index, num_queries)
    return _alpha_sum_edge_index_forward_only(alpha, row_ptr, alpha_index, num_queries)
