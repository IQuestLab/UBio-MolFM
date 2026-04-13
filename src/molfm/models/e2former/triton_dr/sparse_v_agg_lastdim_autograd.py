# -*- coding: utf-8 -*-
# Strategy: program_id -> (n, h_block, c_block); inside one kernel, sum over all K
# Problem setting:
#   out[n,h,c] = sum_k alpha[n,k,h] * value[idx[n,k], h, c]
# Shapes:
#   value: [N,H,C], alpha: [N,K,H] (broadcast over C), idx: [N,K]
# Triton: forward + backward(dAlpha, dValue) + benchmark + optional debug check

from time import perf_counter
import torch
import triton
import triton.language as tl
from .triton_sparse_qk_autograd import prepare_sparse_qk_edge_index_metadata
# ===== autotune-safe for dV (drop-in) =====
import torch, triton

_SAFE_TUNED_KEYS = set()


def make_cfgs_edge_dot_csr():
    cfgs = []
    for bh in (16, 32, 64):
        for nw in (4, 8):
            for ns in (2, 3):
                cfgs.append(
                    triton.Config(
                        {"BLOCK_H": bh},
                        num_warps=nw,
                        num_stages=ns,
                    )
                )
    return cfgs

def make_cfgs_edge_dot():
    cfgs = []
    for bh in (32, 64, 128):
        for bc in (8, 16, 32):
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

@triton.autotune(configs=make_cfgs_edge_dot_csr(), key=["H", "C"])
@triton.jit
def _edge_dot_hc_csr_kernel(
    X, Y, 
    Q_ROW_PTR, Q_COL_INDEX, Q_ALPHA_INDEX, 
    OUT,
    N, H, C,
    s_xn, s_xh, s_xc,
    s_yn, s_yh, s_yc,
    s_oe, s_oh,
    BLOCK_H: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Grid: (N, ceil(H / BLOCK_H))
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    if pid_n >= N:
        return

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    start_ptr = tl.load(Q_ROW_PTR + pid_n)
    end_ptr   = tl.load(Q_ROW_PTR + pid_n + 1)

    if start_ptr >= end_ptr:
        return

    x_ptr = X + pid_n * s_xn + h_offs[:, None] * s_xh + c_offs[None, :] * s_xc
    x_tile = tl.load(x_ptr, mask=h_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)

    for ptr in range(start_ptr, end_ptr):
        value_row = tl.load(Q_COL_INDEX + ptr) 
        edge_idx  = tl.load(Q_ALPHA_INDEX + ptr)

        # load Value (Y)
        y_ptr = Y + value_row * s_yn + h_offs[:, None] * s_yh + c_offs[None, :] * s_yc
        y_tile = tl.load(y_ptr, mask=h_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)

        # C dim
        acc = tl.sum(x_tile * y_tile, axis=1)

        out_ptr = OUT + edge_idx * s_oe + h_offs * s_oh
        tl.store(out_ptr, acc, mask=h_mask)


def _edge_dot_hc_csr(x, y, q_row_ptr, q_col_index, q_alpha_index):
    n = x.shape[0]
    e = q_col_index.numel()
    out = torch.empty((e, x.shape[1]), device=x.device, dtype=torch.float32)
    if e == 0:
        return out
        
    block_c = triton.next_power_of_2(x.shape[2])
    grid = lambda meta: (n, triton.cdiv(x.shape[1], meta["BLOCK_H"]))
    
    _edge_dot_hc_csr_kernel[grid](
        x, y, 
        q_row_ptr, q_col_index, q_alpha_index, 
        out,
        n, x.shape[1], x.shape[2],
        *x.stride(),
        *y.stride(),
        *out.stride(),
        BLOCK_C=block_c,
    )
    return out
# ----------------- autotune space -----------------
def make_cfgs():
    cfgs = []
    for BH in (32,):
        for BC in (4,16):
            for BK in (4, 16):   
                for nw in (4, ):
                    for ns in (2, 3):
                        cfgs.append(
                            triton.Config({'BLOCK_H': BH, 'BLOCK_C': BC, 'BLOCK_K': BK},
                                          num_warps=nw, num_stages=ns)
                        )
    
    seen, out = set(), []
    for c in cfgs:
        key = (c.kwargs['BLOCK_H'], c.kwargs['BLOCK_C'], c.kwargs['BLOCK_K'], c.num_warps, c.num_stages)
        if key not in seen:
            seen.add(key); out.append(c)
    return out

AUTOTUNE_CFGS = make_cfgs()

# ----------------- Backward: dValue (atomics) -----------------
# convert previous kernel to spmv
def make_cfgs_bwd_values():
    cfgs = []
    # A larger BLOCK_H leads to a higher metadata reuse rate, 
    # but also increases register pressure.
    for BH in [4, 8, 16, 32]: 
        cfgs.append(triton.Config({'BLOCK_H': BH}, num_warps=4, num_stages=2))
    return cfgs

@triton.autotune(configs=make_cfgs_bwd_values(), key=['H', 'C'])
@triton.jit
def _bwd_value_n_spmv_kernel(
    # Pointers
    Row_Ptr, Col_Idx,       # CSR Structure
    Alpha_Indices,          # [NNZ]
    Alpha,                  # [N, K, H]
    GO,                     # Dense Input [N, H, C]
    DV_Out,                 # Output [M, H, C]
    
    # Strides
    stride_alpha_nk, stride_alpha_h, 
    stride_go_n, stride_go_h, stride_go_c, 
    stride_dv_m, stride_dv_h, stride_dv_c, 
    
    # Dimensions
    H, C,
    
    # Meta-parameters
    BLOCK_H: tl.constexpr, 
    BLOCK_C: tl.constexpr
):
    # grid：(M, ceil(H / BLOCK_H))
    pid_m = tl.program_id(0)     # Target Row (Key m)
    pid_h_blk = tl.program_id(1) # Head Block Index

    offs_h = pid_h_blk * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # 3.(v2) read csr only once
    ptr_start = tl.load(Row_Ptr + pid_m)
    ptr_end   = tl.load(Row_Ptr + pid_m + 1)

    # 4. init the acc [BLOCK_H, BLOCK_C]
    acc = tl.zeros([BLOCK_H, BLOCK_C], dtype=tl.float32)

    # 5. Reduction Loop
    for ptr in range(ptr_start, ptr_end):
        # A. CSR Index Loading
        current_n = tl.load(Col_Idx + ptr)
        original_edge_idx = tl.load(Alpha_Indices + ptr)

        a_ptr = Alpha + (original_edge_idx * stride_alpha_nk) + (offs_h * stride_alpha_h)
        
        # [BLOCK_H]
        alpha_vec = tl.load(a_ptr, mask=mask_h, other=0.0).to(tl.float32)

        go_base = GO + current_n * stride_go_n
        go_ptrs = go_base + offs_h[:, None] * stride_go_h + offs_c[None, :] * stride_go_c
        
        # [BLOCK_H, BLOCK_C]
        go_tile = tl.load(go_ptrs, mask=mask_h[:, None] & mask_c[None, :], other=0.0).to(tl.float32)

        # D. FMA (Broadcasting)
        # [BLOCK_H, 1] * [BLOCK_H, BLOCK_C] -> [BLOCK_H, BLOCK_C]
        acc += alpha_vec[:, None] * go_tile

    # 6. Store Output [BLOCK_H, BLOCK_C]
    dv_base = DV_Out + pid_m * stride_dv_m
    dv_ptrs = dv_base + offs_h[:, None] * stride_dv_h + offs_c[None, :] * stride_dv_c
    
    tl.store(dv_ptrs, acc, mask=mask_h[:, None] & mask_c[None, :])


def sparse_v_agg_triton_n_tiled_test(value, alpha, idx):
    # input must be N*H*C
    return SparseVAgg_N_Tiled_Fn.apply(value, alpha, idx)


def sparse_v_agg_triton_n_tiled(value, alpha, idx, row_ptr = None, col_idx = None, idx_value = None, var_lens = None):
    value = value.permute(0,2,1)
    return SparseVAgg_N_Tiled_Fn.apply(value, alpha, idx, row_ptr, col_idx, idx_value, var_lens).permute(0,2,1)


def prepare_sparse_v_agg_edge_index_metadata(query_index, value_index, num_queries, num_values):
    return prepare_sparse_qk_edge_index_metadata(
        query_index,
        value_index,
        num_queries,
        num_values,
    )


def _edge_grouped_reduce(alpha, x, row_ptr, col_index, alpha_index):
    out_rows = row_ptr.numel() - 1
    out = torch.zeros((out_rows, x.shape[1], x.shape[2]), device=x.device, dtype=torch.float32)
    block_c = triton.next_power_of_2(x.shape[2])
    grid = lambda meta: (out_rows, triton.cdiv(x.shape[1], meta["BLOCK_H"]))
    _bwd_value_n_spmv_kernel[grid](
        row_ptr,
        col_index,
        alpha_index,
        alpha,
        x,
        out,
        alpha.stride(0), alpha.stride(1),
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        x.shape[1], x.shape[2],
        BLOCK_C=block_c,
    )
    return out


@triton.autotune(configs=make_cfgs_edge_dot(), key=["H", "C"])
@triton.jit
def _edge_dot_hc_kernel(
    X, Y, QUERY_INDEX, VALUE_INDEX, OUT,
    E, H, C,
    s_xn, s_xh, s_xc,
    s_yn, s_yh, s_yc,
    s_qe,
    s_ve,
    s_oe, s_oh,
    BLOCK_H: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_e >= E:
        return

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    query_row = tl.load(QUERY_INDEX + pid_e * s_qe, mask=True, other=0).to(tl.int32)
    value_row = tl.load(VALUE_INDEX + pid_e * s_ve, mask=True, other=0).to(tl.int32)

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    c0 = 0
    while c0 < C:
        c_offs = c0 + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        mask = h_mask[:, None] & c_mask[None, :]
        x_ptr = X + query_row * s_xn + h_offs[:, None] * s_xh + c_offs[None, :] * s_xc
        y_ptr = Y + value_row * s_yn + h_offs[:, None] * s_yh + c_offs[None, :] * s_yc
        x_tile = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
        y_tile = tl.load(y_ptr, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x_tile * y_tile, axis=1)
        c0 += BLOCK_C

    out_ptr = OUT + pid_e * s_oe + h_offs * s_oh
    tl.store(out_ptr, acc, mask=h_mask)

class SparseVAggEdgeBackward_Fn(torch.autograd.Function):
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
        d_alpha = _edge_dot_hc_csr(go, v, q_row_ptr, q_col_index, q_alpha_index)
        d_value = _edge_grouped_reduce(a, go, v_row_ptr, v_col_index, v_alpha_index)

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
                d_go = d_go + _edge_grouped_reduce(a, gg_dvalue_f, q_row_ptr, q_col_index, q_alpha_index)
            if gg_dalpha_f is not None:
                d_go = d_go + _edge_grouped_reduce(gg_dalpha_f, v, q_row_ptr, q_col_index, q_alpha_index)
            d_go = d_go.to(go.dtype)

        if ctx.needs_input_grad[1] and gg_dalpha_f is not None:
            d_value = _edge_grouped_reduce(
                gg_dalpha_f, go, v_row_ptr, v_col_index, v_alpha_index
            ).to(v.dtype)

        if ctx.needs_input_grad[2] and gg_dvalue_f is not None:
            d_alpha = _edge_dot_hc_csr(go, gg_dvalue_f, q_row_ptr, q_col_index, q_alpha_index).to(a.dtype)
        return d_go, d_value, d_alpha, None, None, None, None, None, None, None, None


class SparseVAggEdgeIndex_Fn(torch.autograd.Function):
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
        v = value.contiguous()
        a = alpha.contiguous().to(torch.float32)
        out = _edge_grouped_reduce(a, v, q_row_ptr, q_col_index, q_alpha_index)
        ctx.save_for_backward(
            v, a, query_index, value_index,
            q_row_ptr, q_col_index, q_alpha_index,
            v_row_ptr, v_col_index, v_alpha_index,
        )
        return out.to(v.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (
            v, a, query_index, value_index,
            q_row_ptr, q_col_index, q_alpha_index,
            v_row_ptr, v_col_index, v_alpha_index,
        ) = ctx.saved_tensors

        d_value = d_alpha = None
        if torch.is_grad_enabled():
            d_value, d_alpha = SparseVAggEdgeBackward_Fn.apply(
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
            go = grad_out.contiguous().to(torch.float32)
            if ctx.needs_input_grad[0]:
                d_value = _edge_grouped_reduce(a, go, v_row_ptr, v_col_index, v_alpha_index).to(v.dtype)
            if ctx.needs_input_grad[1]:
                d_alpha = _edge_dot_hc_csr(go, v, q_row_ptr, q_col_index, q_alpha_index).to(a.dtype)
        return d_value, d_alpha, None, None, None, None, None, None, None, None


def sparse_v_agg_edge_index_triton(value, alpha, query_index, value_index, num_queries, metadata=None):
    """
    value: [M, H, C]
    alpha: [E, H]
    query_index: [E]
    value_index: [E]
    out: [N, H, C], with
      out[query_index[e], h, c] += alpha[e, h] * value[value_index[e], h, c]
    """
    query_index = query_index.contiguous().to(torch.int32)
    value_index = value_index.contiguous().to(torch.int32)

    if metadata is None:
        metadata = prepare_sparse_v_agg_edge_index_metadata(
            query_index, value_index, num_queries, value.shape[0]
        )
    elif len(metadata) != 6:
        raise ValueError("metadata must be a 6-tuple from prepare_sparse_v_agg_edge_index_metadata")

    return SparseVAggEdgeIndex_Fn.apply(value, alpha, query_index, value_index, *metadata)
