# -*- coding: utf-8 -*-


import torch
import triton
import triton.language as tl
_SAFE_TUNED_KEYS = set()

def make_cfgs_edge_qk_csr():
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
def make_cfgs_gather_opt():
    cfgs = []
    for BH in [4, 8, 16, 32]: 
        cfgs.append(triton.Config({'BLOCK_H': BH}, num_warps=4, num_stages=2))
    return cfgs

def make_cfgs_n_tiled():
    cfgs = []
    for BH in (32, 128):          # H tile
        for BK in (4, ):        # K tile
            for BD in (8, 16):        
                
                if BH*BK + BH*BD + BK*BD > 64_000:
                    continue
                for nw in (4, ):
                    for ns in (2, 3):
                        cfgs.append(
                            triton.Config({'BLOCK_H': BH, 'BLOCK_K': BK, 'BLOCK_D': BD},
                                          num_warps=nw, num_stages=ns)
                        )
    
    keyset, out = set(), []
    for c in cfgs:
        t = (c.kwargs['BLOCK_H'], c.kwargs['BLOCK_K'], c.kwargs['BLOCK_D'], c.num_warps, c.num_stages)
        if t not in keyset:
            keyset.add(t); out.append(c)
    return out

AUTOTUNE_CFGS = make_cfgs_n_tiled()

@triton.autotune(configs=make_cfgs_edge_qk_csr(), key=["H", "D"])
@triton.jit
def _edge_qk_fwd_csr_kernel(
    Q, K, 
    Q_ROW_PTR, Q_COL_INDEX, Q_ALPHA_INDEX, 
    OUT,
    N, H, D,
    s_qn, s_qh, s_qd,
    s_kn, s_kh, s_kd,
    s_oe, s_oh,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Grid: (N, ceil(H / BLOCK_H))
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_n >= N:
        return

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # find start and end
    start_ptr = tl.load(Q_ROW_PTR + pid_n)
    end_ptr   = tl.load(Q_ROW_PTR + pid_n + 1)

    if start_ptr >= end_ptr:
        return

    q_ptr = Q + pid_n * s_qn + h_offs[:, None] * s_qh + d_offs[None, :] * s_qd
    q_tile = tl.load(q_ptr, mask=h_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    for ptr in range(start_ptr, end_ptr):
        neighbor_index = tl.load(Q_COL_INDEX + ptr) # Corresponding key node index
        edge_idx       = tl.load(Q_ALPHA_INDEX + ptr) # Global edge index

        k_ptr = K + neighbor_index * s_kn + h_offs[:, None] * s_kh + d_offs[None, :] * s_kd
        k_tile = tl.load(k_ptr, mask=h_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        acc = tl.sum(q_tile * k_tile, axis=1)

        out_ptr = OUT + edge_idx * s_oe + h_offs * s_oh
        tl.store(out_ptr, acc, mask=h_mask)


def _edge_qk_forward_only(query, key, q_row_ptr, q_col_index, q_alpha_index, num_edges):
    assert query.ndim == 3 and key.ndim == 3
    N, H, D = query.shape
    M, Hk, Dk = key.shape
    E = num_edges
    assert H == Hk and D == Dk

    q = query.contiguous()
    k = key.contiguous()

    if E == 0:
        return torch.empty((0, H), device=q.device, dtype=torch.float32)

    out = torch.empty((E, H), device=q.device, dtype=torch.float32)
    block_d = triton.next_power_of_2(D)
    grid = lambda meta: (N, triton.cdiv(H, meta["BLOCK_H"]))
    
    _edge_qk_fwd_csr_kernel[grid](
        q, k, 
        q_row_ptr, q_col_index, q_alpha_index, 
        out,
        N, H, D,
        *q.stride(),
        *k.stride(),
        *out.stride(),
        BLOCK_D=block_d,
    )
    return out

# -----------------------
# Forward: program_id -> n
# -----------------------
@triton.autotune(configs=AUTOTUNE_CFGS, key=['H','Kdim','D'])
@triton.jit
def _fwd_n_tiled_varlen(
    Q, K, IDX, GATE, OUT, DOT, VarLens,
    N, H, Kdim, D,
    s_qn, s_qh, s_qd,
    s_kn, s_kh, s_kd,
    s_in, s_ik,
    s_gn, s_gk, s_gh,
    s_on, s_ok, s_oh,
    s_dn, s_dk, s_dh,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    if pid_n >= N: 
        return

    k0 = 0
    Len = tl.load(VarLens + pid_n, mask=True)
    while k0 < Len:
        k_offs = k0 + tl.arange(0, BLOCK_K)             # [BK]
        k_mask = k_offs < Kdim
        # idx[n, k] -> [BK]
        idx_ptr  = IDX + pid_n * s_in + k_offs * s_ik
        idx_vals = tl.load(idx_ptr, mask=k_mask, other=0).to(tl.int32)

        h0 = 0
        while h0 < H:
            h_offs = h0 + tl.arange(0, BLOCK_H)         # [BH]
            h_mask = h_offs < H

            acc = tl.zeros((BLOCK_K, BLOCK_H), dtype=tl.float32)  # [BK,BH]

            d0 = 0
            while d0 < D:
                d_offs = d0 + tl.arange(0, BLOCK_D)     # [BD]
                d_mask = d_offs < D

                # q[n,h,d] -> [BH,BD]
                q_ptr  = Q + pid_n*s_qn + h_offs[:, None]*s_qh + d_offs[None, :]*s_qd
                q_tile = tl.load(q_ptr, mask=h_mask[:, None] & d_mask[None, :], other=0.).to(tl.float32)

                # k[idx,h,d] -> [BK,BH,BD]
                k_base = idx_vals[:, None, None]*s_kn + h_offs[None, :, None]*s_kh + d_offs[None, None]*s_kd
                k_ptr  = K + k_base
                kmask  = k_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]
                k_tile = tl.load(k_ptr, mask=kmask, other=0.).to(tl.float32)

                acc += tl.sum(k_tile * q_tile[None, :, :], axis=2)   # [BK,BH]
                d0  += BLOCK_D

            
            dot_ptr = DOT + pid_n*s_dn + k_offs[:, None]*s_dk + h_offs[None, :]*s_dh
            tl.store(dot_ptr, acc, mask=k_mask[:, None] & h_mask[None, :])

            
            gate_ptr  = GATE + pid_n*s_gn + k_offs[:, None]*s_gk + h_offs[None, :]*s_gh
            gate_tile = tl.load(gate_ptr, mask=k_mask[:, None] & h_mask[None, :], other=0.).to(tl.float32)
            out_tile  = (acc * gate_tile).to(OUT_DTYPE)
            out_ptr   = OUT + pid_n*s_on + k_offs[:, None]*s_ok + h_offs[None, :]*s_oh
            tl.store(out_ptr, out_tile, mask=k_mask[:, None] & h_mask[None, :])

            h0 += BLOCK_H
        k0 += BLOCK_K


# -----------------------
# Forward: program_id -> n
# -----------------------
@triton.autotune(configs=AUTOTUNE_CFGS, key=['H','Kdim','D'])
@triton.jit
def _fwd_n_tiled_varlen(
    Q, K, IDX, GATE, OUT, DOT, VarLens,
    N, H, Kdim, D,
    s_qn, s_qh, s_qd,
    s_kn, s_kh, s_kd,
    s_in, s_ik,
    s_gn, s_gk, s_gh,
    s_on, s_ok, s_oh,
    s_dn, s_dk, s_dh,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    if pid_n >= N: 
        return

    k0 = 0
    Len = tl.load(VarLens + pid_n, mask=True)
    while k0 < Len:
        k_offs = k0 + tl.arange(0, BLOCK_K)             # [BK]
        k_mask = k_offs < Kdim
        # idx[n, k] -> [BK]
        idx_ptr  = IDX + pid_n * s_in + k_offs * s_ik
        idx_vals = tl.load(idx_ptr, mask=k_mask, other=0).to(tl.int32)

        h0 = 0
        while h0 < H:
            h_offs = h0 + tl.arange(0, BLOCK_H)         # [BH]
            h_mask = h_offs < H

            acc = tl.zeros((BLOCK_K, BLOCK_H), dtype=tl.float32)  # [BK,BH]

            d0 = 0
            while d0 < D:
                d_offs = d0 + tl.arange(0, BLOCK_D)     # [BD]
                d_mask = d_offs < D

                # q[n,h,d] -> [BH,BD]
                q_ptr  = Q + pid_n*s_qn + h_offs[:, None]*s_qh + d_offs[None, :]*s_qd
                q_tile = tl.load(q_ptr, mask=h_mask[:, None] & d_mask[None, :], other=0.).to(tl.float32)

                # k[idx,h,d] -> [BK,BH,BD]
                k_base = idx_vals[:, None, None]*s_kn + h_offs[None, :, None]*s_kh + d_offs[None, None]*s_kd
                k_ptr  = K + k_base
                kmask  = k_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]
                k_tile = tl.load(k_ptr, mask=kmask, other=0.).to(tl.float32)

                acc += tl.sum(k_tile * q_tile[None, :, :], axis=2)   # [BK,BH]
                d0  += BLOCK_D

            
            dot_ptr = DOT + pid_n*s_dn + k_offs[:, None]*s_dk + h_offs[None, :]*s_dh
            tl.store(dot_ptr, acc, mask=k_mask[:, None] & h_mask[None, :])

            
            gate_ptr  = GATE + pid_n*s_gn + k_offs[:, None]*s_gk + h_offs[None, :]*s_gh
            gate_tile = tl.load(gate_ptr, mask=k_mask[:, None] & h_mask[None, :], other=0.).to(tl.float32)
            out_tile  = (acc * gate_tile).to(OUT_DTYPE)
            out_ptr   = OUT + pid_n*s_on + k_offs[:, None]*s_ok + h_offs[None, :]*s_oh
            tl.store(out_ptr, out_tile, mask=k_mask[:, None] & h_mask[None, :])

            h0 += BLOCK_H
        k0 += BLOCK_K

# -----------------------

# dQ[n,h,d] = Σ_k α[n,k,h] * K[idx[n,k],h,d]
# -----------------------
@triton.autotune(configs=AUTOTUNE_CFGS, key=['H','Kdim','D'])
@triton.jit
def _bwdq_n_tiled(
    K, IDX, ALPHA, dQ,
    N, H, Kdim, D,
    s_kn, s_kh, s_kd,
    s_in, s_ik,
    s_an, s_ak, s_ah,
    s_qn, s_qh, s_qd,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    if pid_n >= N: 
        return

    h0 = 0
    while h0 < H:
        h_offs = h0 + tl.arange(0, BLOCK_H)             # [BH]
        h_mask = h_offs < H

        d0 = 0
        while d0 < D:
            d_offs = d0 + tl.arange(0, BLOCK_D)         # [BD]
            d_mask = d_offs < D
            acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)  # [BH,BD]

            k0 = 0
            while k0 < Kdim:
                k_offs = k0 + tl.arange(0, BLOCK_K)     # [BK]
                k_mask = k_offs < Kdim

                # α[n,k,h] -> [BK,BH]
                a_ptr  = ALPHA + pid_n*s_an + k_offs[:, None]*s_ak + h_offs[None, :]*s_ah
                a_tile = tl.load(a_ptr, mask=k_mask[:, None] & h_mask[None, :], other=0.).to(tl.float32)

                # K[idx,h,d] -> [BK,BH,BD]
                idx_ptr  = IDX + pid_n*s_in + k_offs*s_ik
                idx_vals = tl.load(idx_ptr, mask=k_mask, other=0).to(tl.int32)
                k_base   = idx_vals[:, None, None]*s_kn + h_offs[None, :, None]*s_kh + d_offs[None, None]*s_kd
                k_ptr    = K + k_base
                kmask    = k_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]
                k_tile   = tl.load(k_ptr, mask=kmask, other=0.).to(tl.float32)

                acc += tl.sum(k_tile * a_tile[:, :, None], axis=0)  # [BH,BD]
                k0  += BLOCK_K

            dq_ptr = dQ + pid_n*s_qn + h_offs[:, None]*s_qh + d_offs[None, :]*s_qd
            tl.store(dq_ptr, acc, mask=h_mask[:, None] & d_mask[None, :])

            d0 += BLOCK_D
        h0 += BLOCK_H

@triton.autotune(configs=AUTOTUNE_CFGS, key=['H','Kdim','D'])
@triton.jit
def _bwdq_n_tiled_vlen(
    K, IDX, ALPHA, dQ, VarLens,
    N, H, Kdim, D,
    s_kn, s_kh, s_kd,
    s_in, s_ik,
    s_an, s_ak, s_ah,
    s_qn, s_qh, s_qd,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    if pid_n >= N: 
        return

    h0 = 0
    Len = tl.load(VarLens + pid_n, mask = True).to(tl.int32)
    while h0 < H:
        h_offs = h0 + tl.arange(0, BLOCK_H)             # [BH]
        h_mask = h_offs < H

        d0 = 0
        while d0 < D:
            d_offs = d0 + tl.arange(0, BLOCK_D)         # [BD]
            d_mask = d_offs < D
            acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)  # [BH,BD]

            k0 = 0
            while k0 < Len:
                k_offs = k0 + tl.arange(0, BLOCK_K)     # [BK]
                k_mask = k_offs < Len

                # α[n,k,h] -> [BK,BH]
                a_ptr  = ALPHA + pid_n*s_an + k_offs[:, None]*s_ak + h_offs[None, :]*s_ah
                a_tile = tl.load(a_ptr, mask=k_mask[:, None] & h_mask[None, :], other=0.).to(tl.float32)

                # K[idx,h,d] -> [BK,BH,BD]
                idx_ptr  = IDX + pid_n*s_in + k_offs*s_ik
                idx_vals = tl.load(idx_ptr, mask=k_mask, other=0).to(tl.int32)
                k_base   = idx_vals[:, None, None]*s_kn + h_offs[None, :, None]*s_kh + d_offs[None, None]*s_kd
                k_ptr    = K + k_base
                kmask    = k_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]
                k_tile   = tl.load(k_ptr, mask=kmask, other=0.).to(tl.float32)

                acc += tl.sum(k_tile * a_tile[:, :, None], axis=0)  # [BH,BD]
                k0  += BLOCK_K

            dq_ptr = dQ + pid_n*s_qn + h_offs[:, None]*s_qh + d_offs[None, :]*s_qd
            tl.store(dq_ptr, acc, mask=h_mask[:, None] & d_mask[None, :])

            d0 += BLOCK_D
        h0 += BLOCK_H



@triton.autotune(configs=make_cfgs_gather_opt(), key=['H', 'D'])
@triton.jit
def _bwdk_n_tiled_spmv(
    Row_Ptr, Col_Idx, Alpha_Indices,
    Alpha, Q_ptr, dK_ptr,
    s_a_row, s_a_h,            
    s_q_n, s_q_h, s_q_d,       
    s_dk_m, s_dk_h, s_dk_d,    
    H, D,
    BLOCK_H: tl.constexpr,     
    BLOCK_D: tl.constexpr      
):
    # Grid: (M, ceil(H / BLOCK_H))
    pid_m = tl.program_id(0)
    pid_h_blk = tl.program_id(1)

    offs_h = pid_h_blk * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    ptr_start = tl.load(Row_Ptr + pid_m)
    ptr_end   = tl.load(Row_Ptr + pid_m + 1)

    # 3. 2D Accumulator [BLOCK_H, BLOCK_D]
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # 4. Gather Loop
    for ptr in range(ptr_start, ptr_end):
        # A. Metadata Load (Broadcast)
        current_n = tl.load(Col_Idx + ptr)
        original_edge_idx = tl.load(Alpha_Indices + ptr)

        # B. Alpha [BLOCK_H] (Vectorized Load if stride is 1)
        a_ptr = Alpha + original_edge_idx * s_a_row + offs_h * s_a_h
        alpha_vec = tl.load(a_ptr, mask=mask_h, other=0.0).to(tl.float32)

        # C. Q [BLOCK_H, BLOCK_D] (2D Load)
        q_ptr_base = Q_ptr + current_n * s_q_n 
        q_ptrs = q_ptr_base + offs_h[:, None] * s_q_h + offs_d[None, :] * s_q_d
        q_tile = tl.load(q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        # D. Broadcasting & FMA
        acc += alpha_vec[:, None] * q_tile

    # 5. Store
    dk_ptrs = dK_ptr + pid_m * s_dk_m + offs_h[:, None] * s_dk_h + offs_d[None, :] * s_dk_d
    tl.store(dk_ptrs, acc, mask=mask_h[:, None] & mask_d[None, :])

# -----------------------
class SparseQK_N_Tiled_Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, idx, gate, row_ptr = None, col_idx = None, idx_value=None, var_lens = None):
        assert query.is_cuda and key.is_cuda and gate.is_cuda and idx.is_cuda
        idx = idx.int()

        assert idx.dtype == torch.int32
        N, H, D = query.shape
        Kdim = idx.shape[1]
        q, k, g = query.contiguous(), key.contiguous(), gate.contiguous()

        out = torch.zeros((N, Kdim, H), dtype=q.dtype, device=q.device)
        dot = torch.zeros((N, Kdim, H), dtype=torch.float32, device=q.device)

        s_qn, s_qh, s_qd = q.stride()
        s_kn, s_kh, s_kd = k.stride()
        s_in, s_ik       = idx.stride()
        s_gn, s_gk, s_gh = g.stride()
        s_on, s_ok, s_oh = out.stride()
        s_dn, s_dk, s_dh = dot.stride()

        if out.dtype == torch.float16: out_dtype = tl.float16
        elif out.dtype == torch.bfloat16: out_dtype = tl.bfloat16
        elif out.dtype == torch.float32: out_dtype = tl.float32
        else: raise TypeError(f"unsupported dtype: {out.dtype}")

        grid = lambda meta: (N,)
        _fwd_n_tiled_varlen[grid](
            q, k, idx, g, out, dot, var_lens,
            N, H, Kdim, D,
            s_qn, s_qh, s_qd,
            s_kn, s_kh, s_kd,
            s_in, s_ik,
            s_gn, s_gk, s_gh,
            s_on, s_ok, s_oh,
            s_dn, s_dk, s_dh,
            OUT_DTYPE=out_dtype,
        )
        ctx.save_for_backward(q, k, idx, g,dot, row_ptr, col_idx, idx_value, var_lens)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, idx, g,dot, row_ptr, col_idx, idx_value, var_lens = ctx.saved_tensors
        N, H, D = q.shape
        M = k.shape[0]
        Kdim = idx.shape[1]
        device = q.device

        
        need_graph = torch.is_grad_enabled()
        if need_graph:
            
            dQ, dK, _, dgate, _ = BackwardAsFunction_Fn.apply(grad_out,q,k,idx , g,dot, var_lens, row_ptr, col_idx, idx_value)
        else:
            # raise ValueError("errrrrrrror")
            
            alpha = (g * grad_out ).to(torch.float32)
            dQ = torch.zeros_like(q, dtype=torch.float32)
            dK = torch.zeros_like(k, dtype=torch.float32)
            # print(dK.shape)
            s_kn, s_kh, s_kd = k.stride()
            s_in, s_ik       = idx.stride()
            s_an, s_ak, s_ah = alpha.stride()
            s_qn, s_qh, s_qd = dQ.stride()
            s_qn0, s_qh0, s_qd0 = q.stride()
            s_kn0, s_kh0, s_kd0 = dK.stride()

            grid = lambda meta: (N,)
            _bwdq_n_tiled_vlen[grid](
                k, idx, alpha, dQ, var_lens,
                N, H, Kdim, D,
                s_kn, s_kh, s_kd,
                s_in, s_ik,
                s_an, s_ak, s_ah,
                s_qn, s_qh, s_qd,
            )
            
            BLOCK_D = triton.next_power_of_2(D)
            
            grid_opt = lambda meta: (M, triton.cdiv(H, meta['BLOCK_H']))
            _bwdk_n_tiled_spmv[grid_opt](
                row_ptr, col_idx, idx_value,
                alpha, q, dK,
                H, 1,
                *q.stride(), *dK.stride(),
                H, D,
                BLOCK_D=BLOCK_D
            )
            qk = torch.zeros((N, Kdim, H), device=q.device, dtype=torch.float32)

            _dot_only_n_tiled_vlen[(N,)](
                q, k, idx, qk, var_lens,
                N, H, Kdim, D,
                *q.stride(), *k.stride(), *idx.stride(), *qk.stride(),
            )
            dgate  = (qk * grad_out.to(torch.float32)).to(g.dtype)

        return dQ.to(q.dtype), dK.to(k.dtype), None, dgate, None, None, None, None


@triton.autotune(configs=AUTOTUNE_CFGS, key=['H','Kdim','D'])
@triton.jit
def _dot_only_n_tiled_vlen(
    Q, K, IDX, DOT, VarLens,
    N, H, Kdim, D,
    s_qn, s_qh, s_qd,
    s_kn, s_kh, s_kd,
    s_in, s_ik,
    s_dn, s_dk, s_dh,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    if pid_n >= N: 
        return

    k0 = 0
    Len = tl.load(VarLens + pid_n)
    while k0 < Len:
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < Len
        idx_ptr  = IDX + pid_n*s_in + k_offs*s_ik
        idx_vals = tl.load(idx_ptr, mask=k_mask, other=0).to(tl.int32)
        h0 = 0
        while h0 < H:
            h_offs = h0 + tl.arange(0, BLOCK_H)
            h_mask = h_offs < H
            acc = tl.zeros((BLOCK_K, BLOCK_H), dtype=tl.float32)
            d0 = 0
            while d0 < D:
                d_offs = d0 + tl.arange(0, BLOCK_D)
                d_mask = d_offs < D
                q_ptr  = Q + pid_n*s_qn + h_offs[:,None]*s_qh + d_offs[None,:]*s_qd
                q_tile = tl.load(q_ptr, mask=h_mask[:,None] & d_mask[None,:], other=0.).to(tl.float32)
                k_base = idx_vals[:,None,None]*s_kn + h_offs[None,:,None]*s_kh + d_offs[None,None]*s_kd
                k_ptr  = K + k_base
                kmask  = k_mask[:,None,None] & h_mask[None,:,None] & d_mask[None,None,:]
                k_tile = tl.load(k_ptr, mask=kmask, other=0.).to(tl.float32)
                acc += tl.sum(k_tile * q_tile[None,:,:], axis=2)
                d0  += BLOCK_D
            dot_ptr = DOT + pid_n*s_dn + k_offs[:,None]*s_dk + h_offs[None,:]*s_dh
            tl.store(dot_ptr, acc, mask=k_mask[:,None] & h_mask[None,:])
            h0 += BLOCK_H
        k0 += BLOCK_K

def make_cfgs_edge_qk():
    cfgs = []
    for bh in (32, 64, 128):
        for bd in (8, 16, 32):
            if bh * bd > 64_000:
                continue
            for nw in (4, 8):
                for ns in (2, 3):
                    cfgs.append(
                        triton.Config(
                            {"BLOCK_H": bh, "BLOCK_D": bd},
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return cfgs
def _edge_index_is_sorted(index):
    if index.numel() <= 1:
        return True
    return not (index[1:] < index[:-1]).any().item()


def _make_row_ptr_from_sorted_index(sorted_index, num_rows):
    counts = torch.bincount(sorted_index.to(torch.int64), minlength=num_rows)
    row_ptr = torch.empty(num_rows + 1, device=sorted_index.device, dtype=torch.int32)
    row_ptr[0] = 0
    row_ptr[1:] = counts.cumsum(dim=0).to(torch.int32)
    return row_ptr


def _prepare_segment_metadata(row_index, col_index, num_rows):
    if row_index.numel() == 0:
        empty = torch.empty((0,), device=row_index.device, dtype=torch.int32)
        row_ptr = torch.zeros((num_rows + 1,), device=row_index.device, dtype=torch.int32)
        return row_ptr, empty, empty

    if _edge_index_is_sorted(row_index):
        edge_perm = torch.arange(row_index.numel(), device=row_index.device, dtype=torch.int32)
        sorted_row_index = row_index
        sorted_col_index = col_index
    else:
        sort_perm = torch.argsort(row_index, stable=True)
        edge_perm = sort_perm.to(torch.int32)
        sorted_row_index = row_index.index_select(0, sort_perm)
        sorted_col_index = col_index.index_select(0, sort_perm)

    row_ptr = _make_row_ptr_from_sorted_index(sorted_row_index, num_rows)
    return row_ptr, sorted_col_index.contiguous(), edge_perm.contiguous()


def prepare_sparse_qk_edge_index_metadata(query_index, neighbor_index, num_queries, num_keys):
    query_index_i32 = query_index.contiguous().to(torch.int32)
    neighbor_index_i32 = neighbor_index.contiguous().to(torch.int32)

    q_row_ptr, q_col_index, q_alpha_index = _prepare_segment_metadata(
        query_index_i32, neighbor_index_i32, num_queries
    )
    key_row_ptr, key_col_index, key_alpha_index = _prepare_segment_metadata(
        neighbor_index_i32, query_index_i32, num_keys
    )
    return (
        q_row_ptr,
        q_col_index,
        q_alpha_index,
        key_row_ptr,
        key_col_index,
        key_alpha_index,
    )


def _edge_grouped_reduce_triton(alpha, x, row_ptr, col_index, alpha_index, out_rows):
    H = x.shape[1]
    D = x.shape[2]
    out = torch.zeros((out_rows, H, D), device=x.device, dtype=torch.float32)
    block_d = triton.next_power_of_2(D)
    grid = lambda meta: (out_rows, triton.cdiv(H, meta["BLOCK_H"]))
    _bwdk_n_tiled_spmv[grid](
        row_ptr,
        col_index,
        alpha_index,
        alpha,
        x,
        out,
        *alpha.stride(),
        *x.stride(),
        *out.stride(),
        H, D,
        BLOCK_D=block_d,
    )
    return out

class SparseQKEdgeIndex_Fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        query_index,
        neighbor_index,
        q_row_ptr,
        q_col_index,
        q_alpha_index,
        key_row_ptr,
        key_col_index,
        key_alpha_index,
    ):
        # out = _edge_qk_forward_only(query, key, query_index, neighbor_index)
        E = query_index.numel()
        out = _edge_qk_forward_only(query, key, q_row_ptr, q_col_index, q_alpha_index, E)
        ctx.save_for_backward(
            query,
            key,
            query_index,
            neighbor_index,
            q_row_ptr,
            q_col_index,
            q_alpha_index,
            key_row_ptr,
            key_col_index,
            key_alpha_index,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            query,
            key,
            query_index,
            neighbor_index,
            q_row_ptr,
            q_col_index,
            q_alpha_index,
            key_row_ptr,
            key_col_index,
            key_alpha_index,
        ) = ctx.saved_tensors

        dquery = dkey = None

        need_graph = torch.is_grad_enabled()
        if need_graph:
            dquery, dkey = SparseQKEdgeBackward_Fn.apply(
                grad_out,
                query,
                key,
                query_index,
                neighbor_index,
                q_row_ptr,
                q_col_index,
                q_alpha_index,
                key_row_ptr,
                key_col_index,
                key_alpha_index,
            )
        else:
            grad = grad_out.contiguous().to(torch.float32)
            if ctx.needs_input_grad[0]:
                dquery = _edge_grouped_reduce_triton(
                    grad, key, q_row_ptr, q_col_index, q_alpha_index, query.shape[0]
                ).to(query.dtype)
            if ctx.needs_input_grad[1]:
                dkey = _edge_grouped_reduce_triton(
                    grad, query, key_row_ptr, key_col_index, key_alpha_index, key.shape[0]
                ).to(key.dtype)

        return dquery, dkey, None, None, None, None, None, None, None, None


class SparseQKEdgeBackward_Fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_out,
        query,
        key,
        query_index,
        neighbor_index,
        q_row_ptr,
        q_col_index,
        q_alpha_index,
        key_row_ptr,
        key_col_index,
        key_alpha_index,
    ):
        grad = grad_out.contiguous().to(torch.float32)
        dquery = _edge_grouped_reduce_triton(
            grad, key, q_row_ptr, q_col_index, q_alpha_index, query.shape[0]
        )
        dkey = _edge_grouped_reduce_triton(
            grad, query, key_row_ptr, key_col_index, key_alpha_index, key.shape[0]
        )

        ctx.save_for_backward(
            grad,
            query,
            key,
            query_index,
            neighbor_index,
            q_row_ptr,
            q_col_index,
            q_alpha_index,
            key_row_ptr,
            key_col_index,
            key_alpha_index,
        )
        return dquery.to(query.dtype), dkey.to(key.dtype)

    @staticmethod
    def backward(ctx, gg_dquery, gg_dkey):
        (
            grad,
            query,
            key,
            query_index,
            neighbor_index,
            q_row_ptr,
            q_col_index,
            q_alpha_index,
            key_row_ptr,
            key_col_index,
            key_alpha_index,
        ) = ctx.saved_tensors

        d_grad_out = d_query = d_key = None
        gg_dquery_f = None if gg_dquery is None else gg_dquery.contiguous().to(torch.float32)
        gg_dkey_f = None if gg_dkey is None else gg_dkey.contiguous().to(torch.float32)

        if ctx.needs_input_grad[0]:
            d_grad_out = torch.zeros_like(grad, dtype=torch.float32)
            E = query_index.numel()
            if gg_dquery_f is not None:
                d_grad_out = d_grad_out + _edge_qk_forward_only(
                    gg_dquery_f, key, q_row_ptr, q_col_index, q_alpha_index, E
                )
            if gg_dkey_f is not None:
                d_grad_out = d_grad_out + _edge_qk_forward_only(
                    query, gg_dkey_f, q_row_ptr, q_col_index, q_alpha_index, E
                )
            d_grad_out = d_grad_out.to(grad.dtype)

        if ctx.needs_input_grad[1] and gg_dkey_f is not None:
            d_query = _edge_grouped_reduce_triton(
                grad, gg_dkey_f, q_row_ptr, q_col_index, q_alpha_index, query.shape[0]
            ).to(query.dtype)

        if ctx.needs_input_grad[2] and gg_dquery_f is not None:
            d_key = _edge_grouped_reduce_triton(
                grad, gg_dquery_f, key_row_ptr, key_col_index, key_alpha_index, key.shape[0]
            ).to(key.dtype)

        return d_grad_out, d_query, d_key, None, None, None, None, None, None, None, None


def sparse_qk_edge_index_triton(query, key, query_index, neighbor_index, metadata=None):
    """
    query: [N, H, D]
    key:   [M, H, D]
    query_index: [E]
    neighbor_index: [E]
    out:   [E, H], where out[e, h] = dot(query[query_index[e], h], key[neighbor_index[e], h])

    Backward uses grouped reductions:
    1. dquery groups by query_index (reusing sorted input when available)
    2. dkey groups by neighbor_index after a stable reorder
    3. metadata can be precomputed and reused across calls
    """
    if metadata is None:
        metadata = prepare_sparse_qk_edge_index_metadata(
            query_index, neighbor_index, query.shape[0], key.shape[0]
        )
    else:
        if len(metadata) != 6:
            raise ValueError("metadata must be a 6-tuple from prepare_sparse_qk_edge_index_metadata")

    if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad):
        return SparseQKEdgeIndex_Fn.apply(query, key, query_index, neighbor_index, *metadata)

    q_row_ptr, q_col_index, q_alpha_index, _, _, _ = metadata
    return _edge_qk_forward_only(
        query,
        key,
        q_row_ptr,
        q_col_index,
        q_alpha_index,
        query_index.numel(),
    )


class BackwardAsFunction_Fn(torch.autograd.Function):
    """
    Treat the backward kernel as a normal function:
        inputs:  (idx[N,K], k[N,H,D], q[N,H,D], tmp_out[N,K,H], g[N,K,H])
        outputs: (tmp_Q[N,H,D], tmp_K[N,H,D], None, tmp_gate[N,K,H], None)
    Then implement backward for that function, i.e. the second derivative of
    the original operator, and keep using Triton.
    """
    @staticmethod
    def forward(ctx, tmp_out,q,k,idx,  g,dot, var_len, row_ptr, col_idx, idx_value):
        assert idx.is_cuda and k.is_cuda and q.is_cuda and tmp_out.is_cuda and g.is_cuda
        idx = idx.to(torch.int32, copy=False)
        N, H, D = q.shape
        M       = k.shape[0]
        Kdim    = idx.shape[1]

        w = tmp_out.to(torch.float32) * g.to(torch.float32)
        
        tmp_Q = torch.zeros_like(q, dtype=torch.float32)
        _bwdq_n_tiled_vlen[(N,)](
            k, idx, w, tmp_Q, var_len,
            N, H, Kdim, D,
            *k.stride(), *idx.stride(), *w.stride(), *tmp_Q.stride(),
        )

        
        tmp_K = torch.zeros_like(k, dtype=torch.float32)
        BLOCK_D = triton.next_power_of_2(D)
        grid_opt = lambda meta: (M, triton.cdiv(H, meta['BLOCK_H']))
        _bwdk_n_tiled_spmv[grid_opt](
            row_ptr, col_idx, idx_value,
            w, q, tmp_K,
            H, 1,
            *q.stride(), *tmp_K.stride(),
            H, D,
            BLOCK_D=BLOCK_D
        )
        # ---- tmp_gate = tmp_out * dot(q, K[idx]) 
        qk = dot
        tmp_gate = (tmp_out.to(torch.float32) * qk)

        
        ctx.save_for_backward(tmp_out,q,k, idx,  g, qk, var_len,row_ptr, col_idx, idx_value)
        return tmp_Q.to(q.dtype), tmp_K.to(k.dtype), None, tmp_gate.to(tmp_out.dtype), None

    @staticmethod
    def backward(ctx, gg_tmp_Q, gg_tmp_K, _, gg_tmp_gate, __):
        """
        Upstream gradients:
          gg_tmp_Q:    [N,H,D]
          gg_tmp_K:    [N,H,D]
          gg_tmp_gate: [N,K,H]
        Need to return:
          (d_idx=None, d_k2, d_q2, d_tmp_out2, d_g2)
        """
        tmp_out, q, k,  idx,  g, qk,var_len,row_ptr, col_idx, idx_value = ctx.saved_tensors
        N, H, D = q.shape
        M       = k.shape[0]
        Kdim    = idx.shape[1]

        
        ggQ = gg_tmp_Q.to(torch.float32) if gg_tmp_Q is not None else torch.zeros_like(q, dtype=torch.float32)
        ggK = gg_tmp_K.to(torch.float32) if gg_tmp_K is not None else torch.zeros_like(k, dtype=torch.float32)
        ggG = gg_tmp_gate.to(torch.float32) if gg_tmp_gate is not None else torch.zeros((N, Kdim, H), device=q.device, dtype=torch.float32)

        # ------------------------------
        
        #    tmp_Q: dw_Q[n,k,h] = <ggQ[n,h,:], K[idx[n,k],h,:]>
        #    tmp_K: dw_K[n,k,h] = <ggK[idx[n,k],h,:], Q[n,h,:]>
        # ------------------------------
        dw = torch.zeros((N, Kdim, H), device=q.device, dtype=torch.float32)

        # dw_Q = dot(ggQ, K[idx])
        _dot_only_n_tiled_vlen[(N,)](
            ggQ.contiguous(), k, idx, dw, var_len,
            N, H, Kdim, D,
            *ggQ.stride(), *k.stride(), *idx.stride(), *dw.stride(),
        )
        
        tmp = torch.zeros_like(dw)
        _dot_only_n_tiled_vlen[(N,)](
            q, ggK.contiguous(), idx, tmp, var_len,
            N, H, Kdim, D,
            *q.stride(), *ggK.stride(), *idx.stride(), *tmp.stride(),
        )
        dw.add_(tmp)

        # ------------------------------
        
        # ------------------------------
        beta = (tmp_out.to(torch.float32) * ggG )  # [N,K,H]

        # ------------------------------
        
        
        
        # ------------------------------
        
        w = (tmp_out.to(torch.float32) * g.to(torch.float32))

        d_q2_a = torch.zeros_like(q, dtype=torch.float32)
        _bwdq_n_tiled_vlen[(N,)](
            ggK.contiguous(), idx, w, d_q2_a, var_len,      # alpha = w, K = ggK
            N, H, Kdim, D,
            *ggK.stride(), *idx.stride(), *w.stride(), *d_q2_a.stride(),
        )
        d_q2_b = torch.zeros_like(q, dtype=torch.float32)
        _bwdq_n_tiled_vlen[(N,)](
            k, idx, beta, d_q2_b, var_len,                # alpha = beta, K = k
            N, H, Kdim, D,
            *k.stride(), *idx.stride(), *beta.stride(), *d_q2_b.stride(),
        )
        d_q2 = d_q2_a + d_q2_b

        # ------------------------------
        
        
        
        # ------------------------------
        d_k2 = torch.zeros_like(k, dtype=torch.float32)
        BLOCK_D = triton.next_power_of_2(D)

        grid_opt = lambda meta: (M, triton.cdiv(H, meta['BLOCK_H']))
        d_k2_a = torch.zeros_like(k, dtype=torch.float32)
        _bwdk_n_tiled_spmv[grid_opt](
            row_ptr, col_idx, idx_value, 
            w, ggQ, d_k2_a,
            H, 1,
            *ggQ.stride(), *d_k2.stride(),
            H, D,
            BLOCK_D=BLOCK_D
        )

        d_k2_b = torch.zeros_like(k, dtype=torch.float32)
        _bwdk_n_tiled_spmv[grid_opt](
            row_ptr, col_idx, idx_value, 
            beta, q, d_k2_b,
            H, 1,
            *q.stride(), *d_k2.stride(),
            H, D,
            BLOCK_D=BLOCK_D
        )
        d_k2 = d_k2_a + d_k2_b
        # ------------------------------
        
        
        
        # ------------------------------
        d_tmp_out2 = dw * (g.to(torch.float32)) + (ggG * qk.to(torch.float32))
        d_g2       = dw * (tmp_out.to(torch.float32))

        
        d_k2  = d_k2.to(k.dtype)
        d_q2  = d_q2.to(q.dtype)
        d_tmp_out2 = d_tmp_out2.to(tmp_out.dtype)
        d_g2  = d_g2.to(g.dtype)

        
        return d_tmp_out2, d_q2, d_k2,None,  d_g2,None,None,None,None, None


def sparse_qk_triton_n_tiled(query, key, idx, gate, row_ptr, col_idx, idx_value, var_lens):
    return SparseQK_N_Tiled_Fn.apply(query, key, idx, gate, row_ptr, col_idx, idx_value, var_lens)
