# -*- coding: utf-8 -*-


from time import perf_counter
import torch
import triton
import triton.language as tl
_SAFE_TUNED_KEYS = set()

# -----------------------

# -----------------------
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

# -----------------------
# -----------------------
@triton.autotune(configs=AUTOTUNE_CFGS, key=['H','Kdim','D'])
@triton.jit
def _fwd_n_tiled(
    Q, K, IDX, GATE, OUT, DOT,
    scale,
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
    while k0 < Kdim:
        k_offs = k0 + tl.arange(0, BLOCK_K)             # [BK]
        k_mask = k_offs < Kdim
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

                q_ptr  = Q + pid_n*s_qn + h_offs[:, None]*s_qh + d_offs[None, :]*s_qd
                q_tile = tl.load(q_ptr, mask=h_mask[:, None] & d_mask[None, :], other=0.).to(tl.float32)

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
            out_tile  = (acc * gate_tile * scale).to(OUT_DTYPE)
            out_ptr   = OUT + pid_n*s_on + k_offs[:, None]*s_ok + h_offs[None, :]*s_oh
            tl.store(out_ptr, out_tile, mask=k_mask[:, None] & h_mask[None, :])

            h0 += BLOCK_H
        k0 += BLOCK_K

# -----------------------

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

                a_ptr  = ALPHA + pid_n*s_an + k_offs[:, None]*s_ak + h_offs[None, :]*s_ah
                a_tile = tl.load(a_ptr, mask=k_mask[:, None] & h_mask[None, :], other=0.).to(tl.float32)

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

# -----------------------

# -----------------------
@triton.autotune(configs=AUTOTUNE_CFGS, key=['H','Kdim','D'],reset_to_zero=["dK"])
@triton.jit
def _bwdk_n_tiled(
    Q, IDX, ALPHA, dK,
    N, H, Kdim, D,
    s_qn, s_qh, s_qd,
    s_in, s_ik,
    s_an, s_ak, s_ah,
    s_kn, s_kh, s_kd,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    if pid_n >= N:
        return

    k0 = 0
    while k0 < Kdim:
        k_offs = k0 + tl.arange(0, BLOCK_K)             # [BK]
        k_mask = k_offs < Kdim

        h0 = 0
        while h0 < H:
            h_offs = h0 + tl.arange(0, BLOCK_H)         # [BH]
            h_mask = h_offs < H

            a_ptr  = ALPHA + pid_n*s_an + k_offs[:, None]*s_ak + h_offs[None, :]*s_ah
            a_tile = tl.load(a_ptr, mask=k_mask[:, None] & h_mask[None, :], other=0.).to(tl.float32)  # [BK,BH]

            idx_ptr  = IDX + pid_n*s_in + k_offs*s_ik
            idx_vals = tl.load(idx_ptr, mask=k_mask, other=0).to(tl.int32)                             # [BK]

            d0 = 0
            while d0 < D:
                d_offs = d0 + tl.arange(0, BLOCK_D)     # [BD]
                d_mask = d_offs < D

                q_ptr  = Q + pid_n*s_qn + h_offs[:, None]*s_qh + d_offs[None, :]*s_qd
                q_tile = tl.load(q_ptr, mask=h_mask[:, None] & d_mask[None, :], other=0.).to(tl.float32)

                contrib = a_tile[:, :, None] * q_tile[None, :, :]

                
                dk_ptrs = dK + idx_vals[:, None, None]*s_kn + h_offs[None, :, None]*s_kh + d_offs[None, None]*s_kd
                kmask   = k_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]
                tl.atomic_add(dk_ptrs, contrib, mask=kmask)

                d0 += BLOCK_D
            h0 += BLOCK_H
        k0 += BLOCK_K

# -----------------------

# -----------------------
class SparseQK_N_Tiled_Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, idx, gate, scale: float):
        assert query.is_cuda and key.is_cuda and gate.is_cuda and idx.is_cuda
        idx = idx.int()

        assert idx.dtype == torch.int32
        N, H, D = query.shape
        Kdim = idx.shape[1]
        q, k, g = query.contiguous(), key.contiguous(), gate.contiguous()

        out = torch.empty((N, Kdim, H), dtype=q.dtype, device=q.device)
        dot = torch.empty((N, Kdim, H), dtype=torch.float32, device=q.device)

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

        _fwd_n_tiled[grid](
            q, k, idx, g, out, dot,
            scale,
            N, H, Kdim, D,
            s_qn, s_qh, s_qd,
            s_kn, s_kh, s_kd,
            s_in, s_ik,
            s_gn, s_gk, s_gh,
            s_on, s_ok, s_oh,
            s_dn, s_dk, s_dh,
            OUT_DTYPE=out_dtype,
        )

        ctx.save_for_backward(q, k, idx, g,dot)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, idx, g,dot = ctx.saved_tensors
        scale = ctx.scale
        N, H, D = q.shape
        Kdim = idx.shape[1]
        device = q.device

        
        need_graph = torch.is_grad_enabled()
        if need_graph:
            
            dQ, dK, _, dgate, _ = BackwardAsFunction_Fn.apply(grad_out,q,k,idx , g, scale,dot)
        else:
            
            alpha = (g * grad_out * scale).to(torch.float32)
            dQ = torch.empty_like(q, dtype=torch.float32)
            dK = torch.zeros_like(k, dtype=torch.float32)

            s_kn, s_kh, s_kd = k.stride()
            s_in, s_ik       = idx.stride()
            s_an, s_ak, s_ah = alpha.stride()
            s_qn, s_qh, s_qd = dQ.stride()
            s_qn0, s_qh0, s_qd0 = q.stride()
            s_kn0, s_kh0, s_kd0 = dK.stride()

            grid = lambda meta: (N,)
            _bwdq_n_tiled[grid](
                k, idx, alpha, dQ,
                N, H, Kdim, D,
                s_kn, s_kh, s_kd,
                s_in, s_ik,
                s_an, s_ak, s_ah,
                s_qn, s_qh, s_qd,
            )

            _bwdk_n_tiled[grid](
                q, idx, alpha, dK,
                N, H, Kdim, D,
                s_qn0, s_qh0, s_qd0,
                s_in, s_ik,
                s_an, s_ak, s_ah,
                s_kn0, s_kh0, s_kd0,
            )
            qk = torch.empty((N, Kdim, H), device=q.device, dtype=torch.float32)
            _dot_only_n_tiled[(N,)](
                q, k, idx, qk,
                N, H, Kdim, D,
                *q.stride(), *k.stride(), *idx.stride(), *qk.stride(),
            )
            dgate  = (qk * grad_out.to(torch.float32) * scale).to(g.dtype)

        return dQ.to(q.dtype), dK.to(k.dtype), None, dgate, None




@triton.autotune(configs=AUTOTUNE_CFGS, key=['H','Kdim','D'])
@triton.jit
def _dot_only_n_tiled(
    Q, K, IDX, DOT,
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
    while k0 < Kdim:
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < Kdim
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


class BackwardAsFunction_Fn(torch.autograd.Function):
    """
        inputs:  (idx[N,K], k[N,H,D], q[N,H,D], tmp_out[N,K,H], g[N,K,H], scale)
        outputs: (tmp_Q[N,H,D], tmp_K[N,H,D], None, tmp_gate[N,K,H], None)
    """
    @staticmethod
    def forward(ctx, tmp_out,q,k,idx,  g, scale: float,dot):
        assert idx.is_cuda and k.is_cuda and q.is_cuda and tmp_out.is_cuda and g.is_cuda
        idx = idx.to(torch.int32, copy=False)
        N, H, D = q.shape
        Kdim    = idx.shape[1]

        w = (tmp_out.to(torch.float32) * g.to(torch.float32) * float(scale))  # [N,K,H]

        
        tmp_Q = torch.empty_like(q, dtype=torch.float32)
        _bwdq_n_tiled[(N,)](
            k, idx, w, tmp_Q,
            N, H, Kdim, D,
            *k.stride(), *idx.stride(), *w.stride(), *tmp_Q.stride(),
        )

        
        tmp_K = torch.zeros_like(k, dtype=torch.float32)
        _bwdk_n_tiled[(N,)](
            q, idx, w, tmp_K,
            N, H, Kdim, D,
            *q.stride(), *idx.stride(), *w.stride(), *tmp_K.stride(),
        )

        qk = dot
        tmp_gate = (tmp_out.to(torch.float32) * qk * float(scale))

        
        ctx.save_for_backward(tmp_out,q,k, idx,  g, qk)
        ctx.scale = float(scale)
        return tmp_Q.to(q.dtype), tmp_K.to(k.dtype), None, tmp_gate.to(tmp_out.dtype), None

    @staticmethod
    def backward(ctx, gg_tmp_Q, gg_tmp_K, _, gg_tmp_gate, __):
        """
          gg_tmp_Q:    [N,H,D]
          gg_tmp_K:    [N,H,D]
          gg_tmp_gate: [N,K,H]
          (d_idx=None, d_k2, d_q2, d_tmp_out2, d_g2, d_scale=None)
        """
        tmp_out,q,k,  idx,  g, qk = ctx.saved_tensors
        scale = ctx.scale
        N, H, D = q.shape
        Kdim    = idx.shape[1]

        
        ggQ = gg_tmp_Q.to(torch.float32) if gg_tmp_Q is not None else torch.zeros_like(q, dtype=torch.float32)
        ggK = gg_tmp_K.to(torch.float32) if gg_tmp_K is not None else torch.zeros_like(k, dtype=torch.float32)
        ggG = gg_tmp_gate.to(torch.float32) if gg_tmp_gate is not None else torch.zeros((N, Kdim, H), device=q.device, dtype=torch.float32)

        # ------------------------------
        
        # ------------------------------
        dw = torch.empty((N, Kdim, H), device=q.device, dtype=torch.float32)

        _dot_only_n_tiled[(N,)](
            ggQ.contiguous(), k, idx, dw,
            N, H, Kdim, D,
            *ggQ.stride(), *k.stride(), *idx.stride(), *dw.stride(),
        )
        
        tmp = torch.empty_like(dw)
        _dot_only_n_tiled[(N,)](
            q, ggK.contiguous(), idx, tmp,
            N, H, Kdim, D,
            *q.stride(), *ggK.stride(), *idx.stride(), *tmp.stride(),
        )
        dw.add_(tmp)

        # ------------------------------
        
        # ------------------------------
        beta = (tmp_out.to(torch.float32) * ggG * float(scale))  # [N,K,H]

        # ------------------------------
        
        
        
        # ------------------------------
        
        w = (tmp_out.to(torch.float32) * g.to(torch.float32) * float(scale))

        d_q2_a = torch.empty_like(q, dtype=torch.float32)
        _bwdq_n_tiled[(N,)](
            ggK.contiguous(), idx, w, d_q2_a,      # alpha = w, K = ggK
            N, H, Kdim, D,
            *ggK.stride(), *idx.stride(), *w.stride(), *d_q2_a.stride(),
        )
        d_q2_b = torch.empty_like(q, dtype=torch.float32)
        _bwdq_n_tiled[(N,)](
            k, idx, beta, d_q2_b,                 # alpha = beta, K = k
            N, H, Kdim, D,
            *k.stride(), *idx.stride(), *beta.stride(), *d_q2_b.stride(),
        )
        d_q2 = d_q2_a + d_q2_b

        # ------------------------------
        
        
        
        # ------------------------------
        d_k2 = torch.zeros_like(k, dtype=torch.float32)
        _bwdk_n_tiled[(N,)](
            ggQ.contiguous(), idx, w, d_k2,       # alpha = w, Q = ggQ
            N, H, Kdim, D,
            *ggQ.stride(), *idx.stride(), *w.stride(), *d_k2.stride(),
        )
        _bwdk_n_tiled[(N,)](
            q, idx, beta, d_k2,                   # alpha = beta, Q = q
            N, H, Kdim, D,
            *q.stride(), *idx.stride(), *beta.stride(), *d_k2.stride(),
        )

        # ------------------------------
        
        
        
        # ------------------------------
        d_tmp_out2 = dw * (g.to(torch.float32) * float(scale)) + (ggG * qk.to(torch.float32) * float(scale))
        d_g2       = dw * (tmp_out.to(torch.float32) * float(scale))

        
        d_k2  = d_k2.to(k.dtype)
        d_q2  = d_q2.to(q.dtype)
        d_tmp_out2 = d_tmp_out2.to(tmp_out.dtype)
        d_g2  = d_g2.to(g.dtype)

        
        return d_tmp_out2, d_q2, d_k2,None,  d_g2, None,None




def sparse_qk_triton_n_tiled(query, key, idx, gate, scale=0.1):
    return SparseQK_N_Tiled_Fn.apply(query, key, idx, gate, scale)
