# -*- coding: utf-8 -*-
# Problem setting:
# Shapes:

from time import perf_counter
import torch
import triton
import triton.language as tl
import torch, triton

_SAFE_TUNED_KEYS = set()




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


@triton.autotune(configs=AUTOTUNE_CFGS, key=['H','C'])
@triton.jit
def _fwd_n_kernel(
    V, IDX, A, OUT,
    N, H, C, Kdim,
    s_vn, s_vh, s_vc,   # strides of value
    s_in, s_ik,         # strides of idx
    s_an, s_ak, s_ah,   # strides of alpha
    s_on, s_oh, s_oc,   # strides of out
    BLOCK_H: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,            
    OUT_DTYPE: tl.constexpr,
):
    pid_n  = tl.program_id(0)
    pid_hb = tl.program_id(1)
    pid_cb = tl.program_id(2)
    if pid_n >= N: return

    h_offs = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)  # [BH]
    c_offs = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)  # [BC]
    h_mask = h_offs < H
    c_mask = c_offs < C

    acc = tl.zeros((BLOCK_H, BLOCK_C), dtype=tl.float32)  # [BH,BC]

    k = 0
    while k < Kdim:
        idx_ptr = IDX + pid_n * s_in + k * s_ik
        m = tl.load(idx_ptr, mask=True, other=0).to(tl.int32)

        a_ptr = A + pid_n*s_an + k*s_ak + h_offs*s_ah
        a_h   = tl.load(a_ptr, mask=h_mask, other=0.0).to(tl.float32)

        v_ptr  = V + m*s_vn + h_offs[:, None]*s_vh + c_offs[None, :]*s_vc
        v_tile = tl.load(v_ptr, mask=h_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)

        acc += a_h[:, None] * v_tile
        k += 1

    out_ptr = OUT + pid_n*s_on + h_offs[:, None]*s_oh + c_offs[None, :]*s_oc
    tl.store(out_ptr, acc.to(OUT_DTYPE), mask=h_mask[:, None] & c_mask[None, :])


@triton.autotune(configs=AUTOTUNE_CFGS, key=['H','C','Kdim'])
@triton.jit
def _bwd_alpha_n_kernel(
    V, IDX, GO, dA,
    N, H, C, Kdim,
    s_vn, s_vh, s_vc,
    s_in, s_ik,
    s_gn, s_gh, s_gc,
    s_an, s_ak, s_ah,
    BLOCK_H: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n  = tl.program_id(0)
    pid_hb = tl.program_id(1)
    pid_kb = tl.program_id(2)
    if pid_n >= N: return

    h_offs = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)  # [BH]
    k_offs = pid_kb * BLOCK_K + tl.arange(0, BLOCK_K)  # [BK]
    h_mask = h_offs < H
    k_mask = k_offs < Kdim

    idx_ptr  = IDX + pid_n*s_in + k_offs*s_ik
    idx_vals = tl.load(idx_ptr, mask=k_mask, other=0).to(tl.int32)           # [BK]

    acc = tl.zeros((BLOCK_K, BLOCK_H), dtype=tl.float32)                      # [BK,BH]

    c0 = 0
    while c0 < C:
        c_offs = c0 + tl.arange(0, BLOCK_C)  # [BC]
        c_mask = c_offs < C

        v_base = idx_vals[:, None, None]*s_vn + h_offs[None, :, None]*s_vh + c_offs[None, None, :]*s_vc
        v_ptr  = V + v_base
        vmask  = k_mask[:, None, None] & h_mask[None, :, None] & c_mask[None, None, :]
        v_tile = tl.load(v_ptr, mask=vmask, other=0.0).to(tl.float32)

        go_ptr  = GO + pid_n*s_gn + h_offs[:, None]*s_gh + c_offs[None, :]*s_gc
        go_tile = tl.load(go_ptr, mask=h_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)

        acc += tl.sum(v_tile * go_tile[None, :, :], axis=2)
        c0 += BLOCK_C

    da_ptr = dA + pid_n*s_an + k_offs[:, None]*s_ak + h_offs[None, :]*s_ah
    tl.store(da_ptr, acc, mask=k_mask[:, None] & h_mask[None, :])


@triton.autotune(configs=AUTOTUNE_CFGS, key=['H','C'],reset_to_zero=["dV"])
@triton.jit
def _bwd_value_n_kernel(
    A, IDX, GO, dV,
    N, H, C, Kdim,
    s_an, s_ak, s_ah,   # strides of alpha
    s_in, s_ik,         # strides of idx
    s_gn, s_gh, s_gc,   # strides of grad_out
    s_vn, s_vh, s_vc,   
    BLOCK_H: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,        
):
    pid_n  = tl.program_id(0)
    pid_hb = tl.program_id(1)
    pid_cb = tl.program_id(2)
    if pid_n >= N: return

    h_offs = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)  # [BH]
    c_offs = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)  # [BC]
    h_mask = h_offs < H
    c_mask = c_offs < C

    go_ptr  = GO + pid_n*s_gn + h_offs[:, None]*s_gh + c_offs[None, :]*s_gc
    go_tile = tl.load(go_ptr, mask=h_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)

    k = 0
    while k < Kdim:
        idx_ptr = IDX + pid_n*s_in + k*s_ik
        m = tl.load(idx_ptr, mask=True, other=0).to(tl.int32)

        a_ptr = A + pid_n*s_an + k*s_ak + h_offs*s_ah
        a_h   = tl.load(a_ptr, mask=h_mask, other=0.0).to(tl.float32)

        contrib = a_h[:, None] * go_tile  # [BH,BC], fp32

        dv_ptr = dV + m*s_vn + h_offs[:, None]*s_vh + c_offs[None, :]*s_vc
        tl.atomic_add(dv_ptr, contrib, mask=h_mask[:, None] & c_mask[None, :])

        k += 1

class VAgg_BwdFn(torch.autograd.Function):
    """
    """

    @staticmethod
    def forward(ctx, go, v, a, idx):
        
        go = go.contiguous().to(torch.float32)
        v  = v.contiguous()
        a  = a.contiguous()
        if idx.dtype != torch.int32:
            idx = idx.to(torch.int32)
        idx = idx.contiguous()

        N2, H, C = v.shape
        N,Kdim = idx.shape[:2]

        dA = torch.empty((N, Kdim, H), dtype=torch.float32, device=v.device)
        s_vn, s_vh, s_vc = v.stride()
        s_in, s_ik       = idx.stride()
        s_gn, s_gh, s_gc = go.stride()
        s_an, s_ak, s_ah = dA.stride()

        grid_alpha = lambda meta: (N, triton.cdiv(H, meta['BLOCK_H']), triton.cdiv(Kdim, meta['BLOCK_K']))
        _bwd_alpha_n_kernel[grid_alpha](
            v, idx, go, dA,
            N, H, C, Kdim,
            s_vn, s_vh, s_vc,
            s_in, s_ik,
            s_gn, s_gh, s_gc,
            s_an, s_ak, s_ah,
        )

        dV = torch.zeros_like(v, dtype=torch.float32)
        s_an0, s_ak0, s_ah0 = a.stride()
        s_vn0, s_vh0, s_vc0 = dV.stride()

        grid_value = lambda meta: (N, triton.cdiv(H, meta['BLOCK_H']), triton.cdiv(C, meta['BLOCK_C']))
        _bwd_value_n_kernel[grid_value](
            A=a, IDX=idx, GO=go, dV=dV,
            N=N, H=H, C=C, Kdim=Kdim,
            s_an=s_an0, s_ak=s_ak0, s_ah=s_ah0,
            s_in=s_in,  s_ik=s_ik,
            s_gn=s_gn,  s_gh=s_gh,  s_gc=s_gc,
            s_vn=s_vn0, s_vh=s_vh0, s_vc=s_vc0,
        )

        
        ctx.save_for_backward(go, v, a, idx)
        ctx.shape = (N, H, C, Kdim)
        return dV.to(v.dtype), dA.to(a.dtype)

    @staticmethod
    def backward(ctx, gg_dV, gg_dA):
        """

        dV[m,h,c] = Σ_{n,k: idx[n,k]=m} a[n,k,h] * go[n,h,c]
          ⇒ ∂L/∂go[n,h,c]  ⊃ Σ_k a[n,k,h] * gg_dV[idx[n,k],h,c]
          ⇒ ∂L/∂a[n,k,h]   ⊃ Σ_c go[n,h,c] * gg_dV[idx[n,k],h,c]

        dA[n,k,h] = Σ_c v[idx[n,k],h,c] * go[n,h,c]
          ⇒ ∂L/∂go[n,h,c]  ⊃ Σ_k gg_dA[n,k,h] * v[idx[n,k],h,c]
          ⇒ ∂L/∂v[m,h,c]   ⊃ Σ_{n,k: idx[n,k]=m} gg_dA[n,k,h] * go[n,h,c]
        """
        go, v, a, idx = ctx.saved_tensors
        N, H, C, Kdim = ctx.shape

        gg_dV = (torch.zeros_like(v,  dtype=torch.float32) if gg_dV is None
                 else gg_dV.contiguous().to(torch.float32))
        gg_dA = (torch.zeros((N, Kdim, H), device=v.device, dtype=torch.float32) if gg_dA is None
                 else gg_dA.contiguous().to(torch.float32))
        go_f  = go.contiguous().to(torch.float32)
        v_f   = v.contiguous().to(torch.float32)
        a_f   = a.contiguous().to(torch.float32)
        idx_i = idx.to(torch.int32, copy=False).contiguous()

        
        grad_go_v = torch.empty_like(go_f, dtype=torch.float32)
        
        
        
        
        
        s_vn, s_vh, s_vc = gg_dV.stride()
        s_in, s_ik       = idx_i.stride()
        s_an, s_ak, s_ah = a_f.stride()
        s_on, s_oh, s_oc = go_f.stride()
        grid = lambda meta: (N, triton.cdiv(H, meta['BLOCK_H']), triton.cdiv(C, meta['BLOCK_C']))
        _fwd_n_kernel[grid](
            gg_dV, idx_i, a_f, grad_go_v,
            N, H, C, Kdim,
            s_vn, s_vh, s_vc,
            s_in, s_ik,
            s_an, s_ak, s_ah,
            s_on, s_oh, s_oc,
            OUT_DTYPE=tl.float32,
        )

        grad_go_a = torch.empty_like(go_f, dtype=torch.float32)
        s_vn2, s_vh2, s_vc2 = v_f.stride()
        s_an2, s_ak2, s_ah2 = gg_dA.stride()
        _fwd_n_kernel[grid](
            v_f, idx_i, gg_dA, grad_go_a,
            N, H, C, Kdim,
            s_vn2, s_vh2, s_vc2,
            s_in,  s_ik,
            s_an2, s_ak2, s_ah2,
            s_on,  s_oh, s_oc,
            OUT_DTYPE=tl.float32,
        )
        d_go = (grad_go_v + grad_go_a).to(go.dtype)

        
        d_a = torch.empty_like(gg_dA, dtype=torch.float32)
        s_vn3, s_vh3, s_vc3 = gg_dV.stride()
        s_gn3, s_gh3, s_gc3 = go_f.stride()
        s_an3, s_ak3, s_ah3 = d_a.stride()
        grid_alpha = lambda meta: (N, triton.cdiv(H, meta['BLOCK_H']), triton.cdiv(Kdim, meta['BLOCK_K']))
        _bwd_alpha_n_kernel[grid_alpha](
            gg_dV, idx_i, go_f, d_a,
            N, H, C, Kdim,
            s_vn3, s_vh3, s_vc3,
            s_in,  s_ik,
            s_gn3, s_gh3, s_gc3,
            s_an3, s_ak3, s_ah3,
        )
        d_a = d_a.to(a.dtype)

        d_v = torch.zeros_like(v_f, dtype=torch.float32)
        s_an4, s_ak4, s_ah4 = gg_dA.stride()
        s_vn4, s_vh4, s_vc4 = d_v.stride()
        grid_value = lambda meta: (N, triton.cdiv(H, meta['BLOCK_H']), triton.cdiv(C, meta['BLOCK_C']))
        _bwd_value_n_kernel[grid_value](
            A=gg_dA, IDX=idx_i, GO=go_f, dV=d_v,
            N=N, H=H, C=C, Kdim=Kdim,
            s_an=s_an4, s_ak=s_ak4, s_ah=s_ah4,
            s_in=s_in,  s_ik=s_ik,
            s_gn=go_f.stride()[0], s_gh=go_f.stride()[1], s_gc=go_f.stride()[2],
            s_vn=s_vn4, s_vh=s_vh4, s_vc=s_vc4,
        )
        d_v = d_v.to(v.dtype)

        return d_go, d_v, d_a, None

# ----------------- Autograd wrapper -----------------
class SparseVAgg_N_Tiled_Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, alpha, idx):
        assert value.is_cuda and alpha.is_cuda and idx.is_cuda
        if idx.dtype != torch.int32:
            idx = idx.to(torch.int32)

        N2, H, C = value.shape
        N = idx.shape[0]
        Kdim = idx.shape[1]
        v = value.contiguous()
        a = alpha.contiguous()
        idx = idx.contiguous()
        out = torch.empty((N, H, C), dtype=v.dtype, device=v.device)

        s_vn, s_vh, s_vc = v.stride()
        s_in, s_ik       = idx.stride()
        s_an, s_ak, s_ah = a.stride()
        s_on, s_oh, s_oc = out.stride()

        if out.dtype == torch.float16: OUT_DTYPE = tl.float16
        elif out.dtype == torch.bfloat16: OUT_DTYPE = tl.bfloat16
        elif out.dtype == torch.float32: OUT_DTYPE = tl.float32
        else: raise TypeError("unsupported dtype")

        grid = lambda meta: (N, triton.cdiv(H, meta['BLOCK_H']), triton.cdiv(C, meta['BLOCK_C']))
        _fwd_n_kernel[grid](
            v, idx, a, out,
            N, H, C, Kdim,
            s_vn, s_vh, s_vc,
            s_in, s_ik,
            s_an, s_ak, s_ah,
            s_on, s_oh, s_oc,
            OUT_DTYPE=OUT_DTYPE,
        )
        ctx.save_for_backward(v, a, idx)
        ctx.shape = (N, H, C, Kdim)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        v, a, idx = ctx.saved_tensors
        N, H, C, Kdim = ctx.shape
        go = grad_out.contiguous().to(torch.float32)

    
        need_graph = torch.is_grad_enabled()

        if need_graph:
            
            dV, dA = VAgg_BwdFn.apply(go, v, a, idx)
        else:
            # dAlpha
            dA = torch.empty((N, Kdim, H), dtype=torch.float32, device=v.device)
            s_vn, s_vh, s_vc = v.stride()
            s_in, s_ik       = idx.stride()
            s_gn, s_gh, s_gc = go.stride()
            s_an, s_ak, s_ah = dA.stride()

            grid_alpha = lambda meta: (N, triton.cdiv(H, meta['BLOCK_H']), triton.cdiv(Kdim, meta['BLOCK_K']))
            _bwd_alpha_n_kernel[grid_alpha](
                v, idx, go, dA,
                N, H, C, Kdim,
                s_vn, s_vh, s_vc,
                s_in, s_ik,
                s_gn, s_gh, s_gc,
                s_an, s_ak, s_ah,
            )

            dV = torch.zeros_like(v, dtype=torch.float32)
            s_an0, s_ak0, s_ah0 = a.stride()
            s_vn0, s_vh0, s_vc0 = dV.stride()

            grid_value = lambda meta: (N, triton.cdiv(H, meta['BLOCK_H']), triton.cdiv(C, meta['BLOCK_C']))

            
            _bwd_value_n_kernel[grid_value](A=a, IDX=idx, GO=go, dV=dV, 
                            N=N, H=H, C=C, Kdim=Kdim,
                s_an=s_an0, s_ak=s_ak0, s_ah=s_ah0,
                s_in=s_in,  s_ik=s_ik,
                s_gn=s_gn,  s_gh=s_gh,  s_gc=s_gc,
                s_vn=s_vn0, s_vh=s_vh0, s_vc=s_vc0,
                )


        return dV.to(v.dtype), dA.to(a.dtype), None

def sparse_v_agg_triton_n_tiled_test(value, alpha, idx):
    # input must be N*H*C
    return SparseVAgg_N_Tiled_Fn.apply(value, alpha, idx)


def sparse_v_agg_triton_n_tiled(value, alpha, idx):
    value = value.permute(0,2,1)
    return SparseVAgg_N_Tiled_Fn.apply(value, alpha, idx).permute(0,2,1)

