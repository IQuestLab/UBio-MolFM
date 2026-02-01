
import os
import sys
import math
import argparse
import torch


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from sparse_v_agg_lastdim_autograd import sparse_v_agg_lastdim

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# -----------------------------

# -----------------------------
def dump_tensor(name, t):
    print(f"[DUMP] {name}: shape={tuple(t.shape)}, dtype={t.dtype}, "
          f"requires_grad={t.requires_grad}, is_leaf={t.is_leaf}, grad_fn={t.grad_fn}")
    if t.numel() == 0:
        print("       (empty tensor)")
        return

    if t.is_floating_point() or t.is_complex():
        t_absmax = t.abs().max().item()
        t_min = t.min().item()
        t_max = t.max().item()
        t_mean = t.mean().item()
        print(f"       stats: min={t_min:+.3e}, max={t_max:+.3e}, "
              f"mean={t_mean:+.3e}, absmax={t_absmax:.3e}")
    else:
        t_min = int(t.min().item())
        t_max = int(t.max().item())
        uniq = -1
        try:
            
            if t.numel() <= 1_000_000:
                uniq = int(t.unique().numel())
        except Exception:
            pass
        extra = f", unique={uniq}" if uniq >= 0 else ""
        print(f"       stats(int): min={t_min}, max={t_max}{extra}")


# -----------------------------

# -----------------------------
def ref_forward(value: torch.Tensor, idx: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    gathered = value[idx]                         # [N,K,L,C]
    out = (alpha.unsqueeze(2) * gathered).sum(dim=1)  # [N,L,C]
    return out


def max_err(a: torch.Tensor, b: torch.Tensor):
    with torch.no_grad():
        diff = (a - b).abs()
        abs_max = diff.max().item() if diff.numel() else 0.0
        denom = torch.maximum(a.abs(), b.abs())
        rel = (diff / torch.clamp(denom, min=1e-20)).max().item() if diff.numel() else 0.0
        return abs_max, rel


# -----------------------------

# -----------------------------
def make_data(N, L, C, K, dtype, device):
    gen = torch.Generator(device=device).manual_seed(1234)

    value = torch.randn((N, L, C), dtype=dtype, device=device, generator=gen, requires_grad=True)
    alpha = torch.randn((N, K, C), dtype=dtype, device=device, generator=gen, requires_grad=True)
    idx = torch.randint(low=0, high=N, size=(N, K), dtype=torch.int64, device=device, generator=gen)

    
    g = torch.randn((N, L, C), dtype=dtype, device=device, generator=gen)

    
    rv = torch.randn((N, L, C), dtype=dtype, device=device, generator=gen)
    ra = torch.randn((N, K, C), dtype=dtype, device=device, generator=gen)

    return value, idx, alpha, g, rv, ra


# -----------------------------

# -----------------------------
def test_case(N, L, C, K, dtype, device, use_amp=False,
              tol_f_abs=2e-4, tol_f_rel=2e-4,
              tol_g_abs=5e-4, tol_g_rel=5e-4,
              tol_h_abs=1e-3, tol_h_rel=1e-3):

    value, idx, alpha, g, rv, ra = make_data(N, L, C, K, dtype, device)

    print("[ENV] torch={}, triton={}, cuda.is_available={}".format(
        torch.__version__,  
        getattr(torch, "version", type("x",(object,),{})()).__dict__.get("triton", "unknown"),
        torch.cuda.is_available()
    ))
    dump_tensor("value", value)
    dump_tensor("alpha", alpha)
    dump_tensor("idx", idx)

    
    if use_amp and dtype in (torch.float16, torch.bfloat16):
        amp_dtype = torch.bfloat16 if dtype is torch.bfloat16 else torch.float16
        autocast = torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype)
    else:
        class _null:
            def __enter__(self): return None
            def __exit__(self, *args): return False
        autocast = _null()

    
    with autocast:
        tri_out = sparse_v_agg_lastdim(value, idx, alpha)
        ref_out = ref_forward(value, idx, alpha)

    dump_tensor("tri_out", tri_out)
    dump_tensor("ref_out", ref_out)

    f_abs, f_rel = max_err(tri_out.float(), ref_out.float())

    
    L_tri = (tri_out * g).sum()
    L_ref = (ref_out * g).sum()

    print(f"[INFO] L_tri.requires_grad={L_tri.requires_grad}, L_ref.requires_grad={L_ref.requires_grad}")

    tri_dV, tri_dA = torch.autograd.grad(L_tri, (value, alpha), create_graph=True, retain_graph=True)
    ref_dV, ref_dA = torch.autograd.grad(L_ref, (value, alpha), create_graph=True, retain_graph=True)

    dump_tensor("tri_dV", tri_dV)
    dump_tensor("tri_dA", tri_dA)
    dump_tensor("ref_dV", ref_dV)
    dump_tensor("ref_dA", ref_dA)

    gV_abs, gV_rel = max_err(tri_dV.float(), ref_dV.float())
    gA_abs, gA_rel = max_err(tri_dA.float(), ref_dA.float())

    
    print("[INFO] tri_dV.requires_grad={}, grad_fn={}".format(tri_dV.requires_grad, tri_dV.grad_fn))
    print("[INFO] tri_dA.requires_grad={}, grad_fn={}".format(tri_dA.requires_grad, tri_dA.grad_fn))

    
    s_tri = (tri_dV * rv).sum() + (tri_dA * ra).sum()
    s_ref = (ref_dV * rv).sum() + (ref_dA * ra).sum()

    print(f"[INFO] s_tri.requires_grad={s_tri.requires_grad}, s_ref.requires_grad={s_ref.requires_grad}")

    tri_d2V, tri_d2A = torch.autograd.grad(s_tri, (value, alpha), retain_graph=False, allow_unused=False)
    ref_d2V, ref_d2A = torch.autograd.grad(s_ref, (value, alpha), retain_graph=False, allow_unused=False)

    dump_tensor("tri_d2V", tri_d2V)
    dump_tensor("tri_d2A", tri_d2A)
    dump_tensor("ref_d2V", ref_d2V)
    dump_tensor("ref_d2A", ref_d2A)

    hV_abs, hV_rel = max_err(tri_d2V.float(), ref_d2V.float())
    hA_abs, hA_rel = max_err(tri_d2A.float(), ref_d2A.float())

    
    ok_f  = (f_abs <= tol_f_abs) and (f_rel <= tol_f_rel)
    ok_gV = (gV_abs <= tol_g_abs) and (gV_rel <= tol_g_rel)
    ok_gA = (gA_abs <= tol_g_abs) and (gA_rel <= tol_g_rel)
    ok_hV = (hV_abs <= tol_h_abs) and (hV_rel <= tol_h_rel)
    ok_hA = (hA_abs <= tol_h_abs) and (hA_rel <= tol_h_rel)
    ok = ok_f and ok_gV and ok_gA and ok_hV and ok_hA

    def fmt(x): return f"{x:.3e}"

    print(f"\nShape N={N}, L={L}, C={C}, K={K}")
    print(f"  fwd:    abs={fmt(f_abs)}, rel={fmt(f_rel)}   -> {'OK' if ok_f else 'FAIL'}")
    print(f"  dV:     abs={fmt(gV_abs)}, rel={fmt(gV_rel)} -> {'OK' if ok_gV else 'FAIL'}")
    print(f"  dA:     abs={fmt(gA_abs)}, rel={fmt(gA_rel)} -> {'OK' if ok_gA else 'FAIL'}")
    print(f"  d2V(H): abs={fmt(hV_abs)}, rel={fmt(hV_rel)} -> {'OK' if ok_hV else 'FAIL'}")
    print(f"  d2A(H): abs={fmt(hA_abs)}, rel={fmt(hA_rel)} -> {'OK' if ok_hA else 'FAIL'}")
    print(f"  ==> {'PASS' if ok else 'FAIL'}\n")

    return ok


def main():
    parser = argparse.ArgumentParser(description="2nd-order numerical validation for sparse_v_agg_lastdim (with logs)")
    parser.add_argument("--device", default="cuda", type=str, help="cuda or cpu")
    parser.add_argument("--dtype",  default="float32", type=str, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--amp", action="store_true", help="use autocast")
    parser.add_argument("--small", action="store_true", help="run smaller shapes")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    device = args.device

    print("== sparse_v_agg_lastdim 2nd-order accuracy validation ==")
    print(f"AMP autocast: {args.amp}")
    print(f"DTYPE: {args.dtype}\n")
    print(f"[ENV] torch={torch.__version__}, triton={getattr(torch, 'version', type('x',(object,),{})()).__dict__.get('triton','unknown')}, cuda.is_available={torch.cuda.is_available()}")

    if args.small:
        shapes = [
            (64,  9, 128, 13),
            (96,  9, 128, 16),
            (128, 4, 128, 32),
        ]
    else:
        shapes = [
            (256, 9, 256, 13),
            (512, 9, 256, 32),
            (384, 9, 256, 48),
            (256, 1, 256, 64),
        ]

    all_ok = True
    for (N, L, C, K) in shapes:
        ok = test_case(
            N, L, C, K, dtype=dtype, device=device, use_amp=args.amp,
            tol_f_abs=2e-4, tol_f_rel=2e-4,
            tol_g_abs=5e-4, tol_g_rel=5e-4,
            tol_h_abs=1e-3, tol_h_rel=1e-3,
        )
        all_ok &= ok

    print("Overall:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()

