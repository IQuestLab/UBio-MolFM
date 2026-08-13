# Created for High-Performance Single-GPU Structure Relaxation

from typing import Optional

import torch
from ase.optimize.optimize import Optimizer


class TorchLBFGS(Optimizer):
    """
    GPU-driven Limited-memory BFGS (L-BFGS) optimizer.

    The two-loop recursion runs entirely on GPU: the pairwise history dot
    products are batched into single BLAS calls instead of looping per history
    entry.
    """

    def __init__(
        self,
        atoms,
        restart: Optional[str] = None,
        logfile: str = "-",
        trajectory: Optional[str] = None,
        maxstep: float = 0.2,
        memory: int = 100,
        damping: float = 1.0,
        alpha: float = 70.0,
        fmax: float = 0.05,
        **kwargs,
    ):
        if not hasattr(atoms, "_device") or not hasattr(atoms, "_dtype"):
            raise TypeError("TorchLBFGS requires a TorchAtoms instance.")

        super().__init__(atoms, restart, logfile, trajectory, **kwargs)

        self.device = atoms._device
        self.dtype = atoms._dtype
        self.maxstep = maxstep
        self.memory = memory
        self.damping = damping
        self.alpha = alpha
        self.fmax = fmax

        self.step_count = 0
        self.r0: Optional[torch.Tensor] = None
        self.f0: Optional[torch.Tensor] = None

        self.dim = len(atoms) * 3

        # Circular-buffer history
        self._s_hist = torch.empty((self.memory, self.dim), device=self.device, dtype=self.dtype)
        self._y_hist = torch.empty((self.memory, self.dim), device=self.device, dtype=self.dtype)
        self._rho_hist = torch.empty((self.memory,), device=self.device, dtype=self.dtype)

        # Pre-allocated buffers for the reordered views used in the two-loop
        # recursion.  Filled row-by-row each step to avoid the 2×[m,dim]
        # allocation that index_select would incur.
        self._S_buf = torch.empty((self.memory, self.dim), device=self.device, dtype=self.dtype)
        self._Y_buf = torch.empty((self.memory, self.dim), device=self.device, dtype=self.dtype)

        self._hist_len = 0
        self._ptr = 0

    def log(self, forces: Optional[torch.Tensor] = None):
        pass

    def converged(self, forces=None) -> bool:
        # ASE's OptimizableAtoms.get_gradient() returns -forces.ravel()
        # (1-D, shape [n_atoms * 3]).  Reshape back for the per-atom norm.
        if forces is None:
            forces = self.atoms.get_forces()

        if isinstance(forces, torch.Tensor):
            if forces.numel() > 0:
                if forces.dim() == 1:
                    forces = forces.view(-1, 3)
                fmax_now = torch.max(torch.norm(forces, dim=1))
            else:
                fmax_now = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            return fmax_now.item() < self.fmax
        return super().converged(forces)

    def run(self, steps=1000):
        return super().run(fmax=self.fmax, steps=steps)

    # ------------------------------------------------------------------
    # Core step — batched two-loop recursion
    # ------------------------------------------------------------------
    def step(self, f: Optional[torch.Tensor] = None):
        atoms = self.atoms
        if f is None:
            f = atoms.get_forces()

        f_t = f.to(device=self.device, dtype=self.dtype) if isinstance(f, torch.Tensor) \
              else torch.tensor(f, device=self.device, dtype=self.dtype)
        f_t = f_t.detach()  # L-BFGS recursion does not need autograd

        r_t = atoms.get_positions()
        if not isinstance(r_t, torch.Tensor):
            r_t = torch.tensor(r_t, device=self.device, dtype=self.dtype)

        r = r_t.view(-1)
        g = -f_t.view(-1)

        # Re-allocate buffers if the atom count changed under us.
        new_dim = r.numel()
        if new_dim != self.dim:
            self.dim = new_dim
            self._s_hist = torch.empty((self.memory, self.dim), device=self.device, dtype=self.dtype)
            self._y_hist = torch.empty((self.memory, self.dim), device=self.device, dtype=self.dtype)
            self._rho_hist = torch.empty((self.memory,), device=self.device, dtype=self.dtype)
            self._S_buf = torch.empty((self.memory, self.dim), device=self.device, dtype=self.dtype)
            self._Y_buf = torch.empty((self.memory, self.dim), device=self.device, dtype=self.dtype)
            self._hist_len = 0
            self._ptr = 0
            self.step_count = 0
            self.r0 = None
            self.f0 = None

        # --- Initialization (steepest descent) ---
        if self.step_count == 0:
            self.r0 = r.clone()
            self.f0 = f_t.view(-1).clone()
            p = -g
            p_sq = torch.dot(p, p) if p.numel() > 0 \
                else torch.tensor(0.0, device=self.device, dtype=self.dtype)
            p_norm = torch.sqrt(p_sq).item()
            if p_norm > 1e-8:
                p *= self.alpha / max(1.0, p_norm)
            self._apply_update(atoms, r_t, p, f_t)
            self.step_count += 1
            return

        # --- L-BFGS step ---
        g0 = -self.f0.view(-1)
        s_k = r - self.r0
        y_k = g - g0

        # Update history
        sy = (torch.dot(s_k, y_k) if self.dim > 0
              else torch.tensor(0.0, device=self.device, dtype=self.dtype))
        if sy > 1e-10:
            rho_k = 1.0 / sy
            if self.dim > 0:
                self._s_hist[self._ptr].copy_(s_k)
                self._y_hist[self._ptr].copy_(y_k)
            self._rho_hist[self._ptr] = rho_k
            self._ptr = (self._ptr + 1) % self.memory
            self._hist_len = min(self._hist_len + 1, self.memory)

        m = self._hist_len
        if m == 0:
            p = -g
            self._apply_update(atoms, r_t, p, f_t)
            self.step_count += 1
            return

        # History in newest-first order: index 0 = newest, m-1 = oldest.
        # Fill pre-allocated buffers row-by-row instead of using index_select,
        # which would allocate a fresh [m, dim] tensor (37.5 MB for 15K atoms)
        # every single step.
        indices_list = [(self._ptr - 1 - i) % self.memory for i in range(m)]
        for dst, src in enumerate(indices_list):
            self._S_buf[dst].copy_(self._s_hist[src])
            self._Y_buf[dst].copy_(self._y_hist[src])
        S = self._S_buf[:m]   # view — zero allocation
        Y = self._Y_buf[:m]   # view — zero allocation
        rho = torch.empty(m, device=self.device, dtype=self.dtype)
        for dst, src in enumerate(indices_list):
            rho[dst] = self._rho_hist[src]
        last_idx = (self._ptr - 1) % self.memory

        # ==== Pairwise dot products, batched into single BLAS calls ====
        if self.dim > 0:
            s_dot_g = S @ g                             # [m]
            s_dot_y = S @ Y.T                           # [m,m]
            yy = torch.dot(self._y_hist[last_idx], self._y_hist[last_idx])
        else:
            s_dot_g = torch.zeros(m, device=self.device, dtype=self.dtype)
            s_dot_y = torch.zeros((m, m), device=self.device, dtype=self.dtype)
            yy = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # ---- Forward loop (newest → oldest) ----
        alphas = torch.empty(m, device=self.device, dtype=self.dtype)
        for i in range(m):
            acc = s_dot_g[i]
            if i > 0:
                acc -= torch.dot(s_dot_y[i, :i], alphas[:i])
            alphas[i] = rho[i] * acc

        gamma = (1.0 / (self._rho_hist[last_idx] * yy)
                 if yy > 1e-10 else 1.0)

        # q_final = g after forward loop: g - Σ α_j·y_j
        # Vectorised via BLAS gemv (no per-iteration temporaries).
        q_final = g.clone()
        if self.dim > 0:
            q_final = torch.addmv(q_final, Y.T, alphas, beta=1.0, alpha=-1.0)
        z = gamma * q_final

        # ---- Backward loop (oldest → newest) ----
        # y_dot_z_init[i] = Y[i] · z_init
        y_dot_z_init = (Y @ z).clone() if self.dim > 0 \
            else torch.zeros(m, device=self.device, dtype=self.dtype)
        # y_dot_s[i,j] = Y[i] · S[j]  = s_dot_y[j, i]   (transpose)
        y_dot_s = s_dot_y.T  # [m,m] — S is newest-first in both dims

        betas = torch.empty(m, device=self.device, dtype=self.dtype)
        for i in range(m - 1, -1, -1):
            # z_at_i = z_init + Σ_{j>i} S[j] * (α[j] - β[j])
            # β_i = ρ_i * (Y[i]·z_init + Σ_{j>i} (α[j] - β[j]) * Y[i]·S[j])
            acc = y_dot_z_init[i]
            if i < m - 1:
                # j = i+1 .. m-1  (newer entries, already processed)
                d_alpha_beta = alphas[i + 1:] - betas[i + 1:]
                acc += torch.dot(y_dot_s[i, i + 1:], d_alpha_beta)
            betas[i] = rho[i] * acc

        # z += Σ (αᵢ - βᵢ)·Sᵢ  →  BLAS gemv (no per-iteration temporaries).
        if self.dim > 0 and m > 0:
            z = torch.addmv(z, S.T, alphas - betas, beta=1.0, alpha=1.0)

        p = -z
        self._apply_update(atoms, r_t, p, f_t)
        self.step_count += 1

        # Periodically return cached-but-unused blocks to the CUDA driver so
        # fragmentation from long-running relax runs cannot accumulate.
        if self.step_count % 50 == 0:
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    def _apply_update(self, atoms, r_t, p, f_t):
        """Scale, clip, and apply the search direction.

        Parameters
        ----------
        f_t:
            Forces at the current position (already computed at the top of step()).
            Passed in to avoid an extra get_forces() forward pass.
        """
        p = p * self.damping
        p_3d = p.view(-1, 3)

        if p_3d.numel() > 0:
            max_step = torch.max(torch.norm(p_3d, dim=1)).item()
        else:
            max_step = 0.0

        if max_step >= self.maxstep:
            p_3d *= self.maxstep / max_step

        if self.dim > 0:
            self.r0.copy_(r_t.view(-1))
            self.f0.copy_(f_t.view(-1))
            r_new = r_t + p_3d
            atoms.set_positions(r_new, apply_constraint=True)
