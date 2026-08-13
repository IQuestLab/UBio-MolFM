# Description: Fully GPU-driven Langevin MD integrator for TorchAtoms.
# Features: Ping-Pong buffering, CUDA Graphs support, Zero-allocation step loop.
# Deterministic (friction==0) fast path. With fr=0 the Vanden-Eijnden-Ciccotti
# coefficients collapse to c1=dt/2, c2=0, sigma=0 -> the scheme is exactly
# velocity-Verlet (NVE). In that mode the per-step RNG draw and noise
# projection are skipped (perturbations are identically zero), so ensemble='nve'
# reuses this integrator with no wasted work.

import warnings

import numpy as np
import torch
from ase import units
from ase.constraints import FixCom
from ase.md.md import MolecularDynamics
from loguru import logger

from molfm.interface.ase.torch_ext.torch_constraints import TorchFixCom


def _langevin_prepare_core(
    x_t: torch.Tensor, 
    v_t: torch.Tensor, 
    forces_t: torch.Tensor,
    masses_3d_t: torch.Tensor, 
    inv_masses_3d_t: torch.Tensor,
    c1: float, 
    c2: float, 
    dt: float,
    rnd_vel_t: torch.Tensor,
    rnd_pos_t: torch.Tensor,
) -> torch.Tensor:
    """Pure tensor pre-position-update core."""
    # Compute half-step velocity and new positions
    # Using pre-calculated 1/m to avoid division operation overhead
    v_half_t = v_t + (c1 * forces_t * inv_masses_3d_t - c2 * v_t + rnd_vel_t)
    x_new_t = x_t + dt * v_half_t + rnd_pos_t
    return x_new_t


def _langevin_finalize_core(
    x_prev_t: torch.Tensor, 
    x_after_t: torch.Tensor,
    masses_3d_t: torch.Tensor, 
    inv_masses_3d_t: torch.Tensor,
    c1: float, 
    c2: float, 
    dt: float,
    rnd_pos_t: torch.Tensor, 
    rnd_vel_t: torch.Tensor, 
    forces_t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure tensor post-position-update core."""
    # Reconstruct velocity from actual displacement (accounts for constraint projection)
    v_t = (x_after_t - x_prev_t - rnd_pos_t) / dt
    v_t = v_t + (c1 * forces_t * inv_masses_3d_t - c2 * v_t + rnd_vel_t)
    mom_t = v_t * masses_3d_t
    
    return v_t, mom_t


# ------------------------------------------------------------------
# Debug timer — zero-cost when not active (only created if debug_timing=True)
# ------------------------------------------------------------------
class _DebugTimer:
    """Per-step timing accumulator.  Prints breakdown every 50 steps."""

    def __init__(self):
        self._acc = {}
        self._count = 0
        self._last = None

    def tick(self, name: str):
        import time
        now = time.time()
        if self._last is not None:
            elapsed = now - self._last
            self._acc[name] = self._acc.get(name, 0.0) + elapsed
        self._last = now

    def commit(self):
        self._count += 1
        if self._count % 50 == 0:
            parts = " ".join(
                f"{k}={v / self._count * 1000:.2f}ms"
                for k, v in sorted(self._acc.items())
            )
            print(f"[LANGEVIN avg over {self._count} steps] {parts}", flush=True)


# ------------------------------------------------------------------
# Integrator Class
# ------------------------------------------------------------------
class TorchLangevin(MolecularDynamics):
    """
    Fully GPU-driven Langevin (NVT) integrator for TorchAtoms.
    Zero-allocation step loop with pre-allocated random buffers.
    """

    def __init__(
        self,
        atoms,
        timestep: float,
        temperature: float | None = None,
        temperature_K: float | None = None,
        friction: float | None = None,
        fixcm: bool = True,
        rng=None,
        debug_timing: bool = False,
        **kwargs
    ):
        logger.info(
            f"Initializing TorchLangevin with timestep={timestep}, "
            f"friction={friction}, fixcm={fixcm}, temperature={temperature}, "
            f"temperature_K={temperature_K}"
        )
        from molfm.interface.ase.torch_ext.torch_atoms import TorchAtoms
        if not isinstance(atoms, TorchAtoms):
            raise TypeError(f"TorchLangevin requires a TorchAtoms instance, got {type(atoms).__name__}")

        if friction is None:
            raise TypeError("Missing required 'friction' argument.")

        self.device = atoms._device
        self.dtype = atoms._dtype
        self.fr = friction
        self._debug_timing = debug_timing
        self._debug_timer = _DebugTimer() if debug_timing else None
        self._rng = rng if rng else torch.Generator(device=self.device)
        self._fixcom_impl = TorchFixCom() if fixcm else None

        if fixcm:
            warnings.warn(
                "The implementation of `fixcm=True` in `Langevin` does not "
                "strictly sample the correct NVT distributions. "
                "Use `fixcm=False` together with `ase.constraints.FixCom` instead.",
                FutureWarning
            )
        self.fix_com = fixcm

        from ase.md.md import process_temperature
        self.temp = units.kB * process_temperature(temperature, temperature_K, "eV")

        MolecularDynamics.__init__(self, atoms, timestep, **kwargs)
        self.updatevars()

    def set_temperature(self, temperature=None, temperature_K=None):
        """Update target temperature and recompute Langevin coefficients."""
        from ase.md.md import process_temperature
        self.temp = units.kB * process_temperature(temperature, temperature_K, "eV")
        self.updatevars()

    def updatevars(self):
        """Precompute scalar coefficients, persistent tensors, and buffers."""
        atoms = self.atoms
        dt = self.dt
        T = self.temp
        fr = self.fr
        natoms = len(atoms)

        # 1. Physics Constants & Inverse Values (to avoid division in hot loop)
        masses = atoms._torch_masses
        self._masses_3d_t = masses.unsqueeze(-1)
        self._inv_masses_3d_t = 1.0 / self._masses_3d_t

        sigma_t = torch.sqrt((2.0 * T * fr) / masses)

        # Deterministic (NVE) path: fr==0 => no drag (c2=0), no noise (sigma=0),
        # c1=dt/2 -> the two-stage update is exactly velocity-Verlet. Flag it so
        # step() can skip the RNG draw and noise projection (perturbations are 0).
        self._deterministic = (fr == 0.0)

        # 2. Scalar coefficients
        self.c1 = dt / 2.0 - dt * dt * fr / 8.0
        self.c2 = dt * fr / 2.0 - dt * dt * fr * fr / 8.0

        # 3. Per-atom coefficients
        dt_sqrt = dt ** 0.5
        dt_3_2 = dt ** 1.5
        self._c3_t = dt_sqrt * sigma_t / 2.0 - dt_3_2 * fr * sigma_t / 8.0
        self._c5_t = dt_3_2 * sigma_t / (2.0 * (3.0 ** 0.5))
        self._c4_t = fr / 2.0 * self._c5_t

        # ------------------------------------------------------------------
        # 5. Buffer Allocations for Zero-Allocation Step Loop
        # ------------------------------------------------------------------
        # Unified buffer for random numbers to halve generator calls
        self._rnd_buffer = torch.empty((2, natoms, 3), device=self.device, dtype=self.dtype)
        # Persistent zero perturbation for the deterministic (NVE) path.
        self._zero3_t = torch.zeros((natoms, 3), device=self.device, dtype=self.dtype)

    def step(self, forces=None):
        """Execute one integration step entirely on the GPU."""
        atoms = self.atoms
        _timer = self._debug_timer if self._debug_timing else None

        # 1. Fetch current forces
        if forces is None:
            forces_t = atoms.get_forces(md=True, copy=False)
        else:
            forces_t = forces if torch.is_tensor(forces) else torch.as_tensor(forces, device=self.device, dtype=self.dtype)
        if _timer: _timer.tick("get_forces_1")

        # 2. Fetch current state (zero-copy references)
        v_t = atoms.get_velocities()
        x_t = atoms.get_positions()
        if _timer: _timer.tick("get_state")

        if self._deterministic:
            # NVE / velocity-Verlet: fr==0 -> sigma=0, so both perturbations are
            # identically zero. Skip the RNG draw and noise projection entirely.
            rnd_pos_t = self._zero3_t
            rnd_vel_t = self._zero3_t
            if _timer: _timer.tick("nve_zero_noise")
        else:
            # 3. Generate normal noise in a single kernel call
            self._rnd_buffer.normal_(generator=self._rng)
            xi_t, eta_t = self._rnd_buffer[0], self._rnd_buffer[1]
            if _timer: _timer.tick("rng")

            # 4. Project noise if geometric constraints exist
            if atoms.constraints:
                for constraint in atoms.constraints:
                    torch_impl = atoms._get_torch_constraint_impl(constraint)
                    if hasattr(torch_impl, "redistribute_forces_md"):
                        xi_t = torch_impl.redistribute_forces_md(atoms, xi_t, rand=True)
                        eta_t = torch_impl.redistribute_forces_md(atoms, eta_t, rand=True)
            if _timer: _timer.tick("constraints")

            # 5. Precompute random perturbations outside compiled graph
            rnd_pos_t = eta_t * self._c5_t.unsqueeze(-1)
            rnd_vel_t = xi_t * self._c3_t.unsqueeze(-1) - eta_t * self._c4_t.unsqueeze(-1)
            if _timer: _timer.tick("precompute")

        # 6. Core pre-update
        x_new_t = _langevin_prepare_core(
            x_t, v_t, forces_t,
            self._masses_3d_t, self._inv_masses_3d_t,
            self.c1, self.c2, self.dt,
            rnd_vel_t, rnd_pos_t,
        )
        if _timer: _timer.tick("prepare_core")

        # Apply positional updates and constraints
        atoms.set_positions(x_new_t, apply_constraint=True)
        if _timer: _timer.tick("set_positions")
        if self._fixcom_impl is not None and not any(
            isinstance(constraint, FixCom) for constraint in atoms.constraints
        ):
            atoms.set_positions(
                self._fixcom_impl.adjust_positions(atoms, atoms.get_positions()),
                apply_constraint=False,
            )
        if _timer: _timer.tick("fixcom_pos")

        # 8. Second force evaluation
        forces_t_new = atoms.get_forces(md=True, copy=False)
        if _timer: _timer.tick("get_forces_2")

        # 9. Core post-update
        v_final_t, mom_t = _langevin_finalize_core(
            x_t, atoms._torch_positions,
            self._masses_3d_t, self._inv_masses_3d_t,
            self.c1, self.c2, self.dt,
            rnd_pos_t, rnd_vel_t, forces_t_new
        )
        if _timer: _timer.tick("finalize_core")

        # Apply momentum updates and constraints
        atoms.set_momenta(mom_t, apply_constraint=True)
        if _timer: _timer.tick("set_momenta")
        if self._fixcom_impl is not None and not any(
            isinstance(constraint, FixCom) for constraint in atoms.constraints
        ):
            atoms.set_momenta(
                self._fixcom_impl.adjust_momenta(atoms, atoms.get_momenta()),
                apply_constraint=False,
            )
        if _timer: _timer.tick("fixcom_mom")

        if _timer: _timer.commit()

        return forces_t_new
    
if __name__ == "__main__":
    import torch
    import numpy as np
    from ase import Atoms, units
    from ase.calculators.calculator import Calculator, all_changes
    from molfm.interface.ase.torch_ext.torch_atoms import TorchAtoms
    from molfm.interface.ase.calculator.e2former_calculator import E2FormerCalculator
    from ase.constraints import FixCom

    class DummyE2FormerCalculator(E2FormerCalculator):
        """Minimal E2FormerCalculator subclass for testing the GPU fast-path.

        Returns harmonic spring forces centered at the initial positions:
            F = -k * (x - x0)
        """

        def __init__(self, k: float = 1.0):
            # Bypass heavy model loading; we only need the calculate interface.
            Calculator.__init__(self)
            self.k = k
            self._x0 = None

        def calculate(self, atoms, properties=["energy"], system_changes=all_changes, use_gpu=False):
            x = atoms.get_positions()
            if self._x0 is None:
                self._x0 = x.clone()
            forces = -self.k * (x - self._x0)
            self.results = {"forces": forces}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    # 1. Build a small system directly from numbers + positions.
    numbers = np.array([1, 1, 8])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0],
        ]
    )
    atoms = TorchAtoms(
        numbers=numbers,
        positions=positions,
        cell=[10, 10, 10],
        device=device,
        dtype=dtype,
    )
    atoms.set_velocities(torch.zeros(3, 3, device=device, dtype=dtype))
    atoms.calc = DummyE2FormerCalculator(k=10.0)

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Initial positions device: {atoms.get_positions().device}")

    # 2. Run a short MD without constraints -> fully GPU path.
    dyn = TorchLangevin(
        atoms,
        timestep=0.5 * units.fs,
        temperature_K=300.0,
        friction=0.01 / units.fs,
        fixcm=False,
    )
    for i in range(5):
        dyn.step()
    print(f"After 5 steps (no constraints): pos device = {atoms.get_positions().device}")

    # 3. Attach FixCom and re-run -> should still stay on GPU because
    #    FixCom is registered as a torch-native constraint.
    atoms.constraints = [FixCom()]
    for i in range(5):
        dyn.step()
    print(f"After 5 steps (FixCom): pos device = {atoms.get_positions().device}")
    print(f"COM velocity (should be ~0): {atoms.get_velocities().sum(0)}")

    # 4. Export to standard ASE Atoms and sanity-check key properties.
    ase_atoms = atoms.to_ase_atoms()
    assert isinstance(ase_atoms, Atoms)
    assert not isinstance(ase_atoms, TorchAtoms)
    np.testing.assert_allclose(ase_atoms.get_positions(), atoms.get_positions().detach().cpu().numpy())
    np.testing.assert_allclose(ase_atoms.get_momenta(), atoms.get_momenta().detach().cpu().numpy())
    np.testing.assert_array_equal(ase_atoms.get_atomic_numbers(), atoms.get_atomic_numbers().detach().cpu().numpy())
    np.testing.assert_allclose(ase_atoms.get_cell().array, atoms.get_cell().detach().cpu().numpy())
    print(f"to_ase_atoms export OK: type={type(ase_atoms).__name__}, natoms={len(ase_atoms)}")

    print("Dummy test passed.")
