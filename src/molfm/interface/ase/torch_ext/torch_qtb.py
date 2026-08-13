# Description: GPU-native Quantum Thermal Bath (QTB / adQTB) integrator for TorchAtoms,
#   a drop-in alternative to TorchLangevin for nuclear-quantum-effect (NQE) MD. The
#   thermostat replaces the white Langevin random force with COLORED noise whose power
#   spectral density follows the quantum fluctuation-dissipation theorem,
#   theta(omega,T) = (hbar*omega/2)*coth(hbar*omega/2 kT), injecting zero-point energy
#   into high-frequency modes (e.g. the O-H stretch). It needs only energy+forces (no
#   stress) — matching the E2Former calculator.
#
#   Scheme: BBK velocity-Verlet (Dammak PRL 2009 / LAMMPS fix qtb):
#       v += dt/2 * (f - m*gamma*v + R)/m   (half kick)
#       x += dt * v                          (drift)
#       f  = forces(x)
#       v += dt/2 * (f - m*gamma*v + R)/m   (half kick, same R)
#   R has variance 2*m*gamma*theta(omega,T)/dt; theta->kT (classical=True) reduces this
#   to an ordinary BBK Langevin thermostat (the hbar->0 regression).
#
#   adQTB-r (Mangaud/Huppert/Ple JCTC 2019; Mauger JPCL 2021): per-frequency FDT residual
#   Delta(w) = Re[C_vF(w)] - m*gamma*C_vv(w) drives a multiplicative gain on the noise
#   filter to remove zero-point-energy leakage.
#
#   NOTE: the colored noise carries per-atom temporal memory (segment FFT history), so
#   the atom ordering must stay fixed for the whole run. White-noise TorchLangevin is
#   memoryless and carries no such requirement.

import numpy as np
import torch
from ase import units
from ase.constraints import FixCom
from ase.md.md import MolecularDynamics
from loguru import logger

from molfm.interface.ase.torch_ext.torch_constraints import TorchFixCom

# hbar in ASE units (eV * fs); matches ase.units (_hbar/_e*1e15).
HBAR_EV_FS = 0.6582119569509065


def _quantum_theta_over_kT(omega_rad_fs, T_K, hbar=HBAR_EV_FS):
    """Dimensionless quantum/classical energy ratio theta(omega,T)/kT.

    theta = (hbar*omega/2) coth(hbar*omega/2kT); ratio -> 1 as omega->0 (classical),
    -> hbar*omega/2kT for stiff modes (zero-point energy). torch tensor in, out.
    """
    kT = units.kB * T_K
    x = hbar * omega_rad_fs / (2.0 * kT)        # (nfreq,)
    ratio = torch.ones_like(omega_rad_fs)
    big = x > 1e-8
    ratio[big] = x[big] / torch.tanh(x[big])    # theta/kT = x*coth(x)
    return ratio


class _GPUColoredNoise:
    """GPU segment-FFT colored-noise source for QTB.

    Produces frames of shape (natoms,3) in ASE force units (eV/Angstrom), equivalent
    to a classical Langevin force of friction gamma at T but with the quantum spectrum.
    The colored series is unit-variance white noise spectrally shaped by H(omega) =
    sqrt(theta/kT) (dimensionless), scaled by the classical per-dof sigma computed in
    ASE units — so classical=True (H=1) reproduces white noise exactly.
    """

    def __init__(self, natoms, sigma_t, dt_fs, T_K, n_seg, device, dtype,
                 rng, classical=False):
        self.natoms = natoms
        self.dt = float(dt_fs)
        self.T = float(T_K)
        self.n_seg = int(n_seg)
        self.device = device
        self.dtype = dtype
        self._rng = rng
        self.classical = bool(classical)
        self._sigma = sigma_t                    # (natoms,1) eV/Angstrom, ASE units

        freqs = torch.fft.rfftfreq(self.n_seg, d=self.dt, device=device, dtype=dtype)
        omega = 2.0 * np.pi * freqs              # rad/fs
        self.omega = omega
        if classical:
            base = torch.ones_like(omega)
        else:
            base = torch.sqrt(_quantum_theta_over_kT(omega, self.T))
        self._base_filter = base                 # (nfreq,)
        self._filter = base.clone()
        self._buffer = None
        self._cursor = self.n_seg                # force regen on first call

    def set_gain(self, gain_t):
        self._filter = self._base_filter * gain_t

    def _regenerate(self):
        # white noise (n_seg, natoms*3), filter along time axis, back to time domain
        n_dof = self.natoms * 3
        white = torch.empty((self.n_seg, n_dof), device=self.device, dtype=self.dtype)
        white.normal_(generator=self._rng)
        if self.classical:
            colored = white
        else:
            spec = torch.fft.rfft(white, dim=0)
            spec = spec * self._filter.unsqueeze(-1)
            colored = torch.fft.irfft(spec, n=self.n_seg, dim=0)
        self._buffer = colored.view(self.n_seg, self.natoms, 3)
        self._cursor = 0

    def next_frame(self):
        if self._cursor >= self.n_seg:
            self._regenerate()
        frame = self._buffer[self._cursor]
        self._cursor += 1
        return self._sigma * frame               # (natoms,1)*(natoms,3)


class _GPUAdQTB:
    """adQTB-r controller (GPU). Accumulates v,R segments; adapts the noise gain.

    Delta(w) = Re[C_vF(w)] - m*gamma*C_vv(w); residual normalized by peak dissipation
    across bands (stable, bounded), gain *= exp(-lr*resid). lr default 0.005 (validated).
    """

    def __init__(self, noise, masses_3d_t, gamma_invfs, dt_fs, lr=0.005):
        self.noise = noise
        self.m = masses_3d_t                     # (natoms,1)
        self.gamma = float(gamma_invfs)
        self.dt = float(dt_fs)
        self.lr = float(lr)
        self.n_seg = noise.n_seg
        self.device = noise.device
        self.dtype = noise.dtype
        self.gain = torch.ones_like(noise._base_filter)
        self._v = torch.zeros((self.n_seg, noise.natoms, 3), device=self.device, dtype=self.dtype)
        self._r = torch.zeros((self.n_seg, noise.natoms, 3), device=self.device, dtype=self.dtype)
        self._k = 0
        self.last_residual_norm = float("nan")

    def record(self, v_t, R_t):
        self._v[self._k] = v_t
        self._r[self._k] = R_t
        self._k += 1
        if self._k >= self.n_seg:
            self._adapt()
            self._k = 0

    def _adapt(self):
        V = torch.fft.rfft(self._v, dim=0)
        R = torch.fft.rfft(self._r, dim=0)
        norm = self.dt / self.n_seg
        C_vF = (V.conj() * R).real * norm        # (nfreq,natoms,3)
        C_vv = (V.abs() ** 2) * norm
        m = self.m.unsqueeze(0)                  # (1,natoms,1)
        delta = C_vF - m * self.gamma * C_vv
        diss = m * self.gamma * C_vv
        delta_band = delta.mean(dim=(1, 2))      # (nfreq,)
        diss_band = diss.mean(dim=(1, 2))
        scale = float(diss_band.max().item()) + 1e-30
        resid = torch.clamp(delta_band / scale, -1.0, 1.0)
        self.gain = torch.clamp(self.gain * torch.exp(-self.lr * resid), 0.2, 5.0)
        self.noise.set_gain(self.gain)
        self.last_residual_norm = float(torch.sqrt((resid ** 2).mean()).item())


class TorchQTB(MolecularDynamics):
    """GPU-native QTB / adQTB thermostat for TorchAtoms (BBK velocity-Verlet).

    Constructor mirrors TorchLangevin so ase_runner can build it the same way.
    """

    def __init__(self, atoms, timestep, temperature=None, temperature_K=None,
                 friction=None, fixcm=True, rng=None,
                 n_seg=4096, adaptive=True, adqtb_lr=0.005, classical=False,
                 **kwargs):
        from molfm.interface.ase.torch_ext.torch_atoms import TorchAtoms
        if not isinstance(atoms, TorchAtoms):
            raise TypeError(f"TorchQTB requires a TorchAtoms instance, got {type(atoms).__name__}")
        if friction is None:
            raise TypeError("TorchQTB requires 'friction' (inverse ASE time units).")

        self.device = atoms._device
        self.dtype = atoms._dtype
        self.fr = friction                        # 1/(ASE time)
        self._rng = rng if rng else torch.Generator(device=self.device)
        self._fixcom_impl = TorchFixCom() if fixcm else None
        self.fix_com = fixcm
        self.classical = bool(classical)
        self.adaptive = bool(adaptive) and not classical
        self.n_seg = int(n_seg)
        self.adqtb_lr = float(adqtb_lr)

        from ase.md.md import process_temperature
        self.temp = units.kB * process_temperature(temperature, temperature_K, "eV")
        self.T_K = self.temp / units.kB

        MolecularDynamics.__init__(self, atoms, timestep, **kwargs)
        self.updatevars()
        logger.info(f"TorchQTB: dt={timestep:.4f} ASE ({timestep/units.fs:.3f} fs), "
                    f"T={self.T_K:.1f}K, friction={self.fr:.5f}/ASE "
                    f"({self.fr*units.fs:.5f}/fs), n_seg={self.n_seg}, "
                    f"adaptive={self.adaptive}, classical={self.classical}")

    def set_temperature(self, temperature=None, temperature_K=None):
        from ase.md.md import process_temperature
        self.temp = units.kB * process_temperature(temperature, temperature_K, "eV")
        self.T_K = self.temp / units.kB
        self.updatevars()

    def updatevars(self):
        atoms = self.atoms
        masses = atoms._torch_masses                       # (natoms,) amu
        self._m = masses.unsqueeze(-1)                     # (natoms,1)
        self._inv_m = 1.0 / self._m
        natoms = len(atoms)

        # Classical Langevin per-step random-force std in ASE units:
        #   sigma = sqrt(2 m gamma kT / dt)   (gamma, dt, kT all in ASE units)
        sigma = torch.sqrt(2.0 * self._m * self.fr * self.temp / self.dt)

        dt_fs = float(self.dt / units.fs)
        self._noise = _GPUColoredNoise(
            natoms=natoms, sigma_t=sigma, dt_fs=dt_fs, T_K=self.T_K,
            n_seg=self.n_seg, device=self.device, dtype=self.dtype,
            rng=self._rng, classical=self.classical,
        )
        self.adq = (
            _GPUAdQTB(self._noise, self._m, self.fr * units.fs, dt_fs, lr=self.adqtb_lr)
            if self.adaptive else None
        )

    def step(self, forces=None):
        atoms = self.atoms
        if forces is None:
            forces = atoms.get_forces(md=True, copy=False)

        v = atoms.get_velocities()
        x = atoms.get_positions()
        R = self._noise.next_frame()                       # (natoms,3) eV/Angstrom
        if self.fix_com:
            R = R - R.mean(dim=0, keepdim=True)            # no net random force

        g, dt = self.fr, self.dt
        # half kick
        v = v + 0.5 * dt * (forces - g * self._m * v + R) * self._inv_m
        # drift
        atoms.set_positions(x + dt * v, apply_constraint=True)
        if self._fixcom_impl is not None and not any(
            isinstance(c, FixCom) for c in atoms.constraints
        ):
            atoms.set_positions(
                self._fixcom_impl.adjust_positions(atoms, atoms.get_positions()),
                apply_constraint=False,
            )
        if self.adq is not None:
            self.adq.record(v, R)

        forces = atoms.get_forces(md=True, copy=False)
        # second half kick with the SAME R
        v = v + 0.5 * dt * (forces - g * self._m * v + R) * self._inv_m
        atoms.set_momenta(v * self._m, apply_constraint=True)
        if self._fixcom_impl is not None and not any(
            isinstance(c, FixCom) for c in atoms.constraints
        ):
            atoms.set_momenta(
                self._fixcom_impl.adjust_momenta(atoms, atoms.get_momenta()),
                apply_constraint=False,
            )
        return forces
