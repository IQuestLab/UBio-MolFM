# Description: Pure torch Atoms class for fully GPU-driven MD.

import copy
from typing import Sequence

import numpy as np
import torch
from ase import Atoms
from ase.cell import Cell
from ase.calculators.singlepoint import SinglePointCalculator

from molfm.interface.ase.torch_ext.torch_constraints import TORCH_CONSTRAINT_REGISTRY


class TorchAtoms(Atoms):
    """ASE-like Atoms object backed entirely by torch tensors.

    The internal state (positions, momenta, forces, cell, masses) lives on the
    configured ``torch.device``.  Public getters/setters return/accept torch
    tensors so that a torch-native MD integrator can run with zero CPU/GPU
    round-trips.

    Notes
    -----
    This class is intentionally stripped down for GPU MD only.  Legacy ASE
    tools that rely on ``self.arrays`` may see stale data because numpy caches
    are updated on demand (e.g. when exporting to a standard ``Atoms``).
    """

    def __init__(
        self,
        *args,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float64,
        **kwargs,
    ):
        self._device = torch.device(device)
        self._dtype = dtype

        super().__init__(*args, **kwargs)

        # Positions: pop from arrays, but keep the value if super().__init__
        # already set it via the overridden set_positions.
        pos = self.arrays.pop("positions", None)
        if pos is not None:
            self._torch_positions = torch.from_numpy(pos).to(
                device=self._device, dtype=self._dtype
            )
        elif not hasattr(self, "_torch_positions"):
            self._torch_positions = None

        # Momenta
        mom = self.arrays.pop("momenta", None)
        if mom is not None:
            self._torch_momenta = torch.from_numpy(mom).to(
                device=self._device, dtype=self._dtype
            )
        elif not hasattr(self, "_torch_momenta"):
            self._torch_momenta = None

        # Cached forces
        forces = self.arrays.pop("forces", None)
        if forces is not None:
            self._torch_forces = torch.from_numpy(forces).to(
                device=self._device, dtype=self._dtype
            )
        elif not hasattr(self, "_torch_forces"):
            self._torch_forces = None

        # Numbers
        numbers = self.arrays.pop("numbers", None)
        self._torch_numbers = (
            torch.from_numpy(numbers).to(device=self._device, dtype=torch.uint8)
            if numbers is not None
            else None
        )

        # Masses
        masses = self.arrays.pop("masses", None)
        if masses is not None:
            self._torch_masses = torch.from_numpy(masses).to(
                device=self._device, dtype=self._dtype
            )
        else:
            from ase.data import atomic_masses

            self._torch_masses = torch.from_numpy(
                atomic_masses[self._torch_numbers.cpu().numpy()]
            ).to(device=self._device, dtype=self._dtype)

        # Cell
        self._torch_cell = torch.from_numpy(np.asarray(self.cell)).to(
            device=self._device, dtype=self._dtype
        )

        self._torch_pbc = torch.from_numpy(np.asarray(self.pbc)).to(
            device=self._device, dtype=self._dtype
        )
        self._n_atoms = len(pos) if pos is not None else 0
        self._torch_constraint_impl_cache: dict[type, object] = {}

    def __len__(self):
        return self._n_atoms

    @property
    def pbc_tensor(self):
        """Return PBC as a torch tensor on the same device."""
        return torch.from_numpy(self.pbc).to(device=self._device)

    def has(self, name):
        """Check if an array exists, considering torch-backed fields."""
        if name == "positions":
            return self._torch_positions is not None
        if name == "momenta":
            return self._torch_momenta is not None
        if name == "forces":
            return self._torch_forces is not None
        return super().has(name)

    def copy(self):
        """Return a deep copy of this TorchAtoms instance as a TorchAtoms."""
        atoms = self.__class__.__new__(self.__class__)

        atoms._device = self._device
        atoms._dtype = self._dtype
        atoms._n_atoms = self._n_atoms
        atoms._torch_positions = (
            self._torch_positions.clone() if self._torch_positions is not None else None
        )
        atoms._torch_momenta = (
            self._torch_momenta.clone() if self._torch_momenta is not None else None
        )
        atoms._torch_forces = (
            self._torch_forces.clone() if self._torch_forces is not None else None
        )
        atoms._torch_numbers = self._torch_numbers.clone()
        atoms._torch_masses = self._torch_masses.clone()
        atoms._torch_cell = self._torch_cell.clone()
        atoms._torch_pbc = self._torch_pbc.clone()
        atoms._torch_constraint_impl_cache = {}

        atoms._pbc = self.pbc.copy()
        atoms._cellobj = Cell(self._torch_cell.detach().cpu().numpy())
        atoms.info = copy.deepcopy(self.info)
        atoms.constraints = copy.deepcopy(self.constraints)
        atoms.calc = self.calc

        atoms.arrays = {}
        for name, arr in self.arrays.items():
            if name not in ("positions", "momenta", "forces", "numbers", "masses"):
                atoms.arrays[name] = arr.copy()

        celldisp = getattr(self, "_celldisp", None)
        atoms._celldisp = celldisp.copy() if celldisp is not None else None

        return atoms

    # ------------------------------------------------------------------
    # Construction from a standard ASE Atoms
    # ------------------------------------------------------------------
    @classmethod
    def from_ase_atoms(
        cls,
        atoms: Atoms,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float64,
    ) -> "TorchAtoms":
        """Build a ``TorchAtoms`` from a standard ASE ``Atoms`` object."""
        kwargs = {
            "numbers": atoms.get_atomic_numbers(),
            "positions": atoms.get_positions(),
            "cell": atoms.get_cell(),
            "pbc": atoms.pbc.copy(),
            "info": copy.deepcopy(atoms.info),
            "device": device,
            "dtype": dtype,
        }

        if atoms.has("masses"):
            kwargs["masses"] = atoms.get_masses()
        if atoms.has("momenta"):
            kwargs["momenta"] = atoms.get_momenta()
        if atoms.has("tags"):
            kwargs["tags"] = atoms.get_tags()
        if atoms.has("initial_magmoms"):
            kwargs["magmoms"] = atoms.get_initial_magnetic_moments()
        if atoms.has("initial_charges"):
            kwargs["charges"] = atoms.get_initial_charges()

        celldisp = getattr(atoms, "_celldisp", None)
        if celldisp is not None:
            kwargs["celldisp"] = celldisp.copy()

        new_atoms = cls(**kwargs)
        new_atoms.constraints = copy.deepcopy(atoms.constraints)

        if atoms.calc is not None:
            new_atoms.calc = atoms.calc

        # Synchronise any extra arrays that were not handled above.
        for name, arr in atoms.arrays.items():
            if name not in new_atoms.arrays and name not in (
                "positions",
                "momenta",
                "forces",
                "numbers",
                "masses",
            ):
                new_atoms.set_array(name, arr.copy())

        return new_atoms

    # ------------------------------------------------------------------
    # Construction directly from tensors (bypasses ASE round-trip)
    # ------------------------------------------------------------------
    @classmethod
    def from_tensors(
        cls,
        numbers: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float64,
    ) -> "TorchAtoms":
        """Build a ``TorchAtoms`` directly from tensors.

        Avoids GPU->CPU->GPU memory copies and float32->float64->float32
        precision conversions that occur when routing through ``ASE Atoms``.
        All input tensors are moved to the target ``device``/``dtype``.
        """
        atoms = cls.__new__(cls)
        atoms._device = torch.device(device)
        atoms._dtype = dtype
        atoms._n_atoms = int(numbers.numel())

        atoms._torch_positions = positions.to(device=device, dtype=dtype)
        atoms._torch_numbers = numbers.to(device=device, dtype=torch.uint8)
        atoms._torch_cell = cell.to(device=device, dtype=dtype)
        atoms._torch_pbc = pbc.to(device=device, dtype=torch.bool)

        atoms._torch_momenta = None
        atoms._torch_forces = None
        atoms._torch_constraint_impl_cache = {}

        # ASE compatibility stubs (only created when CPU export is needed).
        atoms._pbc = atoms._torch_pbc.detach().cpu().numpy()
        atoms._cellobj = Cell(atoms._torch_cell.detach().cpu().numpy())
        atoms.info = {}
        atoms.constraints = []
        atoms.arrays = {}
        atoms._celldisp = None

        # Masses from atomic numbers (needs one CPU round-trip for numpy indexing).
        from ase.data import atomic_masses

        atoms._torch_masses = torch.from_numpy(
            atomic_masses[atoms._torch_numbers.cpu().numpy()]
        ).to(device=device, dtype=dtype)

        return atoms

    # ------------------------------------------------------------------
    # Export to a standard ASE Atoms
    # ------------------------------------------------------------------
    def to_ase_atoms(self) -> Atoms:
        """Export a standard ASE ``Atoms`` object with the latest state."""
        positions = self._torch_positions.detach().cpu().numpy()
        numbers = self._torch_numbers.detach().cpu().numpy()
        cell = self._torch_cell.detach().cpu().numpy()
        pbc = self.pbc.copy()

        atoms = Atoms(
            numbers=numbers,
            positions=positions,
            cell=cell,
            pbc=pbc,
        )

        if self._torch_momenta is not None:
            atoms.set_momenta(self._torch_momenta.detach().cpu().numpy())
        if self._torch_masses is not None:
            atoms.set_masses(self._torch_masses.detach().cpu().numpy())
        if self._torch_forces is not None:
            atoms.set_array("forces", self._torch_forces.detach().cpu().numpy())

        atoms.info = copy.deepcopy(self.info)
        atoms.constraints = copy.deepcopy(self.constraints)

        calc = self.calc
        if calc is not None:
            energy = None
            forces = None
            results = getattr(calc, "results", None)
            if isinstance(results, dict):
                energy = results.get("energy", None)
                forces = results.get("forces", None)

            if isinstance(energy, torch.Tensor):
                energy = float(energy.detach().cpu().item())
            elif energy is not None:
                energy = float(energy)

            if isinstance(forces, torch.Tensor):
                forces = forces.detach().cpu().numpy()
            elif forces is not None:
                forces = np.asarray(forces)

            singlepoint_results = {}
            if energy is not None:
                singlepoint_results["energy"] = energy
            if forces is not None:
                singlepoint_results["forces"] = forces

            if singlepoint_results:
                atoms.calc = SinglePointCalculator(atoms, **singlepoint_results)
            else:
                atoms.calc = None
        else:
            atoms.calc = None

        for name, arr in self.arrays.items():
            if name not in ("positions", "momenta", "forces", "numbers", "masses"):
                atoms.set_array(name, arr.copy())

        celldisp = getattr(self, "_celldisp", None)
        if celldisp is not None:
            atoms._celldisp = celldisp.copy()

        return atoms

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_tensor(self, obj, dtype=None) -> torch.Tensor:
        if isinstance(obj, torch.Tensor):
            return obj.to(device=self._device, dtype=dtype or self._dtype)
        return torch.as_tensor(obj, device=self._device, dtype=dtype or self._dtype)


    def _check_torch_constraint(self, constraint):
        """Raise if a constraint has no torch-native implementation."""
        if type(constraint) not in TORCH_CONSTRAINT_REGISTRY:
            raise RuntimeError(
                f"Constraint {type(constraint).__name__} is not registered in "
                f"TORCH_CONSTRAINT_REGISTRY. Only torch-native constraints are "
                f"supported by TorchAtoms."
            )

    def _get_torch_constraint_impl(self, constraint):
        """Return a cached torch-native constraint helper for a constraint type."""
        constraint_type = type(constraint)
        impl = self._torch_constraint_impl_cache.get(constraint_type)
        if impl is None:
            self._check_torch_constraint(constraint)
            impl = TORCH_CONSTRAINT_REGISTRY[constraint_type]()
            self._torch_constraint_impl_cache[constraint_type] = impl
        return impl

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------
    def _torch_wrap_positions(self, positions, cell, pbc=None, center=(0.5, 0.5, 0.5)):
        """Pure-torch equivalent of ase.geometry.wrap_positions."""
        if pbc is None:
            pbc = self.pbc
        if isinstance(pbc, bool):
            pbc = [pbc] * 3
        pbc = list(pbc)
        center_t = torch.as_tensor(center, device=positions.device, dtype=positions.dtype)
        # fractional = positions @ inv(cell)  ==  solve(cell.T, positions.T).T
        fractional = torch.linalg.solve(cell.T, positions.T).T
        shift = torch.zeros_like(fractional)
        for i, periodic in enumerate(pbc):
            if periodic:
                shift[:, i] = torch.floor(fractional[:, i] + center_t[i])
        return (fractional - shift) @ cell

    @classmethod
    def wrap_positions_tensor(cls, positions, cell, pbc=None, center=(0.5, 0.5, 0.5)):
        """Wrap positions to unit cell — same semantics as :func:`ase.geometry.wrap_positions`.

        ``center=(0.5, 0.5, 0.5)`` (the default) wraps positions to [0, cell).
        ``center=(0.0, 0.0, 0.0)`` wraps positions to (-cell/2, cell/2].
        """
        if pbc is None:
            raise ValueError("pbc must be provided when using wrap_positions_tensor.")
        if isinstance(pbc, bool):
            pbc = [pbc] * 3
        pbc = list(pbc)
        center_t = torch.as_tensor(center, device=positions.device, dtype=positions.dtype)
        shift = center_t - 0.5
        fractional = torch.linalg.solve(cell.T, positions.T).T - shift
        for i, periodic in enumerate(pbc):
            if periodic:
                fractional[:, i] %= 1.0
                fractional[:, i] += shift[i]
        return fractional @ cell

    def wrap(self, center=(0.5, 0.5, 0.5), pbc=None, pretty_translation=False, eps=1e-7):
        if pretty_translation:
            return super().wrap(
                center=center, pbc=pbc, pretty_translation=pretty_translation, eps=eps
            )
        wrapped = self.wrap_positions_tensor(
            self._torch_positions, self._torch_cell, pbc=pbc if pbc is not None else self.pbc, center=center
        )
        self.set_positions(wrapped)
        return self

    def get_positions(self, wrap=False, **wrap_kw):
        if self._torch_positions is None:
            raise RuntimeError("Positions have not been set.")
        if wrap:
            # Fast-path for common kwargs; fallback to ASE for exotic options.
            if not wrap_kw or set(wrap_kw.keys()).issubset({"pbc", "center"}):
                pbc = wrap_kw.get("pbc", self.pbc)
                center = wrap_kw.get("center", (0.5, 0.5, 0.5))
                return self.wrap_positions_tensor(
                    self._torch_positions, self._torch_cell, pbc=pbc, center=center
                )
            from ase.geometry import wrap_positions
            pos_np = wrap_positions(
                self._torch_positions.detach().cpu().numpy(),
                self.cell,
                **wrap_kw,
            )
            return torch.from_numpy(pos_np).to(device=self._device, dtype=self._dtype)
        return self._torch_positions

    def set_positions(self, newpositions, apply_constraint=True):
        newpositions = self._ensure_tensor(newpositions)
        if self.constraints and apply_constraint:
            for constraint in self.constraints:
                newpositions = self._get_torch_constraint_impl(constraint).adjust_positions(
                    self, newpositions
                )
        self._torch_positions = newpositions

    # ------------------------------------------------------------------
    # Velocities / Momenta
    # ------------------------------------------------------------------
    def get_velocities(self):
        if self._torch_momenta is None:
            return torch.zeros(
                (len(self), 3), device=self._device, dtype=self._dtype
            )
        return self._torch_momenta / self._torch_masses.unsqueeze(-1)

    def set_velocities(self, velocities):
        self.set_momenta(self._torch_masses.unsqueeze(-1) * self._ensure_tensor(velocities))

    def get_momenta(self):
        if self._torch_momenta is None:
            return torch.zeros(
                (len(self), 3), device=self._device, dtype=self._dtype
            )
        return self._torch_momenta

    def set_momenta(self, momenta, apply_constraint=True):
        if momenta is None:
            self._torch_momenta = None
            return
        momenta = self._ensure_tensor(momenta)
        if apply_constraint and self.constraints:
            for constraint in self.constraints:
                momenta = self._get_torch_constraint_impl(constraint).adjust_momenta(
                    self, momenta
                )
        self._torch_momenta = momenta

    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------
    def get_forces(self, apply_constraint=True, md=False, copy=True):
        if self.calc is None:
            if self._torch_forces is not None:
                forces = self._torch_forces
            else:
                raise RuntimeError(
                    "Atoms object has no calculator and no cached forces."
                )
        else:
            from molfm.interface.ase.calculator.e2former_calculator import E2FormerCalculator
            # if not isinstance(self.calc, E2FormerCalculator):
            #     raise TypeError(
            #         "TorchAtoms only supports E2FormerCalculator for GPU MD."
            #     )
            forces = self.calc.get_forces(self)
            if not isinstance(forces, torch.Tensor):
                forces = torch.as_tensor(
                    np.asarray(forces), device=self._device, dtype=self._dtype
                )

        needs_working_copy = copy or (apply_constraint and bool(self.constraints))
        if needs_working_copy:
            forces = forces.clone()

        if apply_constraint and self.constraints:
            for constraint in self.constraints:
                torch_impl = self._get_torch_constraint_impl(constraint)
                if md and hasattr(torch_impl, "redistribute_forces_md"):
                    # MD-specific redistribution already includes any force projection
                    # required by the constraint, so do not apply adjust_forces again.
                    forces = torch_impl.redistribute_forces_md(self, forces, rand=False)
                elif hasattr(torch_impl, "adjust_forces"):
                    forces = torch_impl.adjust_forces(self, forces)
        return forces

    def set_forces(self, forces, apply_constraint=True):
        if forces is None:
            self._torch_forces = None
            return
        forces = self._ensure_tensor(forces)
        if apply_constraint and self.constraints:
            for constraint in self.constraints:
                torch_impl = self._get_torch_constraint_impl(constraint)
                if hasattr(torch_impl, "adjust_forces"):
                    forces = torch_impl.adjust_forces(self, forces)
        self._torch_forces = forces

    # ------------------------------------------------------------------
    # Energy / Temperature (ASE compatibility)
    # ------------------------------------------------------------------
    def get_kinetic_energy(self):
        """Return kinetic energy using torch momenta (eV)."""
        if self._torch_momenta is None:
            return 0.0
        p = self._torch_momenta
        m = self._torch_masses.unsqueeze(-1)
        return float(0.5 * (p * (p / m)).sum())

    # ------------------------------------------------------------------
    # ASE-compatible property getters for trajectory I/O and calculators
    # ------------------------------------------------------------------
    @property
    def positions(self):
        if hasattr(self, "_torch_positions"):
            if self._torch_positions is None:
                raise AttributeError("Positions have not been set.")
            return self._torch_positions.detach().cpu().numpy()
        return self.arrays.get("positions")

    @property
    def numbers(self):
        if hasattr(self, "_torch_numbers"):
            return self._torch_numbers.detach().cpu().numpy()
        return self.arrays.get("numbers")

    @property
    def momenta(self):
        if hasattr(self, "_torch_momenta"):
            if self._torch_momenta is None:
                return None
            return self._torch_momenta.detach().cpu().numpy()
        return self.arrays.get("momenta")

    @property
    def forces(self):
        if hasattr(self, "_torch_forces"):
            if self._torch_forces is None:
                return None
            return self._torch_forces.detach().cpu().numpy()
        return self.arrays.get("forces")

    # ------------------------------------------------------------------
    # Cell / Masses / Numbers
    # ------------------------------------------------------------------
    def get_cell(self, complete=False):
        return self._torch_cell

    def get_pbc(self):
        return self._torch_pbc

    def set_cell(self, cell, scale_atoms=False, apply_constraint=True):
        if isinstance(cell, torch.Tensor):
            cell_t = self._ensure_tensor(cell)
            cell_obj = None
        elif isinstance(cell, Cell):
            cell_t = self._ensure_tensor(cell.array)
            cell_obj = cell
        else:
            cell_obj = Cell.new(cell)
            cell_t = self._ensure_tensor(cell_obj.array)

        if apply_constraint and hasattr(self, "_constraints") and self.constraints:
            for constraint in self.constraints:
                self._check_torch_constraint(constraint)
                if hasattr(constraint, "adjust_cell"):
                    if cell_obj is None:
                        cell_obj = Cell(cell_t.detach().cpu().numpy())
                    constraint.adjust_cell(self, cell_obj)

        if scale_atoms and self._torch_positions is not None:
            M_t = torch.linalg.solve(self._torch_cell, cell_t)
            self._torch_positions = self._torch_positions @ M_t

        self._torch_cell = cell_t
        self._cellobj = Cell(cell_t.detach().cpu().numpy())

    def get_masses(self):
        return self._torch_masses.detach().cpu().numpy()

    def set_masses(self, masses="defaults"):
        super().set_masses(masses)
        self._torch_masses = torch.from_numpy(super().get_masses()).to(
            device=self._device, dtype=self._dtype
        )

    def get_atomic_numbers(self):
        return self._torch_numbers

    def set_atomic_numbers(self, numbers):
        super().set_atomic_numbers(numbers)
        self._torch_numbers = torch.from_numpy(self.arrays["numbers"]).to(
            device=self._device, dtype=torch.uint8
        )
        self._torch_masses = torch.from_numpy(super().get_masses()).to(
            device=self._device, dtype=self._dtype
        )
