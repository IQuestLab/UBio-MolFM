# Description: High-performance single-GPU ASERunner for GPU-native simulations.

import atexit
import math
import os
import signal
import torch
import numpy as np
import time
from typing import Optional, Union, Dict, Any, List, Tuple
from loguru import logger

try:
    import swanlab
except ImportError:  # optional experiment tracker
    swanlab = None

from ase import units, Atoms
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from molfm.interface.ase.torch_ext.torch_atoms import TorchAtoms
from molfm.interface.ase.torch_ext.torch_lbfgs import TorchLBFGS
from molfm.interface.ase.torch_ext.torch_langevin import TorchLangevin
from molfm.interface.ase.calculator.e2former_calculator import E2FormerCalculator
from molfm.interface.ase.torch_ext.h5_trajectory import H5Trajectory
from molfm.interface.ase.torch_ext.torch_constraints import FixCom
from molfm.interface.ase.torch_ext.torch_mc_barostat import (
    TorchMCBarostat, build_molecule_index, molecule_first_atom, masses_from_numbers,
)


def _resolve_array(value, *, kind: str, required: bool):
    """Resolve positions / numbers / cell into an ndarray.

    str / PathLike ending in ``.npy`` → loaded via ``np.load(path, mmap_mode="r")``.
    ndarray / list → ``np.asarray`` (no-copy when already ndarray).
    ``None`` → returned as ``None`` unless ``required``.

    Note: the mmap array will typically be materialized to RAM when ASE's
    ``Atoms()`` constructor calls ``np.asarray(..., dtype=float)``. mmap therefore
    saves the peak memory during the load step, not the steady state.
    """
    if value is None:
        if required:
            raise ValueError(f"{kind}: must provide ndarray, list, or .npy path")
        return None

    if isinstance(value, (str, os.PathLike)):
        path = os.path.abspath(os.path.expanduser(str(value)))
        if not path.endswith(".npy"):
            raise ValueError(f"{kind}: only .npy file paths supported, got: {value}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{kind} npy not found: {path}")
        logger.info(f"Loading {kind} from npy (mmap=r): {path}")
        return np.load(path, mmap_mode="r")

    return np.asarray(value)


def _validate_array_shapes(pos, nums, cell):
    """Cross-check positions / numbers / cell shapes."""
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")
    if nums.ndim != 1:
        raise ValueError(f"numbers must have shape (N,), got {nums.shape}")
    if pos.shape[0] != nums.shape[0]:
        raise ValueError(
            f"positions ({pos.shape[0]}) and numbers ({nums.shape[0]}) atom count mismatch"
        )
    if cell is not None and tuple(cell.shape) != (3, 3):
        raise ValueError(f"cell must have shape (3, 3), got {cell.shape}")


class ASERunner:
    """
    A unified manager for running structure relaxation and molecular dynamics
    entirely on a single GPU.
    """
    def __init__(
        self,
        input_path: Optional[str] = None,
        positions: Optional[Union[np.ndarray, List[List[float]], str]] = None,
        numbers: Optional[Union[np.ndarray, List[int], str]] = None,
        cell: Optional[Union[np.ndarray, List[List[float]], str]] = None,
        pbc: Union[bool, List[bool]] = True,
        checkpoint_path: Optional[str] = None,
        config_path: str = None,
        head_name: str = "omol25",
        device: str = "cuda:0",
        use_faiss: bool = True,
        use_tf32: bool = False,
        use_compile: bool = False,
        recompute_budget: float = -1.0,
        work_dir: str = "runs",
        name: str = "mfm_md",
        use_swanlab: bool = False,
        project_name: str = "MolFM-MD",
        full_sync_interval: int = 10,
        log_interval: int = 10,
        resume_step: Optional[int] = None,
        fixcm: bool = True,
        save_in_fp32: bool = False,
        charge: int = 0,
        multiplicity: int = 1,
        swanlab_config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        ensemble: str = "nvt",
        pressure_bar: float = 1.0,
        barostat_interval: int = 50,
        barostat_dlnV: float = 0.002,
        molecule_source: str = "auto",
        barostat_seed: int = 12345,
        thermostat: str = "langevin",
        qtb_adaptive: bool = True,
        qtb_classical: bool = False,
        qtb_n_seg: int = 4096,
        qtb_adqtb_lr: float = 0.005,
    ):
        torch.backends.cudnn.benchmark = True

        self.device = device
        self.head_name = head_name
        self.full_sync_interval = full_sync_interval
        self.log_interval = log_interval
        self.resume_step = resume_step
        self.wrap = True
        self.fixcm = fixcm
        self.save_in_fp32 = save_in_fp32
        self.charge = charge
        self.multiplicity = multiplicity
        # MD seed: drives the Maxwell-Boltzmann initial velocities and the
        # thermostat noise stream, so a run is reproducible for a fixed seed.
        self.seed = int(seed)
        os.makedirs(work_dir, exist_ok=True)

        # NPT / barostat configuration. The barostat object itself is built lazily
        # in run_md() once the configuration (and molecule map) is known.
        self.ensemble = str(ensemble).lower()
        if self.ensemble not in ("nvt", "npt", "nve"):
            raise ValueError(f"ensemble must be 'nvt', 'npt', or 'nve', got {ensemble!r}")
        self.pressure_bar = pressure_bar
        self.barostat_interval = barostat_interval
        self.barostat_dlnV = barostat_dlnV
        self.molecule_source = molecule_source
        self.barostat_seed = barostat_seed
        self.barostat: Optional[TorchMCBarostat] = None
        self._total_mass_amu = 0.0

        # Thermostat selection: "langevin" (white noise) or "qtb" (quantum thermal
        # bath, colored noise for nuclear quantum effects).
        self.thermostat = str(thermostat).lower()
        if self.thermostat not in ("langevin", "qtb"):
            raise ValueError(f"thermostat must be 'langevin' or 'qtb', got {thermostat!r}")
        self.qtb_adaptive = qtb_adaptive
        self.qtb_classical = qtb_classical
        self.qtb_n_seg = qtb_n_seg
        self.qtb_adqtb_lr = qtb_adqtb_lr


        # Store configuration for lazy initialization in _setup().
        self._input_path = input_path
        self._positions = positions
        self._numbers = numbers
        self._cell = cell
        self._pbc = pbc
        # The geometry input is read at most once, and is needed even on a resume
        # (to verify the trajectory describes the same system).
        self._input_atoms_cache: Optional[Atoms] = None
        self.dyn = None

        self.calc = E2FormerCalculator(
            checkpoint_path=checkpoint_path, config_path=config_path, head_name=head_name,
            device=self.device, use_faiss=use_faiss, use_tf32=use_tf32, use_compile=use_compile,
            recompute_budget=recompute_budget,
            charge=charge, multiplicity=multiplicity,
        )

        # Atoms are created lazily in _setup().
        self.atoms = None
        self.full_numbers = None
        self.full_pbc = None
        self.full_natoms = 0

        self.relax_path = os.path.join(work_dir, f"{name}_relax.h5")
        self.md_path = os.path.join(work_dir, f"{name}_md.h5")

        # Logger setup.
        # Init runs AFTER trajectory paths are resolved so we can publish the
        # absolute on-disk paths to the run config (useful for re-locating output
        # later from the SwanLab UI).
        self.use_swanlab = use_swanlab
        if self.use_swanlab and swanlab is None:
            logger.warning(
                "use_swanlab=True but the swanlab package is not installed — "
                "continuing without experiment tracking."
            )
            self.use_swanlab = False
        if self.use_swanlab:
            # No default workspace: unset means "the account's own default
            # workspace", which is the only thing we can assume about a user.
            workspace = os.environ.get("swanlab_workspace") or None
            swan_cfg = dict(swanlab_config) if swanlab_config else {}
            swan_cfg.setdefault("device", self.device)
            swan_cfg["work_dir"] = os.path.abspath(work_dir)
            swan_cfg["relax_traj_path"] = os.path.abspath(self.relax_path)
            swan_cfg["md_traj_path"] = os.path.abspath(self.md_path)
            swanlab.init(project=project_name, experiment_name=name, workspace=workspace, config=swan_cfg)

        # State tracking for H5 writing and logging.
        self.trajectory: Optional[H5Trajectory] = None
        self.step_offset = 0

        self._metrics_cache: Optional[Dict[str, float]] = None
        self._has_atoms = False
        self._stop_requested = False
        self._cleanup_registered = False
        self._register_cleanup_handlers()

    def _open_trajectory(self, path: str, fresh: bool = False):
        """Create the trajectory writer."""
        if self.trajectory is not None:
            self._close_trajectory()

        if fresh and os.path.exists(path):
            self._backup_trajectory_file(path)

        mode = "w" if fresh else "a"
        try:
            self.trajectory = self._get_traj_writer(path, mode)
        except Exception as exc:
            if fresh:
                raise
            logger.error(
                f"Failed to open trajectory {path} in append mode — existing "
                f"file may be corrupted: {exc}.  Recreating from scratch "
                f"(all prior frames will be lost)."
            )
            self.trajectory = self._get_traj_writer(path, "w")
        return self.trajectory

    @staticmethod
    def _backup_trajectory_file(path: str):
        """Rename path to path.back.N, incrementing N to avoid collisions."""
        n = 1
        while True:
            backup = f"{path}.back.{n}"
            if not os.path.exists(backup):
                break
            n += 1
        os.rename(path, backup)
        logger.info(f"Backed up existing trajectory: {os.path.basename(path)} -> {os.path.basename(backup)}")

    def _close_trajectory(self, message: Optional[str] = None):
        """Close the trajectory writer once the task finishes."""
        if self.trajectory is not None:
            self.trajectory.close()
            self.trajectory = None
            if message:
                logger.info(message)

    def _register_cleanup_handlers(self):
        """Register process-exit cleanup once per runner."""
        if self._cleanup_registered:
            return
        self._cleanup_registered = True
        atexit.register(self.close)

        def _handle_signal(signum, _frame):
            logger.warning(
                f"Received signal {signum}, "
                "requesting graceful stop (will finish current step and save final frame)."
            )
            self._stop_requested = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle_signal)
            except Exception:
                pass

    def _should_save_step(self, step: int) -> bool:
        return step > 0 and step % self.full_sync_interval == 0

    def _should_log_step(self, step: int) -> bool:
        return step > 0 and step % self.log_interval == 0

    def _load_full_atoms( self, input_path, pos, nums, cell, pbc,) -> Atoms:
        """Load the full system.

        ``pos`` / ``nums`` / ``cell`` may each be an ndarray, a list, or a path
        to a ``.npy`` file (loaded via mmap). See ``_resolve_array``.
        """
        if input_path:
            if input_path.endswith(".gro"):
                from molfm.interface.data_utils import gro_to_ase_atoms
                return gro_to_ase_atoms(input_path, use_box=True)
            if input_path.endswith(".pdb"):
                from molfm.interface.data_utils import mda_to_ase_atoms
                return mda_to_ase_atoms(input_path, use_box=True)
            return read(input_path)

        pos_arr = _resolve_array(pos, kind="positions", required=True)
        nums_arr = _resolve_array(nums, kind="numbers", required=True)
        cell_arr = _resolve_array(cell, kind="cell", required=False)
        _validate_array_shapes(pos_arr, nums_arr, cell_arr)

        return Atoms(
            numbers=nums_arr,
            positions=pos_arr,
            cell=cell_arr if cell_arr is not None else [10, 10, 10],
            pbc=pbc,
        )

    def _input_atoms(self) -> Optional[Atoms]:
        """The geometry input, read at most once. None when none was given."""
        if self._input_path is None and self._numbers is None and self._positions is None:
            return None
        if self._input_atoms_cache is None:
            self._input_atoms_cache = self._load_full_atoms(
                self._input_path, self._positions, self._numbers, self._cell, self._pbc
            )
        return self._input_atoms_cache

    def _check_resume_matches_input(self, resumed: Atoms, resume_path: str):
        """Abort when the trajectory being resumed holds a different system.

        A resume ignores the geometry input entirely — positions, cell and
        velocities all come from the trajectory. With resume on by default and a
        default ``name``, a second run in the same work_dir would otherwise
        silently continue the *previous* system's trajectory while reporting the
        new input as its source. Comparing atomic numbers catches that; the cell
        is deliberately not compared, because NPT legitimately changes it.
        """
        reference = self._input_atoms()
        if reference is None:
            return
        want = reference.get_atomic_numbers()
        got = resumed.get_atomic_numbers()
        if want.shape == got.shape and np.array_equal(want, got):
            return
        raise ValueError(
            f"Refusing to resume from {resume_path}: it holds a different system than "
            f"the geometry input — input is {reference.get_chemical_formula()} "
            f"({len(reference)} atoms), trajectory is {resumed.get_chemical_formula()} "
            f"({len(resumed)} atoms). Use a different --name / --work_dir, remove the "
            "old trajectory, or pass --resume False to start from the input."
        )

    def _initialize_md_velocities_if_needed(self, temperature_K: float = 300.0):
        """Initialize MB velocities when none exist and MD hasn't advanced."""
        if len(self.atoms) == 0:
            return

        if self.step_offset > 0:
            return
        momenta = getattr(self.atoms, "_torch_momenta", None)
        if momenta is not None and momenta.numel() > 0 and momenta.abs().max().item() > 0:
            return

        MaxwellBoltzmannDistribution(
            self.atoms,
            temperature_K=temperature_K,
            rng=np.random.default_rng(self.seed),
        )

    # ------------------------------------------------------------------
    # Unified setup (initial or resume)
    # ------------------------------------------------------------------

    def _setup(self, resume_path: Optional[str] = None, reset_step_offset: bool = False):
        """Unified initialization and resume entry.

        When atoms have already been initialized and no resume file is found,
        the current in-memory state is preserved.  This allows sequential
        ``relax()`` → ``run_md()`` calls without losing the relaxed structure.
        """
        # 1. Decide the source: resume from H5 or load from input.
        atoms_full, step_offset, did_resume = self._resolve_source(resume_path)
        if atoms_full:
            logger.info(f"Atoms info: natoms={len(atoms_full)} | PBC={atoms_full.pbc} | Cell={atoms_full.get_cell()} | Symbols={atoms_full.get_chemical_formula()}")
        step_offset = 0 if reset_step_offset else step_offset
        if atoms_full is not None:
            atoms_full.wrap() # Warning: force to wrap here to ensure PBC consistency
            if self.full_numbers is None:
                self.full_numbers = torch.from_numpy(atoms_full.get_atomic_numbers())
                self.full_pbc = torch.from_numpy(np.asarray(atoms_full.pbc))

        # 2. If atoms are already initialized and no resume happened, keep
        #    the current state (e.g. relax() finished, now starting run_md()).
        if not did_resume and self.atoms is not None:
            self.step_offset = step_offset
            return

        # 3. Build the GPU atoms in canonical atom order from the input / H5 file.
        self.atoms = TorchAtoms(
            numbers=atoms_full.get_atomic_numbers(),
            positions=atoms_full.get_positions(),
            cell=atoms_full.get_cell(),
            pbc=atoms_full.pbc,
            device=self.device,
        )
        self.atoms.calc = self.calc
        if self.fixcm and len(self.atoms) > 0:
            self.atoms.set_constraint(FixCom())
        self._has_atoms = len(self.atoms) > 0
        self.full_natoms = len(self.atoms)
        self.step_offset = step_offset
        if atoms_full.has("momenta"):
            self.atoms.set_velocities(atoms_full.get_velocities())

    def _resolve_source(self, resume_path: Optional[str] = None):
        """Load from resume H5 or fall back to initial input.

        Returns:
            (atoms_full, step_offset, did_resume)
            did_resume is True when the structure was restored from an H5 file.
        """
        if resume_path is not None and os.path.exists(resume_path):
            restored = None
            try:
                data = H5Trajectory.read_last_frame(resume_path)
                if data["positions"] is not None and data["n_frames"] > 0:
                    # H5 is self-contained: cell, pbc, numbers are always present.
                    step_offset = data["steps"] if data["steps"] is not None else (
                        (data["n_frames"] - 1) * self.full_sync_interval
                    )
                    atoms = Atoms(
                        numbers=data["numbers"],
                        positions=data["positions"],
                        cell=data["cell"],
                        pbc=data["pbc"],
                    )
                    if data["velocities"] is not None:
                        atoms.set_velocities(data["velocities"])
                    restored = (atoms, step_offset)
            except Exception as e:
                logger.warning(f"Resume failed: {e}, falling back to initial input")

            if restored is not None:
                atoms, step_offset = restored
                # Raised outside the try above on purpose: a system mismatch is a
                # user error to fix, not something to silently fall back from.
                self._check_resume_matches_input(atoms, resume_path)
                logger.warning(
                    f"Resumed from {resume_path} at step {step_offset} — the geometry "
                    "input is ignored (positions, cell and velocities come from the "
                    "trajectory). Pass --resume False to start from the input instead."
                )
                return atoms, step_offset, True

        # Not a resume.  If atoms are already initialized, preserve the
        # current in-memory state instead of reloading from the source file.
        if self.atoms is not None:
            return None, 0, False

        atoms = self._input_atoms()
        if atoms is None:
            raise ValueError(
                "No geometry source to start from: provide input_path, or both "
                "positions and numbers."
            )
        return atoms, 0, False

    def _resolve_md_source(self, do_resume: bool = True):
        """Resolve atom source for the MD phase.

        Handles the full source-resolution pipeline: decide where atoms come
        from, log the choice, and truncate the trajectory to *resume_step*
        when applicable.

        Returns:
            (resume_path, source_tag)
            source_tag is ``'md_resume'`` | ``'relax_transition'`` | ``'initial'``.
        """
        if not do_resume:
            logger.warning("[MD] Atoms source: initial input (do_resume=False)")
            return None, "initial"

        if os.path.exists(self.md_path):
            source_tag = "md_resume"
        elif os.path.exists(self.relax_path):
            source_tag = "relax_transition"
        else:
            source_tag = "initial"

        resume_path = {
            "md_resume": self.md_path,
            "relax_transition": self.relax_path,
            "initial": None,
        }[source_tag]

        logger.warning(f"[MD] Atoms source: {source_tag}")

        if resume_path is not None and self.resume_step is not None:
            self._truncate_trajectory_to_step(resume_path)

        return resume_path, source_tag

    def _truncate_trajectory_to_step(self, resume_path: str) -> int:
        """Truncate H5 trajectory to *resume_step*.

        Returns the number of frames retained after truncation.
        """
        if self.resume_step is None:
            return -1
        n = H5Trajectory.truncate_to_step(resume_path, self.resume_step,
                                            full_sync_interval=self.full_sync_interval)
        logger.warning(
            f"Truncated {resume_path} to step <= {self.resume_step} "
            f"({n} frames retained)"
        )
        return n

    def _get_traj_writer(self, path: str, mode='a'):
        """Build the H5 storage object."""
        return H5Trajectory(
            path,
            mode,
            atomic_numbers=self.full_numbers,
            pbc=self.full_pbc,
            natoms=self.full_natoms,
            save_in_fp32=self.save_in_fp32,
            append_only=True,
        )

    def _collect_step_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        """Snapshot positions / velocities / forces plus the scalar diagnostics."""
        # Detach: the model forward runs with requires_grad=True on positions
        # (for force autograd), so the returned pos/vel/force tensors may carry
        # a grad graph.  The H5 background writer cannot convert such tensors
        # to numpy, and gradients are not needed beyond this point anyway.
        pos = self.atoms.get_positions()
        vel = self.atoms.get_velocities()
        force = self.atoms.get_forces()
        if isinstance(pos, torch.Tensor):
            pos = pos.detach()
        if isinstance(vel, torch.Tensor):
            vel = vel.detach()
        if isinstance(force, torch.Tensor):
            force = force.detach()
        epot = float(self.atoms.get_potential_energy())
        ekin = float(self.atoms.get_kinetic_energy())

        f_norms = torch.norm(force, dim=1) if hasattr(force, "norm") else torch.zeros(1)
        fmax = float(f_norms.max().item()) if f_norms.numel() > 0 else 0.0
        return pos, vel, force, epot, ekin, fmax

    def _qtb_info_dict(self, temp: float):
        """Per-frame QTB diagnostics for swanlab + H5 info.

        Returns {} when not running QTB. Keys: per-element kinetic energy + quantum
        enhancement ratio (vs the CLASSICAL target 3/2 kT_target, not the inflated
        instantaneous T), and adQTB FDT residual + noise-gain range.
        """
        if self.thermostat != "qtb":
            return {}
        def _to_np(x):
            return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        numbers = _to_np(self.atoms.get_atomic_numbers())
        mom = _to_np(self.atoms.get_momenta())
        masses = _to_np(self.atoms.get_masses())
        ke_atom = 0.5 * (mom ** 2).sum(axis=1) / masses
        kT_target = units.kB * (self.dyn.T_K if getattr(self.dyn, "T_K", None) else temp)
        classical_per_atom = 1.5 * kT_target
        info = {}
        for Z, sym in [(1, "H"), (8, "O")]:
            mask = (numbers == Z)
            if mask.any():
                ke_el = float(ke_atom[mask].mean())
                info[f"ke_per_{sym}_eV"] = ke_el
                info[f"ke_per_{sym}_ratio"] = (
                    ke_el / classical_per_atom if classical_per_atom > 0 else 0.0)
        adq = getattr(self.dyn, "adq", None)
        if adq is not None:
            info["adqtb_fdt_residual"] = float(adq.last_residual_norm)
            info["adqtb_gain_min"] = float(adq.gain.min().item())
            info["adqtb_gain_max"] = float(adq.gain.max().item())
        return info

    def _save_frame(self, step: int = -1, stage: str = "md"):
        """Write one trajectory frame.

        Args:
            step: Current simulation step number, stored per-frame for
                  accurate resume offset.
        """
        pos, vel, force, epot, ekin, fmax = self._collect_step_data()
        # Cache metrics to avoid recomputing them in the subsequent _log_step call
        self._metrics_cache = {"epot": epot, "ekin": ekin, "fmax": fmax, "etot": epot + ekin}

        cell = self.atoms.get_cell()
        # Guard: the per-frame cell must be finite with a positive volume.
        # Catches silent box corruption (e.g. a stale/zeroed cell slipping
        # through after a barostat move) before it is committed to disk.
        if self.full_pbc.any():
            _vol = float(torch.det(cell).abs().item())
            assert torch.isfinite(cell).all() and _vol > 0.0, (
                f"[barostat] frame {step}: invalid cell before write "
                f"(volume={_vol}, diag={torch.diagonal(cell).tolist()})."
            )
        if self.wrap and self.full_pbc.any():
            pbc = self.full_pbc
            if isinstance(pbc, torch.Tensor):
                pbc = pbc.tolist()
            pos = TorchAtoms.wrap_positions_tensor(
                pos, cell, pbc=pbc, center=(0.5, 0.5, 0.5),
            )
        # QTB diagnostics as per-frame H5 info (empty dict for Langevin/relax ->
        # no info group is created, so the file stays byte-compatible with old runs).
        _temp = (2.0 * ekin) / (3.0 * self.full_natoms * units.kB) if self.full_natoms else 0.0
        qtb_info = self._qtb_info_dict(_temp) if stage == "md" else {}
        self.trajectory.write(
            positions=pos, velocities=vel, cell=cell,
            energy=epot, forces=force, step=step,
            info=(qtb_info or None),
        )

        if self.atoms.pbc.any():
            self.atoms.wrap()

    # ------------------------------------------------------------------
    # NPT: energy-only Monte-Carlo barostat
    # ------------------------------------------------------------------
    def _build_barostat(self, temperature_K: float) -> None:
        """Construct the MC barostat and its molecule map once, before MD."""
        pos_np = self.atoms.get_positions().detach().cpu().numpy()
        numbers_np = self.atoms.get_atomic_numbers().detach().cpu().numpy()

        cell_t = self.atoms.get_cell()
        box_diag = cell_t.detach().cpu().numpy().diagonal().copy()

        mol_index_np = build_molecule_index(
            pos_np, numbers_np, box_diag, source=self.molecule_source,
        )
        n_mol = int(mol_index_np.max()) + 1
        mol_index = torch.from_numpy(mol_index_np).to(self.device, torch.long)
        mol_first = molecule_first_atom(mol_index, n_mol)
        masses = masses_from_numbers(numbers_np, self.device, torch.float64)
        total_mass = float(masses.sum().item())
        logger.info(
            f"[barostat] molecule map ({self.molecule_source}): "
            f"{n_mol} molecules over {len(mol_index_np)} atoms; "
            f"total mass {total_mass:.1f} amu."
        )
        self._total_mass_amu = total_mass

        self.barostat = TorchMCBarostat(
            pressure_bar=self.pressure_bar,
            temperature_K=temperature_K,
            n_mol=n_mol,
            mol_index=mol_index,
            mol_first_atom=mol_first,
            masses=masses,
            dlnV=self.barostat_dlnV,
            seed=self.barostat_seed,
            device=self.device,
            dtype=torch.float64,
        )
        V0 = float(torch.det(cell_t).abs().item())
        logger.info(
            f"[barostat] NPT enabled: P={self.pressure_bar} bar, T={temperature_K} K, "
            f"interval={self.barostat_interval} steps, dlnV={self.barostat_dlnV}; "
            f"V0={V0:.1f} A^3, rho0={self._total_mass_amu * 1.66053907 / V0:.4f} g/cm^3."
        )

    def _mc_barostat_attempt(self) -> bool:
        """One isotropic MC volume move. Returns whether the move was accepted."""
        baro = self.barostat
        cell_old = self.atoms.get_cell().clone()
        V_old = float(torch.det(cell_old).abs().item())

        delta, u = baro.propose()
        s = math.exp(delta / 3.0)
        V_new = V_old * math.exp(delta)

        E_old = float(self.atoms.get_potential_energy())
        pos_old = self.atoms.get_positions().clone()
        new_pos = baro.scale_positions(pos_old, s, cell_old)
        self.atoms.set_cell(cell_old * s, scale_atoms=False)
        self.atoms.set_positions(new_pos)
        E_new = float(self.atoms.get_potential_energy())
        accepted = baro.accept(E_new - E_old, V_old, V_new, u)
        if not accepted:
            self.atoms.set_cell(cell_old, scale_atoms=False)
            self.atoms.set_positions(pos_old)

        baro.record(accepted)
        return accepted

    def close(self):
        """Release runner state."""
        self._close_trajectory()

    def _log_step(self, step: int, interval_time: float, total_elapsed: float, task_name: str):
        """Log system thermodynamics. Uses cached metrics when available."""
        if self._metrics_cache:
            m = self._metrics_cache
            epot, ekin, fmax, etot = m["epot"], m["ekin"], m["fmax"], m["etot"]
        else:
            # Fallback: recompute if we skipped the _save_frame step
            epot, ekin = float(self.atoms.get_potential_energy()), float(self.atoms.get_kinetic_energy())
            force = self.atoms.get_forces()
            fmax = float(torch.norm(force, dim=1).max().item()) if len(self.atoms) > 0 else 0.0
            etot = epot + ekin

        temp = (2.0 * ekin) / (3.0 * self.full_natoms * units.kB)
        step_time = (interval_time / self.log_interval) * 1000 if step > 0 else 0
        log_data = {
            f"{task_name.upper()}/step": step, f"{task_name.upper()}/potential_energy": epot,
            f"{task_name.upper()}/total_energy": etot, f"{task_name.upper()}/temperature": temp,
            f"{task_name.upper()}/fmax": fmax, f"{task_name.upper()}/step_time_ms": step_time,
        }
        baro_suffix = ""
        if self.barostat is not None:
            V = float(torch.det(self.atoms.get_cell()).abs().item())
            density = self._total_mass_amu * 1.66053907 / V if V > 0 else 0.0
            acc = self.barostat.accept_ratio
            log_data[f"{task_name.upper()}/volume_A3"] = V
            log_data[f"{task_name.upper()}/density_g_cm3"] = density
            log_data[f"{task_name.upper()}/baro_accept_ratio"] = acc
            baro_suffix = f" | V: {V:9.1f} A^3 | rho: {density:6.4f} g/cm^3 | acc: {acc:5.2f}"

        # QTB NQE diagnostics. Per-element kinetic energy is the primary NQE
        # signal: light atoms (H) gain large zero-point energy, heavy atoms (O)
        # stay near classical 3/2 kT. NOTE: the equipartition "temperature"
        # above is ARTIFICIALLY HIGH under quantum QTB (the excess is zero-point, not
        # thermal) — judge NQE by these per-element KEs vs the classical TARGET, not T.
        qtb_suffix = ""
        qtb_info = self._qtb_info_dict(temp)
        for k, v in qtb_info.items():
            log_data[f"{task_name.upper()}/{k}"] = v
        if "ke_per_H_eV" in qtb_info:
            qtb_suffix = (f" | keH: {qtb_info['ke_per_H_eV']:.4f} eV "
                          f"({qtb_info.get('ke_per_H_ratio', 0.0):.2f}x cl)")
        if self.use_swanlab: swanlab.log(log_data)
        logger.info(f"[{task_name}] Step {step:6d} | Etot: {etot:12.4f} | Fmax: {fmax:8.4f} | T: {temp:7.2f} K | Step: {step_time:6.2f} ms{baro_suffix}{qtb_suffix}")

    def relax(self, fmax: float = 0.05, steps: int = 500, do_resume: bool = True,
              maxstep: float = 0.2, damping: float = 1.0):
        """Structure optimization using GPU-native L-BFGS.

        Note: ``full_sync_interval`` only controls how often we write frames
        and log metrics.

        Args:
            fmax: Force convergence threshold (eV/Å).
            steps: Maximum number of optimization steps.
            do_resume: Whether to resume from an existing trajectory.
            maxstep: Maximum atomic displacement per L-BFGS step (Å).
            damping: Step-size scaling factor (< 1.0 for conservative steps).
        """
        try:
            if do_resume and self.resume_step is not None and os.path.exists(self.relax_path):
                self._truncate_trajectory_to_step(self.relax_path)
            self._setup(resume_path=self.relax_path if do_resume else None)
            if self.step_offset >= steps:
                return

            opt = TorchLBFGS(self.atoms, fmax=fmax,
                             maxstep=maxstep, damping=damping)
            self._open_trajectory(self.relax_path, fresh=(self.step_offset == 0))

            if self.step_offset == 0:
                self._save_frame(step=0, stage="relax")
                self._log_step(0, 0.0, 0.0, "RELAX")

            start_time = interval_start = time.time()

            step = 0
            for _ in range(steps - self.step_offset):
                if self._has_atoms:
                    opt.step()

                self._metrics_cache = None  # Bust cache for next step
                step += 1
                total_step = self.step_offset + step
                if self._should_save_step(total_step) or self._stop_requested:
                    self._save_frame(step=total_step, stage="relax")
                if self._stop_requested:
                    break
                if self._should_log_step(total_step):
                    now = time.time()
                    self._log_step(total_step, now - interval_start, now - start_time, "RELAX")
                    interval_start = now

            if (not self._stop_requested
                    and step > 0
                    and step % self.full_sync_interval != 0):
                self._save_frame(step=self.step_offset + step, stage="relax")
        finally:
            self._close_trajectory()

    def run_md(self, steps: int = 1000, timestep_fs: float = None, temperature_K: float = 300.0,
               friction: Optional[float] = None, do_resume: bool = True):
        """GPU-native Langevin / QTB MD.

        When *do_resume* is True, the atom source is resolved as follows:
          1. If a previous MD trajectory exists, resume from it.
          2. Otherwise, if a RELAX trajectory exists, start MD from the
             relax phase's last frame (with fresh velocities).
          3. Otherwise, start from the current in-memory structure or
             the initial input.
        """
        try:
            resume_path, source_tag = self._resolve_md_source(do_resume=do_resume)

            # _setup handles step_offset reset for relax→MD transition.
            self._setup(resume_path=resume_path,
                        reset_step_offset=(source_tag == "relax_transition"))

            if self.ensemble == "nve":
                # NVE = deterministic velocity-Verlet. Force friction to 0 (the
                # TorchLangevin coefficients then reduce exactly to velocity-Verlet)
                # and forbid QTB, which is itself a thermostat and contradicts NVE.
                if self.thermostat == "qtb":
                    raise RuntimeError(
                        "ensemble='nve' is incompatible with thermostat='qtb': QTB is a "
                        "thermostat (colored-noise heat bath). Use thermostat='langevin' "
                        "for NVE, or ensemble='nvt'/'npt' for QTB.")
                if friction not in (None, 0, 0.0):
                    logger.warning(
                        f"[nve] ignoring friction={friction}; NVE forces friction=0 "
                        f"(deterministic velocity-Verlet).")
                friction = 0.0
            elif friction is None:
                friction = 0.001

            self._initialize_md_velocities_if_needed(temperature_K=temperature_K)

            # Integrator Setup
            rng = torch.Generator(device=self.device)
            rng.manual_seed(self.seed)

            if self.thermostat == "qtb":
                from molfm.interface.ase.torch_ext.torch_qtb import TorchQTB
                dyn = TorchQTB(
                    self.atoms,
                    timestep=timestep_fs * units.fs,
                    temperature_K=temperature_K,
                    friction=friction / units.fs,
                    fixcm=self.fixcm,
                    rng=rng,
                    n_seg=self.qtb_n_seg,
                    adaptive=self.qtb_adaptive,
                    adqtb_lr=self.qtb_adqtb_lr,
                    classical=self.qtb_classical,
                ) if self._has_atoms else None
            else:
                dyn = TorchLangevin(
                    self.atoms,
                    timestep=timestep_fs * units.fs,
                    temperature_K=temperature_K,
                    friction=friction / units.fs,
                    fixcm=self.fixcm,
                    rng=rng,
                ) if self._has_atoms else None

            self.dyn = dyn

            # NPT: build the MC barostat (+ molecule map) once the configuration
            # is available.
            if self.ensemble == "npt":
                self._build_barostat(temperature_K=temperature_K)

            self._open_trajectory(self.md_path, fresh=(self.step_offset == 0))
            if self.step_offset == 0:
                self._save_frame(step=0)
                self._log_step(0, 0.0, 0.0, "MD")
            t_start = interval_start = time.time()

            # Per-step timing accumulators.
            _t_step = _t_write = 0.0
            _n_step = _n_write = 0
            for i in range(1, (steps - self.step_offset) + 1):
                t0 = time.time()
                if dyn:
                    dyn.step()
                else:
                    self.atoms.get_forces()
                _t_step += time.time() - t0
                _n_step += 1

                self._metrics_cache = None
                total_step = self.step_offset + i

                # NPT: attempt an isotropic MC volume move. Placed before
                # save/log so the recorded frame reflects the post-move box.
                if self.barostat is not None and (total_step % self.barostat_interval == 0):
                    self._mc_barostat_attempt()
                if self._should_save_step(total_step) or self._stop_requested:
                    t0 = time.time()
                    self._save_frame(step=total_step)
                    _t_write += time.time() - t0
                    _n_write += 1
                if self._stop_requested:
                    break
                if self._should_log_step(total_step):
                    now = time.time()
                    self._log_step(total_step, now - interval_start, now - t_start, "MD")
                    interval_start = now

            if (not self._stop_requested
                    and (steps - self.step_offset) > 0
                    and (steps - self.step_offset) % self.full_sync_interval != 0):
                self._save_frame(step=steps)

            # Performance summary
            if _n_step > 0:
                avg_step = _t_step / _n_step * 1000
                avg_write = _t_write / _n_write * 1000 if _n_write > 0 else 0.0
                logger.info(
                    f"[MD] Perf: avg dyn.step={avg_step:.1f} ms "
                    f"(n={_n_step}), avg write_frame={avg_write:.1f} ms "
                    f"(n={_n_write})"
                )

            self._close_trajectory("MD task complete.")
        finally:
            if self.trajectory is not None:
                self._close_trajectory()
