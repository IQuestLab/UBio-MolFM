# -*- coding: utf-8 -*-
# ASE Calculator for E2Former, with two input backends:
#   1. Torch backend  — zero-copy GPU-native path for TorchAtoms, used by the
#      GPU-native MD/relaxation stack in molfm/interface/ase/torch_ext/.
#   2. ASE backend    — compatibility path for standard ase.Atoms via numpy.
#
# Both backends share one model: `calculate_batch` dispatches on the flavour of the
# Atoms it is given, and the TorchAtoms path returns raw device tensors so the
# integrators never leave the GPU. The ASE path wraps into the primary cell first
# (`_hard_wrap_atoms_list`) and returns numpy arrays.
#
# `ase_atom_to_dict` / `dict_to_model_input` live in molfm.interface.data_utils and
# are re-exported below for backwards compatibility.

from typing import Any, Dict, List

import numpy as np
import torch
from ase.calculators.calculator import Calculator, CalculatorSetupError, InputError, all_changes
from loguru import logger

from molfm.data.collator import collate_fn
from molfm.interface.ase.torch_ext.torch_atoms import TorchAtoms
from molfm.interface.data_utils import ase_atom_to_dict, dict_to_model_input
from molfm.interface.model_interface import E2FormerModelInterface, maybe_gc_cuda

# Re-exported so that `from ...e2former_calculator import ase_atom_to_dict` keeps
# working now that these helpers live in molfm.interface.data_utils.
__all__ = ["E2FormerCalculator", "ase_atom_to_dict", "dict_to_model_input"]


class E2FormerCalculator(Calculator, E2FormerModelInterface):
    implemented_properties = ["energy", "forces"]
    discard_results_on_any_change = True
    default_parameters = {}

    def __init__(
        self,
        restart=None,
        ignore_bad_restart=False,
        label="e2former-calc",
        atoms=None,
        command=None,
        config_path: str = None,
        config_name: str = None,
        checkpoint_path: str = None,
        head_name: str = None,
        device: str = "cuda:0",
        use_compile: bool = False,
        use_tf32: bool = False,
        use_faiss: bool = True,
        recompute_budget: float = -1.0,
        force_use_aperiodic: bool = False,
        add_ref_energy: bool = True,
        auto_setup: bool = True,
        charge: int = 0,
        multiplicity: int = 1,
        **kwargs,
    ):
        super().__init__(
            restart=restart,
            ignore_bad_restart=ignore_bad_restart,
            label=label,
            atoms=atoms,
            command=command,
            **kwargs,
        )

        self.force_use_aperiodic = force_use_aperiodic
        self.add_ref_energy = add_ref_energy
        self.charge = charge
        self.multiplicity = multiplicity
        E2FormerModelInterface.__init__(
            self,
            config_path=config_path,
            config_name=config_name,
            checkpoint_path=checkpoint_path,
            head_name=head_name,
            device=device,
            use_compile=use_compile,
            use_tf32=use_tf32,
            use_faiss=use_faiss,
            recompute_budget=recompute_budget,
            auto_setup=auto_setup,
        )

        # Keep the atomic reference energies on the model device so the TorchAtoms
        # path can add them without a GPU->CPU round-trip per step.
        if self.ref_energy is not None:
            self.ref_energy = torch.as_tensor(self.ref_energy, device=self.device)

    # ==========================================
    # --- Main dispatcher ---
    # ==========================================

    def calculate(self, atoms=None, properties=None, system_changes=all_changes, symmetry="c1"):
        if properties is None:
            properties = ["energy"]
        for prop in properties:
            if prop not in self.implemented_properties:
                raise InputError(f"Property {prop} is not implemented")

        super().calculate(atoms=atoms)
        if self.atoms is None:
            raise CalculatorSetupError("An Atoms object must be provided")

        self.results = self.calculate_batch([self.atoms], properties=properties)[0]

    def calculate_batch(self, atoms_list: List[Any] = None, properties: List[str] = None) -> List[Dict[str, Any]]:
        """Batched evaluation, dispatching on the input Atoms flavour.

        TorchAtoms inputs stay on the GPU end-to-end and get torch tensors back;
        standard ase.Atoms go through the numpy path and get numpy arrays back.
        """
        if not atoms_list:
            return []
        if properties is None:
            properties = ["energy", "forces"]
        need_force = "forces" in properties

        if isinstance(atoms_list[0], TorchAtoms):
            model_input = self._prepare_torch_input(atoms_list)
            model_output = self.predict(model_input, need_force=need_force)
            maybe_gc_cuda(self.device)
            return self._process_torch_output(model_output, atoms_list)

        # Standard ASE path: wrap into the primary cell first (see module header).
        wrapped_atoms_list = self._hard_wrap_atoms_list(atoms_list)
        model_output = self.predict(self._prepare_model_data(wrapped_atoms_list),
                                    need_force=need_force)
        maybe_gc_cuda(self.device)
        return self._prepare_output(model_output, wrapped_atoms_list)

    def check_state(self, atoms, tol=1e-15):
        """Compare *atoms* against the cached state, without leaving the GPU."""
        if isinstance(atoms, TorchAtoms):
            return self._check_state_torch(atoms, tol=tol)
        return super().check_state(atoms, tol=tol)

    # ==========================================
    # --- TorchAtoms backend (GPU fast path) ---
    # ==========================================

    def _prepare_torch_input(self, atoms_list: List[TorchAtoms]) -> Dict[str, Any]:
        """Build the model input straight from the TorchAtoms device tensors."""
        device = self.device

        out_list = []
        for atoms in atoms_list:
            if self.force_use_aperiodic:
                cell = torch.zeros((3, 3), device=device, dtype=torch.float32)
                pbc = torch.zeros(3, device=device, dtype=torch.int32)
            else:
                cell = atoms.get_cell().to(device, dtype=torch.float32)
                pbc = atoms.get_pbc().to(device, dtype=torch.int32)

            num_atoms = len(atoms)
            mol_data = {
                'data_name': [self.head_name],
                "multiplicity": torch.tensor(self.multiplicity, dtype=torch.int32, device=device).reshape(-1),
                "pbc": pbc,
                "cell": cell,
                "atomic_numbers": atoms._torch_numbers.to(device, dtype=torch.int32).reshape(-1),
                "charge": torch.tensor(self.charge, dtype=torch.int32, device=device).reshape(-1),
                "pos": atoms._torch_positions.to(device, dtype=torch.float32).reshape(-1, 3),
                "num_atoms": torch.tensor(num_atoms, dtype=torch.int32, device=device).reshape(-1),
                "atom_masks": torch.tensor([[True] * num_atoms], device=device),
                "idx": 0,
            }
            out_list.append(mol_data)

        out_batch = collate_fn(out_list)
        for k, v in out_batch.items():
            if isinstance(v, torch.Tensor) and v.device != device:
                out_batch[k] = v.to(device)

        out_batch['data_name'] = [self.head_name]
        return out_batch

    def _process_torch_output(self, model_output: Dict[str, torch.Tensor],
                              atoms_list: List[TorchAtoms]) -> List[Dict[str, torch.Tensor]]:
        """Return raw GPU tensors for the downstream torch integrators."""
        results_list = []
        energies = model_output["pred_energy"].detach()
        forces = model_output.get("pred_forces", None)
        if forces is not None:
            forces = forces.detach()

        for idx, atoms in enumerate(atoms_list):
            energy = energies[idx]
            if self.add_ref_energy and self.ref_energy is not None:
                energy = energy + self.ref_energy[atoms._torch_numbers.to(torch.long)].sum()

            res = {"energy": energy}
            if forces is not None:
                res["forces"] = forces[idx][: len(atoms)]
            results_list.append(res)
        return results_list

    def _check_state_torch(self, atoms: TorchAtoms, tol=1e-13) -> List[str]:
        """Torch-native state comparison (no numpy round-trip)."""
        cached_atoms = getattr(self, "atoms", None)
        if cached_atoms is None or not isinstance(cached_atoms, TorchAtoms):
            return all_changes[:]

        system_changes = []
        if not torch.equal(cached_atoms._torch_numbers, atoms._torch_numbers):
            system_changes.append("numbers")
        if not torch.allclose(cached_atoms._torch_positions, atoms._torch_positions, atol=tol, rtol=0.0):
            system_changes.append("positions")
        if not torch.allclose(cached_atoms._torch_cell, atoms._torch_cell, atol=tol, rtol=0.0):
            system_changes.append("cell")
        if not bool((cached_atoms.pbc == atoms.pbc).all()):
            system_changes.append("pbc")
        return system_changes

    # ==========================================
    # --- Standard ASE backend (numpy path) ---
    # ==========================================

    def _prepare_model_data(self, atoms_list: list, force_use_aperiodic: bool = False):
        data_list = []
        for atoms in atoms_list:
            data_dict = ase_atom_to_dict(atoms)
            if force_use_aperiodic or self.force_use_aperiodic:
                data_dict["cell"] = torch.zeros((3, 3), dtype=torch.float32)
                data_dict["pbc"] = torch.tensor([0, 0, 0], dtype=torch.int32)
            data_list.append(data_dict)

        return dict_to_model_input(
            data_list,
            self.head_name,
            self.device,
        )

    def _hard_wrap_atoms_list(self, atoms_list: list) -> list:
        """Wrap atoms into the primary cell without trying to reconstruct molecules."""
        wrapped_atoms_list = []
        for atoms in atoms_list:
            wrapped_atoms = atoms.copy()
            if wrapped_atoms.pbc.any():
                wrapped_atoms.wrap()
            wrapped_atoms_list.append(wrapped_atoms)
        return wrapped_atoms_list

    def _prepare_output(self, model_output: dict, atoms_list: list) -> list[dict]:
        results_list = []
        energies = model_output["pred_energy"].detach().cpu().numpy()
        forces = model_output.get("pred_forces", None)
        if forces is not None:
            forces = forces.detach().cpu().numpy()

        for idx, atoms in enumerate(atoms_list):
            energy = float(energies[idx])
            if self.add_ref_energy and self.ref_energy is not None:
                # ref_energy lives on the model device; index there, then read back once.
                numbers = torch.as_tensor(
                    atoms.get_atomic_numbers(), dtype=torch.long, device=self.device
                )
                energy += float(self.ref_energy[numbers].sum().item())
            res = {"energy": energy}
            if forces is not None:
                res["forces"] = forces[idx][: len(atoms)]
            results_list.append(res)

        return results_list

    def read(self, label):
        pass
