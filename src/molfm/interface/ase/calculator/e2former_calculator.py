# -*- coding: utf-8 -*-
from typing import Dict, List
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, CalculatorSetupError, InputError, all_changes

from molfm.data.collator import collate_fn 
from molfm.interface.model_interface import E2FormerModelInterface

def ase_atom_to_dict(atoms: Atoms, idx: str = "-1", name: str = None)-> Dict:
    name = name if name is not None else atoms.get_chemical_formula()
    atom_dict = {
        'idx' : idx,
        'atoms': atoms.get_atomic_numbers(),
        'coords': atoms.get_positions(),
        'charge':sum(atoms.get_initial_charges()),
        'multiplicity': 1,
        'smiles_isomeric': atoms.symbols,
        'name': name,
        'cell': atoms.cell.array,
        'pbc': atoms.pbc,
        }
    return atom_dict

def dict_to_model_input(data_list: List[dict], data_name: str, device: str) -> list:
    out_list = []
    for data in data_list:
        pos = data['coords']  # Angstrom
        atomic_numbers = data['atoms']
        num_atoms = len(atomic_numbers)
        charge = data['charge']
        multiplicity = data['multiplicity']

        cell = torch.zeros((3, 3), dtype=torch.float32) if data.get("cell", None) is None else torch.tensor(data['cell'], dtype=torch.float32)
        pbc = torch.tensor([0, 0, 0], dtype=torch.int32) if data.get("pbc", None) is None else torch.tensor(data['pbc'], dtype=torch.int32)

        mol_data = {
            'data_name': [data_name],  # self.head_name is the dataset name
            "multiplicity": torch.tensor(multiplicity, dtype=torch.int32).reshape(-1),
            "pbc": pbc,
            "cell": cell,
            "atomic_numbers": torch.tensor(atomic_numbers, dtype=torch.int32).reshape(-1),
            "charge": torch.tensor(charge, dtype=torch.int32).reshape(-1),
            "pos": torch.tensor(pos, dtype=torch.float32).reshape(-1, 3),
            "num_atoms": torch.tensor(num_atoms, dtype=torch.int32).reshape(-1),
            "atom_masks": torch.tensor([[True] * num_atoms]),
            "idx": 0,
        }

        out_list.append(mol_data)
    
    out_batch = collate_fn(out_list)
    for k, v in out_batch.items():
        if isinstance(v, torch.Tensor):
            if v.device != device:
                # print(k)
                out_batch[k] = v.to(device)

    out_batch['data_name'] = [data_name]
    # out_batch['pos'].requires_grad_(True)

    return out_batch

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
        force_use_aperiodic: bool = False,
        add_ref_energy: bool = True,
        auto_setup: bool = True,
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
            auto_setup=auto_setup,
        )

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
        forces = model_output["pred_forces"].detach().cpu().numpy()

        for idx, atoms in enumerate(atoms_list):
            energy = float(energies[idx])
            if self.add_ref_energy and self.ref_energy is not None:
                energy += float(self.ref_energy[atoms.get_atomic_numbers()].sum())
            results_list.append(
                {
                    "energy": energy,
                    "forces": forces[idx][: len(atoms)],
                }
            )

        return results_list

    def calculate(self, atoms=None, properties=None, system_changes=all_changes, symmetry="c1"):
        if properties is None:
            properties = ["energy"]
        for prop in properties:
            if prop not in self.implemented_properties:
                raise InputError(f"Property {prop} is not implemented")

        super().calculate(atoms=atoms)
        if self.atoms is None:
            raise CalculatorSetupError("An Atoms object must be provided")

        atoms = self.atoms.copy()
        if self.atoms.pbc.any():
            atoms.wrap()

        model_output = self.predict(self._prepare_model_data([atoms]))
        self.results = self._prepare_output(model_output, [atoms])[0]

    def calculate_batch(self, atoms_list=None):
        wrapped_atoms_list = self._hard_wrap_atoms_list(atoms_list)
        model_output = self.predict(self._prepare_model_data(wrapped_atoms_list))
        return self._prepare_output(model_output, wrapped_atoms_list)

    def read(self, label):
        pass
