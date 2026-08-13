# -*- coding: utf-8 -*-
# Data-conversion helpers shared by the ASE calculator and the GPU-native
# ASERunner stack.

import os
from typing import Dict, List

import numpy as np
import torch
from ase import Atoms
from loguru import logger

from molfm.data.collator import collate_fn

# --------------------------------------------------------------------------
# Model input construction
# --------------------------------------------------------------------------


def ase_atom_to_dict(atoms: Atoms, idx: str = "-1", name: str = None) -> Dict:
    name = name if name is not None else atoms.get_chemical_formula()
    atom_dict = {
        'idx': idx,
        'atoms': atoms.get_atomic_numbers(),
        'coords': atoms.get_positions(),
        'charge': np.sum(atoms.get_initial_charges()),
        'multiplicity': 1,
        'smiles_isomeric': atoms.symbols,
        'name': name,
        'cell': atoms.cell.array,
        'pbc': atoms.pbc,
        }
    return atom_dict


def dict_to_model_input(
    data_list: List[dict],
    data_name: str,
    device: str,
    pos_dtype: torch.dtype = torch.float32,
) -> list:
    """Collate raw per-structure dicts into a batched model input on *device*.

    ``pos_dtype`` exists for the GPU-native MD path, which keeps positions in
    float64 so that the integrator does not lose precision over long
    trajectories; the default float32 matches the training pipeline.
    """
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
            "pos": torch.tensor(pos, dtype=pos_dtype).reshape(-1, 3),
            "num_atoms": torch.tensor(num_atoms, dtype=torch.int32).reshape(-1),
            "atom_masks": torch.tensor([[True] * num_atoms]),
            "idx": 0,
        }

        out_list.append(mol_data)

    out_batch = collate_fn(out_list)
    for k, v in out_batch.items():
        if isinstance(v, torch.Tensor):
            if v.device != device:
                out_batch[k] = v.to(device)

    out_batch['data_name'] = [data_name]

    return out_batch


# --------------------------------------------------------------------------
# Structure file readers (.pdb / .gro via MDAnalysis)
# --------------------------------------------------------------------------

GRO_ION_NAME_MAP = {
    "LI": "Li",
    "NA": "Na",
    "K": "K",
    "RB": "Rb",
    "CS": "Cs",
    "MG": "Mg",
    "CA": "Ca",
    "SR": "Sr",
    "BA": "Ba",
    "ZN": "Zn",
    "MN": "Mn",
    "FE": "Fe",
    "CO": "Co",
    "NI": "Ni",
    "CU": "Cu",
    "CD": "Cd",
    "HG": "Hg",
    "AL": "Al",
    "CL": "Cl",
    "BR": "Br",
    "F": "F",
    "I": "I",
}

GRO_IONIC_RESIDUES = {
    "LI", "NA", "K", "RB", "CS", "MG", "CA", "CAL", "CA2", "CAP", "SR", "BA",
    "ZN", "MN", "FE", "CO", "NI", "CU", "CD", "HG", "AL", "CL", "BR", "F", "I",
}


def _normalize_gro_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().upper() if ch.isalnum())


def _guess_gro_element(atom_name: str, residue_name: str) -> str:
    atom_key = _normalize_gro_name(atom_name)
    residue_key = _normalize_gro_name(residue_name)

    if atom_key in GRO_ION_NAME_MAP and residue_key in GRO_IONIC_RESIDUES:
        return GRO_ION_NAME_MAP[atom_key]

    if not atom_key:
        return "X"

    if atom_key[0].isdigit():
        atom_key = atom_key[1:]

    if not atom_key:
        return "X"

    if len(atom_key) >= 2 and atom_key[:2] in {"CL", "BR"}:
        return atom_key[:2].title()

    return atom_key[0].title()


def mda_to_ase_atoms(file_path: str, use_box: bool = True) -> Atoms:
    """
    Reads a molecular structure file using MDAnalysis and converts it to an ASE Atoms object.

    Parameters:
    file_path (str): The full path to the structure file (e.g., .pdb, .gro, .tpr, .xyz).
                      Any format supported by MDAnalysis is acceptable.
    use_box (bool): Whether to attempt to extract the unit cell information (box) from
                    the MDA Universe and apply it to the ASE Atoms object.
                    If the file lacks box information, or if you only need molecular
                    coordinates, set this to False. Defaults to True.

    Returns:
    ase.Atoms: The converted ASE Atoms object.

    Raises:
    FileNotFoundError: If the specified file does not exist.
    """
    import MDAnalysis as mda

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Create the MDAnalysis Universe
    try:
        # The Universe automatically loads topology and coordinates from the file
        universe = mda.Universe(file_path)
        # Select all atoms in the system
        atoms_group = universe.select_atoms("all")
    except Exception as e:
        logger.error(f"Error loading file with MDAnalysis: {e}")
        raise

    # 2. Extract core data
    positions = atoms_group.positions  # Get atomic coordinates (in Angstroms)

    # 2.1 Get element symbols, guess if missing
    try:
        symbols = atoms_group.elements
        if any(s == '' or s is None for s in symbols):
            logger.warning(f"Some elements are missing in {file_path}, guessing from atom names.")
            from MDAnalysis.topology.guessers import guess_atom_element
            symbols = [guess_atom_element(name) if (s == '' or s is None) else s for s, name in zip(symbols, atoms_group.names)]
    except Exception as e:
        logger.warning(f"Failed to extract elements from {file_path} ({e}), guessing from atom names.")
        from MDAnalysis.topology.guessers import guess_atom_element
        symbols = [guess_atom_element(name) for name in atoms_group.names]

    # 3. Create the ASE Atoms object
    symbols = [s.capitalize() for s in symbols]
    ase_atoms = Atoms(
        symbols=symbols,
        positions=positions
    )

    # 4. Handle Unit Cell / Periodic Boundary Conditions (Optional)
    if use_box and universe.dimensions is not None:
        # universe.dimensions returns [a, b, c, alpha, beta, gamma]
        box_data = universe.dimensions

        # Check if cell information is valid (a, b, c must be positive, angles also usually > 0)
        if np.all(box_data[:3] > 0) and np.all(box_data[3:] > 0):
            # Set the cell and pbc (Periodic Boundary Conditions) for ASE
            ase_atoms.set_cell(box_data)

            # Set pbc to True by default, assuming a periodic system if cell info is present
            ase_atoms.set_pbc([True, True, True])
        else:
            logger.warning(f"Box dimensions found in {file_path} are non-positive or invalid: {box_data}. Skipping setting cell/pbc.")

    return ase_atoms


def gro_to_ase_atoms(gro_file_path: str, use_box: bool = True) -> Atoms:
    """Read a .gro file via MDAnalysis and infer elements conservatively."""
    import MDAnalysis as mda

    if not isinstance(gro_file_path, str) or not gro_file_path.endswith('.gro'):
        raise ValueError("Input must be a path to a .gro file.")

    if not os.path.exists(gro_file_path):
        raise FileNotFoundError(f"File not found: {gro_file_path}")

    try:
        universe = mda.Universe(gro_file_path)
        atoms_group = universe.select_atoms("all")
    except Exception as e:
        logger.error(f"Error loading GRO file with MDAnalysis: {e}")
        raise

    positions = atoms_group.positions
    symbols = [_guess_gro_element(atom.name, atom.resname) for atom in atoms_group]

    ase_atoms = Atoms(
        symbols=symbols,
        positions=positions,
    )

    if use_box and universe.dimensions is not None:
        box_data = universe.dimensions
        if np.all(box_data[:3] > 0) and np.all(box_data[3:] > 0):
            ase_atoms.set_cell(box_data)
            ase_atoms.set_pbc([True, True, True])
        else:
            logger.warning(f"Box dimensions found in {gro_file_path} are non-positive or invalid: {box_data}. Skipping setting cell/pbc.")

    return ase_atoms
