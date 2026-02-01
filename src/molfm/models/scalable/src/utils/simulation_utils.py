# -*- coding: utf-8 -*-
import gsd.hoomd
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList, natural_cutoffs


def ase_traj_to_gsd(traj, save_path, size_scale=1.2):
    # use first frame to get atom types and bonds
    atoms = traj[0]

    # atom types
    atom_types = atoms.get_atomic_numbers()
    unique_atom_types = np.unique(atom_types)
    atom_type_map = {unique_atom_types[i]: i for i in range(len(unique_atom_types))}
    atom_type_ids = [atom_type_map[atom_types[i]] for i in range(len(atom_types))]
    unique_atom_types = [str(u) for u in unique_atom_types]

    # bonds
    NL = NeighborList(natural_cutoffs(atoms), self_interaction=False)
    NL.update(atoms)
    bonds = np.array(NL.get_connectivity_matrix().todense().nonzero())
    sorted_bond_types = np.array(
        [
            tuple(sorted([atom_types[bonds[0, i]], atom_types[bonds[1, i]]]))
            for i in range(bonds.shape[-1])
        ]
    )
    unique_bond_types = np.unique(sorted_bond_types, axis=0)
    bond_type_map = {tuple(u): i for i, u in enumerate(unique_bond_types)}
    bondtype_ids = [
        bond_type_map[tuple(sorted_bond_types[i])] for i in range(bonds.shape[-1])
    ]
    bond_groups = [(bonds[0, i], bonds[1, i]) for i in range(bonds.shape[-1])]
    unique_bond_types = [f"{u[0]}-{u[1]}" for u in unique_bond_types]

    with gsd.hoomd.open(save_path, "w") as hoomd_traj:
        for step, atoms in enumerate(traj):
            pos = atoms.get_positions()
            gsd_frame = gsd.hoomd.Frame()
            gsd_frame.configuration.step = step

            gsd_frame.particles.N = pos.shape[0]
            gsd_frame.particles.position = pos
            gsd_frame.particles.types = unique_atom_types
            gsd_frame.particles.typeid = atom_type_ids
            gsd_frame.particles.diameter = [
                covalent_radii[atom_types[i]] * size_scale
                for i in range(len(atom_types))
            ]

            gsd_frame.bonds.N = len(bond_groups)
            gsd_frame.bonds.types = unique_bond_types
            gsd_frame.bonds.typeid = bondtype_ids
            gsd_frame.bonds.group = bond_groups

            cell_length = np.array(
                [atoms.cell[0, 0], atoms.cell[1, 1], atoms.cell[2, 2]]
            )
            cell_angle = np.empty_like(cell_length)
            cell_angle.fill(90)
            gsd_frame.configuration.box = lengths_and_angles_to_tilt_factors(
                cell_length[0],
                cell_length[1],
                cell_length[2],
                cell_angle[0],
                cell_angle[1],
                cell_angle[2],
            )
            hoomd_traj.append(gsd_frame)


def lengths_and_angles_to_tilt_factors(
    a_length,
    b_length,
    c_length,
    alpha,
    beta,
    gamma,
):
    """
    Copied from: https://github.com/mdtraj/mdtraj/blob/main/mdtraj/utils/unitcell.py#L180
    Parameters
    ----------
    a_length : scalar or np.ndarray
        length of Bravais unit vector **a**
    b_length : scalar or np.ndarray
        length of Bravais unit vector **b**
    c_length : scalar or np.ndarray
        length of Bravais unit vector **c**
    alpha : scalar or np.ndarray
        angle between vectors **b** and **c**, in degrees.
    beta : scalar or np.ndarray
        angle between vectors **c** and **a**, in degrees.
    gamma : scalar or np.ndarray
        angle between vectors **a** and **b**, in degrees.

    Returns
    -------
    lx : scalar
        Extent in x direction
    ly : scalar
        Extent in y direction
    lz : scalar
        Extent in z direction
    xy : scalar
        Unit vector **b** tilt with respect to **a**
    xz : scalar
        Unit vector of **c** tilt with respect to **a**
    yz : scalar
        Unit vector of **c** tilt with respect to **b**
    """
    lx = a_length
    xy = b_length * np.cos(np.deg2rad(gamma))
    xz = c_length * np.cos(np.deg2rad(beta))
    ly = np.sqrt(b_length**2 - xy**2)
    yz = (b_length * c_length * np.cos(np.deg2rad(alpha)) - xy * xz) / ly
    lz = np.sqrt(c_length**2 - xz**2 - yz**2)

    return np.array([lx, ly, lz, xy, xz, yz])
