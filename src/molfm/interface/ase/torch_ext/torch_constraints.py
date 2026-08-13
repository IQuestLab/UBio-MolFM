# Description: Torch-native constraint implementations and registration mechanism for GPU MD.

from typing import TYPE_CHECKING, Dict, Type

import torch
from ase.constraints import FixCom

if TYPE_CHECKING:
    from molfm.interface.ase.torch_ext.torch_atoms import TorchAtoms

TORCH_CONSTRAINT_REGISTRY: Dict[Type, Type] = {}


def register_torch_constraint(ase_constraint_cls: Type):
    """Decorator that registers a torch-native implementation for an ASE constraint class."""

    def decorator(torch_impl_cls: Type):
        TORCH_CONSTRAINT_REGISTRY[ase_constraint_cls] = torch_impl_cls
        return torch_impl_cls

    return decorator


class TorchConstraint:
    """Base interface for a torch-native constraint.

    All methods accept and return ``torch.Tensor`` on the atoms' device.
    """

    def adjust_positions(
        self, atoms: "TorchAtoms", newpositions_t: torch.Tensor
    ) -> torch.Tensor:
        """Adjust positions tensor in-place logic, returning the modified tensor."""
        return newpositions_t

    def adjust_momenta(
        self, atoms: "TorchAtoms", momenta_t: torch.Tensor
    ) -> torch.Tensor:
        """Adjust momenta tensor, returning the modified tensor."""
        return momenta_t

    def adjust_forces(
        self, atoms: "TorchAtoms", forces_t: torch.Tensor
    ) -> torch.Tensor:
        """Adjust forces tensor, returning the modified tensor."""
        return forces_t

    def redistribute_forces_md(
        self, atoms: "TorchAtoms", forces_t: torch.Tensor, rand: bool = False
    ) -> torch.Tensor:
        """Redistribute forces for MD constraints (e.g. RATTLE), returning the modified tensor."""
        return forces_t


class TorchFixCom(TorchConstraint):
    """Torch-native FixCom: keeps the center of mass stationary."""

    def adjust_positions(
        self, atoms: "TorchAtoms", newpositions_t: torch.Tensor
    ) -> torch.Tensor:
        masses = atoms._torch_masses

        mass_sum = masses.sum()
        old_cm = (masses.unsqueeze(-1) * atoms._torch_positions).sum(dim=0) / mass_sum
        new_cm = (masses.unsqueeze(-1) * newpositions_t).sum(dim=0) / mass_sum

        return newpositions_t + (old_cm - new_cm)

    def adjust_momenta(
        self, atoms: "TorchAtoms", momenta_t: torch.Tensor
    ) -> torch.Tensor:
        masses = atoms._torch_masses

        v_com = momenta_t.sum(dim=0) / masses.sum()

        return momenta_t - masses.unsqueeze(-1) * v_com

    def adjust_forces(
        self, atoms: "TorchAtoms", forces_t: torch.Tensor
    ) -> torch.Tensor:
        """Follows ASE's official equations: Eqs. (3) and (7) in https://doi.org/10.1021/jp9722824"""
        masses = atoms._torch_masses

        num = (masses.unsqueeze(-1) * forces_t).sum(dim=0)  # Shape: (3,)
        den = masses.square().sum()                         # scalar
        lmd = num / den

        return forces_t - masses.unsqueeze(-1) * lmd

    def redistribute_forces_md(
        self, atoms: "TorchAtoms", forces_t: torch.Tensor, rand: bool = False
    ) -> torch.Tensor:
        """
        Eliminate COM drift from Langevin stochastic noise (xi, eta).
        Crucial for preventing hidden heat leaks in the thermostat.
        """
        return self.adjust_forces(atoms, forces_t)

# Register the constraint
register_torch_constraint(FixCom)(TorchFixCom)
register_torch_constraint(TorchFixCom)(TorchFixCom)
