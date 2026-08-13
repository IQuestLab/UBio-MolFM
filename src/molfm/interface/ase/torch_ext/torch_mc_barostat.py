# Description: Energy-only isotropic Monte-Carlo barostat for the GPU-native ASE
#              stack. E2Former predicts only energy + forces (no stress/virial),
#              so a differential barostat is impossible; an MC volume move needs
#              only get_potential_energy(). Volume moves are random walks in ln V;
#              molecular centres of mass are scaled (NOT every atom — that would
#              distort intramolecular bonds of the flexible all-atom water model
#              and collapse the acceptance ratio).

import math
from typing import Optional, Tuple

import numpy as np
import torch
from ase import units
from ase.data import atomic_masses
from loguru import logger

# Covalent radii (Angstrom) by atomic number, for distance-based bond perception.
# Values from Cordero et al. 2008; only common bio/organic elements are needed.
_COV_RADII = {
    1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    11: 1.66, 12: 1.41, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
    19: 2.03, 20: 1.76, 26: 1.32, 30: 1.22, 35: 1.20, 53: 1.39,
}
_DEFAULT_COV = 0.77


def build_molecule_index(
    positions: np.ndarray,
    numbers: np.ndarray,
    box_diag: np.ndarray,
    source: str = "auto",
    tol: float = 0.30,
) -> np.ndarray:
    """Return an ``atom -> molecule id`` integer array (canonical atom order).

    Parameters
    ----------
    positions : (N, 3) float array, Angstrom (any wrapping; PBC handled here).
    numbers   : (N,) int atomic numbers.
    box_diag  : (3,) float orthorhombic box lengths (Angstrom). Used for the
                minimum-image bond search; assumes an (approximately) orthorhombic
                cell, which is the case for the cubic solvent boxes used here.
    source    : "auto"/"bondgraph" -> distance bond graph + connected components;
                "stride:N"        -> every N consecutive atoms form one molecule
                                     (e.g. "stride:3" for pre-ordered O,H,H water);
                "residue"         -> not available from TorchAtoms; falls back to
                                     bondgraph with a warning.
    tol       : bond tolerance (Angstrom) added to the sum of covalent radii.
                Default 0.30. NOTE: the previous 0.45 was too loose for dense /
                compressed water — a short O...H hydrogen bond (down to ~1.38 A in
                an under-relaxed box) fell within the O-H covalent threshold
                (0.66+0.31+0.45 = 1.42 A) and a connected-components pass then
                fused the two waters into one 2O+4H "molecule". The barostat would
                then rigidly co-scale that pair, injecting non-physical stress.
                With tol=0.30 the O-H threshold is 1.27 A: above the real O-H bond
                (<=1.07 A here) yet below such spurious H-bonds.
    """
    n = int(len(numbers))
    if source.startswith("stride:"):
        k = int(source.split(":", 1)[1])
        if n % k != 0:
            raise ValueError(f"stride:{k} does not divide N={n}")
        return (np.arange(n) // k).astype(np.int64)

    if source not in ("auto", "bondgraph", "residue"):
        raise ValueError(f"Unknown molecule_source: {source!r}")
    if source == "residue":
        logger.warning(
            "[barostat] molecule_source='residue' is unavailable from TorchAtoms "
            "(no residue records on the GPU path); falling back to 'bondgraph'."
        )

    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    L = np.asarray(box_diag, dtype=float)
    rcov = np.array([_COV_RADII.get(int(z), _DEFAULT_COV) for z in numbers])
    rmax = float(2.0 * rcov.max() + tol)

    pos_w = positions % L
    tree = cKDTree(pos_w, boxsize=L)
    pairs = tree.query_pairs(rmax, output_type="ndarray")

    if len(pairs) > 0:
        dvec = positions[pairs[:, 0]] - positions[pairs[:, 1]]
        dvec -= L * np.round(dvec / L)                      # minimum image
        d = np.linalg.norm(dvec, axis=1)
        thr = rcov[pairs[:, 0]] + rcov[pairs[:, 1]] + tol
        keep = (d < thr) & (d > 0.4)
        pairs = pairs[keep]

    if len(pairs) == 0:
        return np.arange(n, dtype=np.int64)                 # all monatomic

    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    labels = labels.astype(np.int64)

    # --- sanity check: surface a malformed molecule map instead of letting the
    # barostat silently co-scale a wrongly-fused pair. A correct water box is all
    # 3-atom (1,1,8) components; a fused pair shows up as an oversized component
    # (e.g. 2O+4H). We warn (not raise) because solutions legitimately contain
    # non-water species — but for a pure-water box this catches the tol failure.
    sizes = np.bincount(labels, minlength=n_comp)
    oversized = np.where(sizes > 3)[0]
    if len(oversized) > 0:
        examples = []
        for c in oversized[:5]:
            idx = np.where(labels == c)[0]
            z = sorted(int(numbers[i]) for i in idx)
            examples.append(f"comp{c}(size={len(idx)}, Z={z})")
        logger.warning(
            f"[barostat] build_molecule_index: {len(oversized)} component(s) "
            f"larger than 3 atoms detected (tol={tol} A) — possible spurious "
            f"inter-molecule bond fusing distinct molecules. For pure water this "
            f"is a bug (expect all 3-atom H2O); consider a tighter tol. "
            f"Examples: {', '.join(examples)}"
        )
    return labels


class TorchMCBarostat:
    """Isotropic energy-only Monte-Carlo barostat (ln-V moves, COM scaling).

    The orchestration of the in-place trial move lives in
    ``ASERunner._mc_barostat_attempt``. This class owns only: the proposal RNG,
    the COM-preserving position scaling, and the Metropolis acceptance test.
    All physics quantities are in ASE units
    (energy eV, length Angstrom, pressure eV/Angstrom^3 via ``units.bar``).
    """

    def __init__(
        self,
        pressure_bar: float,
        temperature_K: float,
        n_mol: int,
        mol_index: Optional[torch.Tensor] = None,
        mol_first_atom: Optional[torch.Tensor] = None,
        masses: Optional[torch.Tensor] = None,
        dlnV: float = 0.002,
        seed: int = 12345,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float64,
    ):
        self.P = float(pressure_bar) * units.bar            # eV / Angstrom^3
        self.kT = units.kB * float(temperature_K)           # eV
        self.n_mol = int(n_mol)
        self.dlnV = float(dlnV)
        self.device = device
        self.dtype = dtype
        self._rng = np.random.default_rng(seed)

        # Per-atom scaling data.
        self.mol_index = mol_index
        self.mol_first_atom = mol_first_atom
        self.masses = masses

        self.n_try = 0
        self.n_acc = 0

    # ------------------------------------------------------------------
    # Proposal + acceptance
    # ------------------------------------------------------------------
    def propose(self) -> Tuple[float, float]:
        """Draw a ln-volume step and a uniform acceptance number."""
        delta = float(self._rng.uniform(-self.dlnV, self.dlnV))
        u = float(self._rng.random())
        return delta, u

    def accept(self, dE: float, V_old: float, V_new: float, u: float) -> bool:
        """Metropolis criterion for an ln-V move.

        dW = dE + P (V_new - V_old) - (N_mol + 1) kT ln(V_new / V_old)

        The ``(N_mol + 1)`` factor is the Jacobian for sampling uniformly in
        ln V (an extra +1 over the linear-V Jacobian of N_mol). It supplies the
        ideal-gas contribution, so no separate N kT/V term is added.
        """
        dW = (dE
              + self.P * (V_new - V_old)
              - (self.n_mol + 1) * self.kT * math.log(V_new / V_old))
        if dW <= 0.0:
            return True
        return u < math.exp(-dW / self.kT)

    def record(self, accepted: bool) -> None:
        self.n_try += 1
        if accepted:
            self.n_acc += 1

    @property
    def accept_ratio(self) -> float:
        return self.n_acc / max(1, self.n_try)

    # ------------------------------------------------------------------
    # COM-preserving isotropic scaling of the full configuration
    # ------------------------------------------------------------------
    def scale_positions(self, pos: torch.Tensor, s: float, cell: torch.Tensor) -> torch.Tensor:
        """Scale every molecule's centre of mass by ``s`` about the cell origin,
        keeping each molecule's internal geometry rigid.

        Molecules are first unwrapped (minimum image relative to their first
        atom, in fractional coordinates — valid for molecules whose intramolecular
        extent is < L/2, which holds for water and the compact peptide solute) so
        the COM is well-defined even when a molecule straddles a periodic face.

        Returns new Cartesian positions in the scaled box ``s * cell``.
        """
        if self.mol_index is None:
            raise RuntimeError("scale_positions called without a molecule map.")
        pos = pos.to(device=self.device, dtype=self.dtype)
        cell = cell.to(device=self.device, dtype=self.dtype)

        inv = torch.linalg.inv(cell)
        frac = pos @ inv                                    # (N, 3) fractional
        ref = frac[self.mol_first_atom][self.mol_index]     # first-atom frac per atom
        dfrac = frac - ref
        dfrac = dfrac - torch.round(dfrac)                  # minimum image (all-periodic)
        pos_uw = (ref + dfrac) @ cell                       # unwrapped Cartesian

        mass = self.masses.to(device=self.device, dtype=self.dtype).unsqueeze(-1)
        num = torch.zeros((self.n_mol, 3), device=self.device, dtype=self.dtype)
        den = torch.zeros((self.n_mol, 1), device=self.device, dtype=self.dtype)
        num.index_add_(0, self.mol_index, pos_uw * mass)
        den.index_add_(0, self.mol_index, mass)
        com = num / den                                     # (n_mol, 3)

        new_pos = pos_uw + (s - 1.0) * com[self.mol_index]

        # Wrap back into the SCALED cell. After COM scaling the atoms can lie
        # outside [0, L); leaving them there makes the E2Former FAISS neighbour
        # search (which uses the atom bounding box) inconsistent and yields a
        # WRONG energy — the original cause of spurious runaway compression.
        # Per-atom wrap is PBC-exact; split molecules are handled by the
        # calculator's cell expansion. The COM-scaling (volume move) is preserved.
        new_cell = cell * s
        inv_new = torch.linalg.inv(new_cell)
        frac = new_pos @ inv_new
        frac = frac - torch.floor(frac)                     # wrap to [0, 1)
        return frac @ new_cell


def molecule_first_atom(mol_index: torch.Tensor, n_mol: int) -> torch.Tensor:
    """Return, for each molecule id, the smallest atom index belonging to it."""
    n = mol_index.numel()
    arange = torch.arange(n, device=mol_index.device, dtype=torch.long)
    first = torch.full((n_mol,), n, device=mol_index.device, dtype=torch.long)
    first.scatter_reduce_(0, mol_index, arange, reduce="amin", include_self=True)
    return first


def masses_from_numbers(numbers_np: np.ndarray, device, dtype) -> torch.Tensor:
    """Per-atom masses (amu) tensor from atomic numbers."""
    return torch.from_numpy(atomic_masses[numbers_np]).to(device=device, dtype=dtype)
