#
#              Rule (MDAnalysis `guess_bonds` + connected-component fragments):
#                  bond(i, j)  iff  0.1 A < d(i, j) < (vdw_i + vdw_j) * 0.55
#              then flag the O of every connected fragment that is exactly three atoms
#              with composition {O, H, H}.
#
#              Two neighbour backends, same criterion:
#                * non-periodic (cell/pbc absent or all-False) -> scipy cKDTree. This is
#                  what the existing omol25 sidecars were generated with, so they stay
#                  bit-for-bit reproducible and need no regeneration.
#                * periodic -> ASE's PBC-aware cell list (minimum-image convention).
#                  Required for solvent boxes: without it every water straddling a box
#                  face splits into 1- and 2-atom fragments and is silently routed HEAVY
#                  (on water216: 188/216 instead of 216/216).

from __future__ import annotations

import numpy as np

# MDAnalysis bond-guessing defaults (mirrored here for speed):
#   threshold(i, j) = (vdw[i] + vdw[j]) * fudge_factor
#   bond valid iff lower_bound < dist(i, j) < threshold
# See MDAnalysis.topology.guessers.guess_bonds.
FUDGE_FACTOR = 0.55
LOWER_BOUND = 0.1

_VDW_TABLE_CACHE: np.ndarray | None = None


def _build_vdw_table() -> np.ndarray:
    """Per-Z vdW radii (Angstrom), shape (119,).

    Mirrors MDAnalysis' bond-guessing inputs: it expects an element-keyed dict where
    pair-bond threshold = (r_i + r_j) * fudge_factor. We use ASE's `vdw_radii`, falling
    back to 1.5x the covalent radius for elements where ASE has no vdW entry (rare metals
    / noble gases). Indexed by Z directly.
    """
    from ase.data import chemical_symbols, covalent_radii, vdw_radii as ase_vdw

    n = len(chemical_symbols)
    out = np.zeros(n, dtype=np.float32)
    for z in range(1, n):
        r = ase_vdw[z] if z < len(ase_vdw) else float("nan")
        if not np.isfinite(r):
            r = float(covalent_radii[z]) * 1.5
        out[z] = float(r)
    return out


def vdw_table() -> np.ndarray:
    global _VDW_TABLE_CACHE
    if _VDW_TABLE_CACHE is None:
        _VDW_TABLE_CACHE = _build_vdw_table()
    return _VDW_TABLE_CACHE


def _flag_water_fragments(
    n: int, bi: np.ndarray, bj: np.ndarray, Z: np.ndarray
) -> np.ndarray:
    """Connected components over the bond graph; flag O of every 3-atom {O,H,H} fragment."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    mask = np.zeros(n, dtype=np.uint8)
    if len(bi) == 0:
        return mask
    data = np.ones(len(bi) * 2, dtype=np.uint8)
    rows = np.concatenate([bi, bj])
    cols = np.concatenate([bj, bi])
    g = csr_matrix((data, (rows, cols)), shape=(n, n))
    _, labels = connected_components(g, directed=False)

    # Build component -> atom_indices with a single argsort.
    order = np.argsort(labels, kind="stable")
    boundaries = np.flatnonzero(np.diff(labels[order])) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]])
    for s, e in zip(starts, ends):
        if e - s != 3:
            continue
        atom_ids = order[s:e]
        zs = Z[atom_ids]
        # multiset == {1, 1, 8}
        if int((zs == 8).sum()) == 1 and int((zs == 1).sum()) == 2:
            mask[atom_ids[zs == 8]] = 1
    return mask


def _is_periodic(cell, pbc) -> bool:
    if pbc is None:
        return False
    pbc_arr = np.asarray(pbc).reshape(-1).astype(bool)
    if not pbc_arr.any():
        return False
    if cell is None:
        return False
    cell_arr = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    # A zeroed cell is how the non-periodic datasets encode "no cell"; treat as aperiodic
    # rather than handing a degenerate cell to ASE.
    return bool(np.abs(cell_arr).sum() > 1e-8)


def predict_hoh_o_mask(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    cell=None,
    pbc=None,
    fudge_factor: float = FUDGE_FACTOR,
) -> np.ndarray:
    """Return a uint8 mask of length N marking HOH oxygens with 1.

    Args:
        positions: (N, 3) Cartesian coordinates in Angstrom.
        atomic_numbers: (N,) atomic numbers.
        cell: optional (3, 3) cell matrix (rows = lattice vectors).
        pbc: optional length-3 booleans. Periodic neighbours are used only when `pbc` has
            at least one True AND `cell` is non-degenerate; otherwise the aperiodic
            cKDTree path runs, reproducing the original non-periodic generator exactly.
        fudge_factor: bond-criterion scale. Exposed because it is a user-facing knob on
            the ASE side (`--hoh_o_fudge_factor`); leave at the default when generating
            training sidecars.
    """
    n = int(len(atomic_numbers))
    if n == 0:
        return np.zeros(0, dtype=np.uint8)

    Z = np.asarray(atomic_numbers, dtype=np.int64)
    pos = np.asarray(positions, dtype=np.float64)
    if pos.shape != (n, 3):
        return np.zeros(n, dtype=np.uint8)

    vdw = vdw_table()
    radii = vdw[np.clip(Z, 0, len(vdw) - 1)]
    if not np.all(radii > 0):
        # Unknown / Z=0 ghost atoms — give up on this sample (same bail-out as the
        # original generator and as ase_runner).
        return np.zeros(n, dtype=np.uint8)

    fudge = float(fudge_factor)
    # Upper-bound bond length: 2 * max(radii) * fudge
    cutoff = 2.0 * float(radii.max()) * fudge

    if _is_periodic(cell, pbc):
        from ase import Atoms
        from ase.neighborlist import neighbor_list

        pbc_arr = np.asarray(pbc).reshape(-1).astype(bool)
        atoms = Atoms(
            numbers=Z,
            positions=pos,
            cell=np.asarray(cell, dtype=np.float64).reshape(3, 3),
            pbc=tuple(bool(x) for x in pbc_arr),
        )
        # One cell-list pass at the largest possible bond length, then the per-pair vdW
        # threshold. neighbor_list applies the minimum-image convention for the periodic
        # directions of `atoms.pbc`.
        i, j, d = neighbor_list("ijd", atoms, cutoff)
        keep = (d > LOWER_BOUND) & (d < (radii[i] + radii[j]) * fudge)
        return _flag_water_fragments(n, i[keep], j[keep], Z)

    from scipy.spatial import cKDTree

    try:
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=cutoff, output_type="ndarray")
    except Exception:
        return np.zeros(n, dtype=np.uint8)
    if pairs.size == 0:
        # No bonds -> no HOH.
        return np.zeros(n, dtype=np.uint8)

    i, j = pairs[:, 0], pairs[:, 1]
    d_ij = np.linalg.norm(pos[i] - pos[j], axis=1)
    keep = (d_ij > LOWER_BOUND) & (d_ij < (radii[i] + radii[j]) * fudge)
    return _flag_water_fragments(n, i[keep], j[keep], Z)
