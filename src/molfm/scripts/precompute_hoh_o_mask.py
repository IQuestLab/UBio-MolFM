# Identify low-precision oxygen atoms (HOH water-O) per LMDB sample and write a
# `<db>.hoh_o_mask.npy` uint8 sidecar concatenated in row-id order (matches
# `atom_cnts`, which is what MoleculeLMDBDataset uses to slice it).
#
# The detection rule lives in `molfm.data.hoh_o_mask` and is shared with the ASE
# inference path (ase_runner._guess_hoh_o), so training and MD agree on which
# oxygens are water. `--pbc` selects the minimum-image convention for periodic
# solvent boxes; without it, a water straddling a box face splits into fragments
# and is silently routed heavy (on water216: 188/216).
#
# Usage:
#   # periodic bulk water
#   python src/molfm/scripts/precompute_hoh_o_mask.py \
#       --data_path /path/to/lmdb/train /path/to/lmdb/val \
#       --pbc auto --workers 8 --report report_hoh_o_water.json
#
#   # non-periodic clusters
#   python src/molfm/scripts/precompute_hoh_o_mask.py \
#       --data_path data/omol25/train --pbc false --workers 32
#
# Idempotent: shards whose sidecar already exists and matches the row count are skipped.
# Safe to interrupt and re-run.

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from molfm.data.hoh_o_mask import predict_hoh_o_mask  # noqa: E402

# Suppress MDAnalysis bond-guess warnings (we know what we're doing).
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis")


def _detect_res_format(
    res_name_to_atoms: dict,
    res_id_to_dict_idx: dict | None,
    Z: np.ndarray,
) -> str | None:
    """Detect whether res_name_to_atoms[rn][gi] holds atom indices or Z values.

    Two formats coexist across processing pipelines:
      - "indices": values are atom indices in [0, n_atoms); union == 0..n-1.
      - "atomic_numbers": values are atomic numbers; per-residue groups are stored in
        residue order, so atom i in residue R (offset O) maps to global index O+i.
    Returns None if neither format fits.
    """
    n = int(len(Z))

    # Try "indices": concatenate all arrays and check they cover [0, n-1].
    try:
        flat = []
        for v in res_name_to_atoms.values():
            for arr in v:
                flat.append(np.asarray(arr, dtype=np.int64))
        if flat:
            cat = np.concatenate(flat) if len(flat) > 1 else flat[0]
            if (
                cat.size == n
                and int(cat.min()) >= 0
                and int(cat.max()) < n
                and np.unique(cat).size == n
            ):
                return "indices"
    except Exception:
        pass

    # Try "atomic_numbers": walk residues in res_id order, see whether the concatenated
    # arrays equal Z exactly.
    if res_id_to_dict_idx:
        try:
            offset = 0
            matched = 0
            for rid in sorted(res_id_to_dict_idx):
                rn, gi = res_id_to_dict_idx[rid]
                arr = np.asarray(res_name_to_atoms[rn][gi], dtype=np.int64)
                end = offset + arr.size
                if end > n:
                    return None
                if not np.array_equal(arr, Z[offset:end]):
                    return None
                matched += 1
                offset = end
            if offset == n and matched > 0:
                return "atomic_numbers"
        except Exception:
            pass
    return None


def extract_gt_hoh_o(
    Z: np.ndarray,
    res_name_to_atoms: dict | None,
    res_id_to_dict_idx: dict | None,
) -> np.ndarray | None:
    """Return ground-truth HOH-O mask or None if residue info is missing/unparseable.

    Handles two coexisting LMDB residue-info formats; see `_detect_res_format`.
    """
    if not res_name_to_atoms:
        return None
    n = int(len(Z))
    fmt = _detect_res_format(res_name_to_atoms, res_id_to_dict_idx, Z)
    if fmt is None:
        return None
    mask = np.zeros(n, dtype=np.uint8)
    if fmt == "indices":
        for arr in res_name_to_atoms.get("HOH", []):
            idxs = np.asarray(arr, dtype=np.int64)
            for i in idxs:
                ii = int(i)
                if 0 <= ii < n and int(Z[ii]) == 8:
                    mask[ii] = 1
        return mask
    # fmt == "atomic_numbers": walk residues in order, indices = (offset, offset+1, ...)
    offset = 0
    for rid in sorted(res_id_to_dict_idx):
        rn, gi = res_id_to_dict_idx[rid]
        arr = np.asarray(res_name_to_atoms[rn][gi], dtype=np.int64)
        if rn == "HOH":
            for k, z in enumerate(arr):
                if int(z) == 8:
                    mask[offset + k] = 1
        offset += arr.size
    return mask


@dataclass
class ShardStats:
    path: str
    n_rows: int = 0
    n_O_atoms: int = 0           # total O atoms across all samples
    n_pred_hoh_o: int = 0        # O atoms flagged as HOH-O by the geometric rule
    n_periodic_samples: int = 0  # samples whose neighbours used the PBC backend
    n_samples_with_gt: int = 0
    n_samples_exact_match: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    elapsed_s: float = 0.0
    status: str = "written"      # "written" | "skipped" | "validated" | "failed"
    error: str = ""

    def add_sample(self, pred: np.ndarray, gt: np.ndarray | None, is_O: np.ndarray) -> None:
        self.n_O_atoms += int(is_O.sum())
        self.n_pred_hoh_o += int(pred.astype(bool).sum())
        if gt is None:
            return
        self.n_samples_with_gt += 1
        # restrict accounting to O atoms only — TN over all atoms would be uninformative
        pred_o = pred[is_O].astype(bool)
        gt_o = gt[is_O].astype(bool)
        tp = int((pred_o & gt_o).sum())
        fp = int((pred_o & ~gt_o).sum())
        fn = int((~pred_o & gt_o).sum())
        tn = int((~pred_o & ~gt_o).sum())
        self.tp += tp; self.fp += fp; self.fn += fn; self.tn += tn
        if fp == 0 and fn == 0:
            self.n_samples_exact_match += 1


@dataclass
class DatasetAgg:
    root: str
    n_shards: int = 0
    n_rows: int = 0
    n_O_atoms: int = 0
    n_pred_hoh_o: int = 0
    n_periodic_samples: int = 0
    n_samples_with_gt: int = 0
    n_samples_exact_match: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    shards: list[str] = field(default_factory=list)

    def absorb(self, s: ShardStats) -> None:
        self.n_shards += 1
        self.n_rows += s.n_rows
        self.n_O_atoms += s.n_O_atoms
        self.n_pred_hoh_o += s.n_pred_hoh_o
        self.n_periodic_samples += s.n_periodic_samples
        self.n_samples_with_gt += s.n_samples_with_gt
        self.n_samples_exact_match += s.n_samples_exact_match
        self.tp += s.tp; self.fp += s.fp; self.fn += s.fn; self.tn += s.tn
        self.shards.append(s.path)

    def summary(self) -> dict:
        d = (self.tp + self.fp + self.fn + self.tn)
        acc = (self.tp + self.tn) / d if d else None
        prec = self.tp / max(self.tp + self.fp, 1) if self.tp + self.fp else None
        rec = self.tp / max(self.tp + self.fn, 1) if self.tp + self.fn else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
        exact = (
            self.n_samples_exact_match / self.n_samples_with_gt
            if self.n_samples_with_gt else None
        )
        return dict(
            root=self.root, n_shards=self.n_shards, n_rows=self.n_rows,
            n_O_atoms=self.n_O_atoms, n_pred_hoh_o=self.n_pred_hoh_o,
            hoh_o_fraction=(self.n_pred_hoh_o / self.n_O_atoms) if self.n_O_atoms else None,
            n_periodic_samples=self.n_periodic_samples,
            n_samples_with_gt=self.n_samples_with_gt,
            exact_match_rate=exact,
            tp=self.tp, fp=self.fp, fn=self.fn, tn=self.tn,
            accuracy=acc, precision=prec, recall=rec, f1=f1,
        )


def _extract_pos_Z(obj) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(obj, "pos"):
        pos = np.asarray(obj.pos)
    elif isinstance(obj, dict) and "pos" in obj:
        pos = np.asarray(obj["pos"])
    else:
        raise AttributeError("no 'pos' on LMDB row")
    if hasattr(obj, "atomic_numbers"):
        Z = np.asarray(obj.atomic_numbers)
    elif isinstance(obj, dict) and "atomic_numbers" in obj:
        Z = np.asarray(obj["atomic_numbers"])
    else:
        raise AttributeError("no 'atomic_numbers' on LMDB row")
    return pos, Z.astype(np.int64)


def _extract_res(obj) -> tuple[dict | None, dict | None]:
    rna = getattr(obj, "res_name_to_atoms", None)
    rid = getattr(obj, "res_id_to_dict_idx", None)
    if rna is None and isinstance(obj, dict):
        rna = obj.get("res_name_to_atoms")
    if rid is None and isinstance(obj, dict):
        rid = obj.get("res_id_to_dict_idx")
    return rna, rid


def _extract_cell_pbc(obj, pbc_mode: str):
    """Resolve (cell, pbc) for the neighbour backend given --pbc.

    "false" -> always aperiodic (KD-tree; the original behaviour).
    "true"  -> force periodic, using the sample's own cell (error if absent).
    "auto"  -> periodic iff the sample carries a non-degenerate cell AND a truthy pbc.
    """
    if pbc_mode == "false":
        return None, None

    def _get(name):
        v = getattr(obj, name, None)
        if v is None and isinstance(obj, dict):
            v = obj.get(name)
        return v

    cell = _get("cell")
    pbc = _get("pbc")
    if cell is None:
        if pbc_mode == "true":
            raise ValueError("--pbc true but LMDB row carries no 'cell'")
        return None, None
    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    if pbc_mode == "true":
        pbc = (True, True, True) if pbc is None else np.asarray(pbc).reshape(-1)
    elif pbc is None:
        return None, None
    return cell, pbc


def _read_atom_cnts(db_path: str, length: int, txn=None) -> np.ndarray | None:
    """atom_cnts in row order, from the `.atom_cnts.npy` sidecar or the in-DB key.

    Mirrors MoleculeLMDBDataset.open_db, which prefers the sidecar and falls back to the
    in-DB `atom_cnts` key; some shards only have the latter.
    """
    ac_path = db_path + ".atom_cnts.npy"
    if os.path.exists(ac_path):
        ac = np.load(ac_path)
        return ac if ac.shape == (length,) else None
    if txn is not None:
        raw = txn.get(b"atom_cnts")
        if raw is not None:
            ac = np.asarray(pickle.loads(raw), dtype=np.int64)
            return ac if ac.shape == (length,) else None
    return None


def process_lmdb_shard(
    db_path: str, force: bool = False, validate_only: bool = False,
    pbc_mode: str = "auto",
) -> ShardStats:
    """Handle the torch_geometric.Data-pickle LMDB format."""
    import lmdb
    stats = ShardStats(path=db_path)
    t0 = time.time()
    sidecar = db_path + ".hoh_o_mask.npy"

    env = lmdb.open(
        db_path, subdir=False, readonly=True, lock=False,
        readahead=True, meminit=False, max_readers=32,
    )
    try:
        txn = env.begin()
        length_raw = txn.get(b"length")
        if length_raw is not None:
            length = pickle.loads(length_raw)
        else:
            length = env.stat()["entries"]
        stats.n_rows = int(length)

        atom_cnts = _read_atom_cnts(db_path, length, txn=txn)

        # Idempotent skip.
        if (
            (not force) and (not validate_only) and os.path.exists(sidecar)
            and atom_cnts is not None
        ):
            existing = np.load(sidecar, mmap_mode="r")
            if existing.shape == (int(atom_cnts.sum()),):
                stats.elapsed_s = 0.0
                stats.status = "skipped"
                return stats

        per_sample_masks: list[np.ndarray] = []
        for i in range(length):
            raw = txn.get(f"{i}".encode("ascii"))
            if raw is None:
                per_sample_masks.append(np.zeros(0, dtype=np.uint8))
                continue
            obj = pickle.loads(raw)
            pos, Z = _extract_pos_Z(obj)
            cell, pbc = _extract_cell_pbc(obj, pbc_mode)
            if cell is not None and pbc is not None and np.asarray(pbc).any():
                stats.n_periodic_samples += 1
            pred = predict_hoh_o_mask(pos, Z, cell=cell, pbc=pbc)
            rna, rid = _extract_res(obj)
            gt = extract_gt_hoh_o(Z, rna, rid)
            is_O = (Z == 8)
            stats.add_sample(pred, gt, is_O)
            per_sample_masks.append(pred)

        if not validate_only:
            full = np.concatenate(per_sample_masks).astype(np.uint8)
            if atom_cnts is not None and full.shape[0] != int(atom_cnts.sum()):
                raise ValueError(
                    f"mask length {full.shape[0]} != sum(atom_cnts) "
                    f"{int(atom_cnts.sum())} for {db_path}"
                )
            np.save(sidecar, full)
            stats.status = "written"
        else:
            stats.status = "validated"
    finally:
        env.close()
    stats.elapsed_s = time.time() - t0
    return stats


def process_aselmdb_shard(
    db_path: str, force: bool = False, validate_only: bool = False,
    pbc_mode: str = "auto",
) -> ShardStats:
    """Handle ASE-DB-backed aselmdb files (omol25). No residue info to validate against."""
    import ase.db
    stats = ShardStats(path=db_path)
    t0 = time.time()
    sidecar = db_path + ".hoh_o_mask.npy"

    db = ase.db.connect(db_path, type="aselmdb", readonly=True)
    ids = db.ids
    length = len(ids)
    stats.n_rows = int(length)

    atom_cnts = _read_atom_cnts(db_path, length)

    if (
        (not force) and (not validate_only) and os.path.exists(sidecar)
        and atom_cnts is not None
    ):
        existing = np.load(sidecar, mmap_mode="r")
        if existing.shape == (int(atom_cnts.sum()),):
            stats.elapsed_s = 0.0
            stats.status = "skipped"
            return stats

    per_sample_masks: list[np.ndarray] = []
    for rid in ids:
        row = db._get_row(rid)
        Z = np.asarray(row.numbers, dtype=np.int64)
        pos = np.asarray(row.positions, dtype=np.float64)
        cell, pbc = (None, None)
        if pbc_mode != "false":
            row_pbc = getattr(row, "pbc", None)
            row_cell = getattr(row, "cell", None)
            if row_cell is not None and (
                pbc_mode == "true" or (row_pbc is not None and np.asarray(row_pbc).any())
            ):
                cell = np.asarray(row_cell, dtype=np.float64).reshape(3, 3)
                pbc = (True, True, True) if row_pbc is None else np.asarray(row_pbc)
        if cell is not None and pbc is not None and np.asarray(pbc).any():
            stats.n_periodic_samples += 1
        pred = predict_hoh_o_mask(pos, Z, cell=cell, pbc=pbc)
        is_O = (Z == 8)
        stats.add_sample(pred, None, is_O)
        per_sample_masks.append(pred)

    if not validate_only:
        full = np.concatenate(per_sample_masks).astype(np.uint8) if per_sample_masks \
            else np.zeros(0, dtype=np.uint8)
        np.save(sidecar, full)
        stats.status = "written"
    else:
        stats.status = "validated"
    stats.elapsed_s = time.time() - t0
    return stats


def _worker(db_path: str, force: bool, validate_only: bool, pbc_mode: str) -> ShardStats:
    try:
        if db_path.endswith(".aselmdb"):
            return process_aselmdb_shard(
                db_path, force=force, validate_only=validate_only, pbc_mode=pbc_mode)
        return process_lmdb_shard(
            db_path, force=force, validate_only=validate_only, pbc_mode=pbc_mode)
    except Exception as exc:  # noqa: BLE001
        return ShardStats(path=db_path, status="failed", error=repr(exc))


def discover_shards(data_path: str) -> list[str]:
    if os.path.isfile(data_path) and (
        data_path.endswith(".lmdb") or data_path.endswith(".aselmdb")
    ):
        return [data_path]
    shards: list[str] = []
    for root, _, files in os.walk(data_path):
        for f in files:
            if f.endswith(".lmdb") or f.endswith(".aselmdb"):
                shards.append(os.path.join(root, f))
    return sorted(shards)


def _format_pct(x: float | None) -> str:
    return f"{x*100:.2f}%" if x is not None else "n/a"


def print_dataset_summary(agg: DatasetAgg) -> None:
    s = agg.summary()
    print(f"\n[{agg.root}]")
    print(f"  shards={s['n_shards']} rows={s['n_rows']:,} O_atoms={s['n_O_atoms']:,}")
    print(
        f"  flagged HOH-O={s['n_pred_hoh_o']:,} ({_format_pct(s['hoh_o_fraction'])} of O)"
        f"  periodic_samples={s['n_periodic_samples']:,}"
    )
    if s["n_samples_with_gt"]:
        print(
            f"  has_gt_samples={s['n_samples_with_gt']:,}"
            f"  exact_match_rate={_format_pct(s['exact_match_rate'])}"
        )
        print(
            f"  O-atom-level: TP={s['tp']:,} FP={s['fp']:,} FN={s['fn']:,} TN={s['tn']:,}"
        )
        print(
            f"  accuracy={_format_pct(s['accuracy'])}"
            f"  precision={_format_pct(s['precision'])}"
            f"  recall={_format_pct(s['recall'])}"
            f"  F1={_format_pct(s['f1'])}"
        )
    else:
        print("  (no ground-truth residue info found in this dataset — mask only)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_path", required=True, nargs="+",
        help="One or more directories containing *.lmdb files (or a single .lmdb).",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--pbc", choices=["auto", "true", "false"], default="auto",
        help="Neighbour backend: 'false' = always aperiodic KD-tree (the original "
             "behaviour, correct for vacuum clusters like omol25); 'true' = force the "
             "PBC-aware cell list using each sample's cell; 'auto' (default) = periodic "
             "iff the sample carries a non-degenerate cell and a truthy pbc.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if a matching sidecar already exists.",
    )
    parser.add_argument(
        "--validate_only", action="store_true",
        help="Compute and report metrics but do NOT write sidecars.",
    )
    parser.add_argument(
        "--accuracy_threshold", type=float, default=0.95,
        help="Per-O-atom accuracy threshold; exit code 2 if any GT-bearing dataset falls below.",
    )
    parser.add_argument("--report", type=str, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    # Group shards by data_path root so we can report per-dataset stats.
    roots: list[tuple[str, list[str]]] = []
    for dp in args.data_path:
        shards = discover_shards(dp)
        if not shards:
            print(f"WARN: no LMDB files under {dp}", file=sys.stderr)
        roots.append((dp, shards))
    total = sum(len(s) for _, s in roots)
    if total == 0:
        print("No shards discovered.", file=sys.stderr)
        sys.exit(1)
    print(f"Discovered {total} shard(s) across {len(roots)} dataset path(s). "
          f"Workers: {args.workers}. pbc={args.pbc}.")

    aggs = {dp: DatasetAgg(root=dp) for dp, _ in roots}
    shard_to_root = {sp: dp for dp, shards in roots for sp in shards}
    all_shards = [sp for _, shards in roots for sp in shards]

    failed: list[tuple[str, str]] = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_worker, sp, args.force, args.validate_only, args.pbc): sp
            for sp in all_shards
        }
        pbar = tqdm(total=len(futs), desc="shards", unit="shard")
        for fut in as_completed(futs):
            sp = futs[fut]
            stats: ShardStats = fut.result()
            if stats.status == "failed":
                failed.append((sp, stats.error))
            else:
                aggs[shard_to_root[sp]].absorb(stats)
            pbar.set_postfix(
                written=sum(1 for a in aggs.values() for _ in a.shards),
                failed=len(failed),
            )
            pbar.update(1)
        pbar.close()
    elapsed = time.time() - t_start
    print(f"Done in {elapsed/60:.1f} min.")

    # Print per-dataset summaries.
    summaries = []
    for dp, _ in roots:
        agg = aggs[dp]
        print_dataset_summary(agg)
        summaries.append(agg.summary())

    if failed:
        print(f"\nFailures: {len(failed)} shard(s)")
        for p, e in failed[:20]:
            print(f"  {p}: {e}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(
                {"datasets": summaries, "elapsed_s": elapsed, "pbc": args.pbc,
                 "failed": [{"path": p, "error": e} for p, e in failed]},
                f, indent=2,
            )
        print(f"\nReport written to {args.report}")

    # Threshold check on datasets that have ground truth.
    below = []
    for s in summaries:
        if s["n_samples_with_gt"] and s["accuracy"] is not None:
            if s["accuracy"] < args.accuracy_threshold:
                below.append((s["root"], s["accuracy"]))
    if below:
        print(f"\nFAIL: {len(below)} dataset(s) below accuracy threshold "
              f"{args.accuracy_threshold*100:.1f}%:")
        for r, a in below:
            print(f"  {r}: accuracy={a*100:.2f}%")
        sys.exit(2)
    if failed:
        sys.exit(3)


if __name__ == "__main__":
    main()
