# Description: Async HDF5 trajectory writer for big-data (GB-scale per frame).
# Supports non-blocking GPU-to-CPU transfer and FP32 compression.
# Contributor note: hardened with append-only commit tracking,
# crash recovery, and safer file close semantics.

import os
import numpy as np
import h5py
import torch
import threading
import queue
import time
from typing import Dict, Any, Optional
from loguru import logger
# ase.Atoms and TorchAtoms are no longer required at import time;
# to_ase_atoms() imports Atoms lazily when needed.

class H5Trajectory:
    """
    High-performance HDF5 trajectory writer for large-scale MD.
    Uses a background thread to handle blocking I/O, allowing the simulation to proceed.
    Includes backpressure control and FP32 precision toggle for massive systems.
    """
    def __init__(self, filename: str, mode: str = 'a',
                 atomic_numbers=None, pbc=None, natoms: int = 0,
                 save_in_fp32: bool = False, max_queue_size: int = 5,
                 enable_swmr: bool = False, recover_on_open_error: bool = True,
                 append_only: bool = True):
        self.filename = filename
        self.save_in_fp32 = save_in_fp32
        self.enable_swmr = enable_swmr
        self.recover_on_open_error = recover_on_open_error

        # Read-only mode: skip all init-time writes so the file is never
        # mutated by concurrent readers or pure-inspection tools.
        self._readonly = (mode == 'r')
        self.append_only = append_only and not self._readonly

        self.dtype_str = 'f4' if save_in_fp32 else 'f8'
        self.torch_dtype = torch.float32 if save_in_fp32 else torch.float64
        self._closed = False
        self._committed_frames = 0

        self.file = self._open_h5_file(filename, mode)
        if self.enable_swmr and not self._readonly:
            try:
                if self.file.mode in ('w', 'a', 'r+'):
                    self.file.swmr_mode = True
            except Exception as exc:
                logger.warning(f"Unable to enable SWMR for {filename}: {exc}")

        # Store static metadata once (write modes only).  Defaults used when
        # no atomic_numbers are provided, so that every H5 file is
        # self-contained and readers never need external fallbacks.
        if 'atomic_numbers' not in self.file and not self._readonly:
            if atomic_numbers is not None:
                if hasattr(atomic_numbers, "cpu"):
                    atomic_numbers = atomic_numbers.cpu().numpy()
                self.file.create_dataset('atomic_numbers', data=np.asarray(atomic_numbers))

                if pbc is not None:
                    if hasattr(pbc, "cpu"):
                        pbc = pbc.cpu().numpy()
                    self.file.create_dataset('pbc', data=np.asarray(pbc).astype(bool))
                else:
                    self.file.create_dataset('pbc', data=np.array([False, False, False]))
            else:
                self.file.create_dataset('atomic_numbers', data=np.array([], dtype=int))
                self.file.create_dataset('pbc', data=np.array([False, False, False]))
        if self.append_only:
            self._init_append_state()

        # Create or fetch resizable datasets (skip creation in read-only mode)
        if natoms <= 0:
            natoms = self.file['atomic_numbers'].shape[0] if 'atomic_numbers' in self.file else 0
        self._init_dataset('positions', (0, natoms, 3), dtype=self.dtype_str)
        self._init_dataset('velocities', (0, natoms, 3), dtype=self.dtype_str)
        self._init_dataset('forces', (0, natoms, 3), dtype=self.dtype_str)
        self._init_dataset('energies', (0,), dtype='f8') # Energy always kept at f8 for precision
        self._init_dataset('cells', (0, 3, 3), dtype=self.dtype_str)
        self._init_dataset('steps', (0,), dtype='i8')  # per-frame step number for accurate resume

        # Legacy compat: for files created before 'steps' dataset existed,
        # backfill -1 sentinels BEFORE _recover_append_state so it sees a
        # consistent state.  Only runs in write/append mode.
        if self.append_only and 'steps' in self.file:
            main_keys = ['positions', 'velocities', 'forces', 'energies', 'cells']
            main_lens = [self.file[k].shape[0] for k in main_keys if k in self.file]
            if main_lens:
                n_main = main_lens[0]
                steps_len = self.file['steps'].shape[0]
                if steps_len < n_main:
                    self.file['steps'].resize(n_main, axis=0)
                    self.file['steps'][steps_len:] = -1
                elif steps_len > n_main:
                    self.file['steps'].resize(n_main, axis=0)

        if self.append_only:
            self._recover_append_state()

        # Async Setup: Producer-Consumer via Threading (write modes only)
        if not self._readonly:
            self.queue = queue.Queue(maxsize=max_queue_size)
            self.stop_event = threading.Event()
            self.error_event = threading.Event()
            self.fatal_error: Optional[Exception] = None

            # Start thread AFTER datasets are initialized to avoid h5py thread-safety issues
            self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self.writer_thread.start()

    def _open_h5_file(self, filename: str, mode: str):
        """Open the HDF5 file with multi-tier crash recovery.

        Tier 1: normal open
        Tier 2: locking=False (stale file locks from a crash)
        Tier 3: h5clear -s to clear stale file-consistency flags, then locking=False
        Tier 4: libver="latest" (SWMR / "already open" errors)
        Tier 5: quarantine broken file + recreate (write modes only)
        """
        # Tier 1 — normal open
        try:
            return h5py.File(filename, mode)
        except OSError:
            pass
        except Exception:
            pass

        # Tier 2 — lock-free open (handles stale file locks)
        try:
            return h5py.File(filename, mode, locking=False)
        except OSError:
            pass
        except Exception:
            pass

        # Tier 3 — clear stale file consistency flags left by a crash
        try:
            import subprocess
            subprocess.run(
                ["h5clear", "-s", filename],
                capture_output=True, timeout=10,
            )
            return h5py.File(filename, mode, locking=False)
        except OSError:
            pass
        except Exception:
            pass

        # Tier 4 — retry with libver="latest" for SWMR / "already open" issues
        try:
            return h5py.File(filename, mode, libver="latest")
        except OSError as exc:
            msg = str(exc)
            if "already open" not in msg and "SWMR write" not in msg:
                raise
        except Exception:
            pass

        # Tier 5 — quarantine and start fresh (write modes only)
        if self.recover_on_open_error and mode in ("a", "r+"):
            broken_name = self._quarantine_broken_file(filename)
            logger.warning(
                f"Quarantined stale HDF5 file to {broken_name}; creating a fresh trajectory."
            )
            return h5py.File(filename, "w", libver="latest")

        raise OSError(f"Failed to open HDF5 file: {filename}")

    def _quarantine_broken_file(self, filename: str) -> str:
        """Move a problematic file out of the way before recreating it."""
        if not os.path.exists(filename):
            return filename
        broken_name = f"{filename}.broken.{time.strftime('%Y%m%d-%H%M%S')}.{os.getpid()}"
        os.replace(filename, broken_name)
        return broken_name

    def _init_append_state(self):
        """Initialize append-only state metadata."""
        if "append_committed_frames" not in self.file.attrs:
            self.file.attrs["append_committed_frames"] = 0

    def _recover_append_state(self):
        """Truncate any partially written tail after a crash."""
        dataset_names = ["positions", "velocities", "forces", "energies", "cells", "steps"]
        present_lengths = [self.file[name].shape[0] for name in dataset_names if name in self.file]
        stored_committed = self.file.attrs.get("append_committed_frames", None)
        if stored_committed is None:
            committed = present_lengths[0] if present_lengths and len(set(present_lengths)) == 1 else 0
        else:
            committed = int(stored_committed)
            if present_lengths:
                committed = min(committed, min(present_lengths))

        if present_lengths:
            current_max = max(present_lengths)
            if current_max > committed:
                logger.warning(
                    f"Recovering append-only trajectory {self.filename}: "
                    f"truncating from {current_max} to {committed} committed frames."
                )
                if self.file.mode != "r":
                    for name in dataset_names:
                        if name in self.file and self.file[name].shape[0] > committed:
                            self.file[name].resize(committed, axis=0)
        self._committed_frames = committed
        if self.file.mode != "r":
            self.file.attrs["append_committed_frames"] = committed
            self.file.flush()

    def _init_dataset(self, name, shape, dtype):
        if name not in self.file:
            if self._readonly:
                return  # read-only: rely on existing datasets
            # Chunking by 1 frame to keep IO predictable for huge frames
            chunks = (1, *shape[1:]) if shape[0] == 0 else None
            self.file.create_dataset(name, shape=shape, maxshape=(None, *shape[1:]),
                                   dtype=dtype, chunks=chunks)

    def _commit_frame(self, frame_index: int):
        """Record the latest fully written frame."""
        self._committed_frames = frame_index + 1
        if self.append_only:
            self.file.attrs["append_committed_frames"] = self._committed_frames
        self.file.flush()

    def write(self, positions, velocities, cell, energy: float, forces,
              step: int = None, info: Optional[Dict[str, float]] = None):
        """Asynchronously queue GPU data for writing. Non-blocking.

        Args:
            positions: GPU tensor (N, 3).
            velocities: GPU tensor (N, 3).
            cell: GPU tensor (3, 3).
            energy: Potential energy in eV.
            forces: GPU tensor (N, 3).
            step: Current simulation step number, stored per-frame for
                  accurate resume offset calculation.
            info: optional dict of per-frame SCALAR metadata (ASE Atoms.info
                  style), e.g. QTB diagnostics {"ke_per_H_eV": ..., "adqtb_fdt_residual": ...}.
                  Each key becomes a resizable scalar dataset under the "info/" group,
                  created lazily on first appearance and backfilled with NaN for
                  earlier frames so all info series stay frame-aligned. Values must be
                  float-castable scalars. Does not touch the core datasets, so runs
                  without info (e.g. Langevin) produce identical files to before.
        """
        if self._readonly:
            raise RuntimeError("Cannot write to a read-only H5Trajectory.")
        if hasattr(self, 'error_event') and self.error_event.is_set():
            raise RuntimeError(f"H5 Background Writer failed: {self.fatal_error}")

        # 1. Start Async GPU-to-CPU copy (Non-blocking on GPU stream)
        pos_t = positions.to(dtype=self.torch_dtype, device='cpu', non_blocking=True)
        vel_t = velocities.to(dtype=self.torch_dtype, device='cpu', non_blocking=True)
        cell_t = cell.to(dtype=self.torch_dtype, device='cpu', non_blocking=True)

        if isinstance(forces, torch.Tensor):
            forces_t = forces.to(dtype=self.torch_dtype, device='cpu', non_blocking=True)
        else:
            forces_t = torch.from_numpy(forces).to(dtype=self.torch_dtype, device='cpu', non_blocking=True)

        # 1b. Synchronize: ensure async copies finish before the tensors are
        #    passed to the background thread.  Without this, a subsequent
        #    MD step may overwrite the GPU buffers while DMA is still in
        #    flight, causing corrupted ("collapsed") positions in the H5.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 2. Package and Put into Queue
        frame_data = {
            'positions': pos_t,
            'velocities': vel_t,
            'forces': forces_t,
            'energies': energy,
            'cells': cell_t,
            'step': step,
            'info': {str(k): float(v) for k, v in info.items()} if info else None,
        }
        
        # Check backpressure
        if self.queue.full():
            logger.warning("H5 Trajectory queue full (Disk I/O bottleneck). Blocking simulation...")
            
        self.queue.put(frame_data)

    def _write_info(self, frame_index: int, info: Dict[str, float]):
        """Write per-frame scalar info values into resizable datasets under "info/".

        Each key gets its own f8 dataset of shape (nframes,). A key first seen at
        frame_index>0 is backfilled with NaN for all prior frames so every info
        series is aligned to the trajectory frame index. Runs in the writer thread only.
        """
        grp = self.file.require_group("info")
        for key, val in info.items():
            if key not in grp:
                ds = grp.create_dataset(key, shape=(0,), maxshape=(None,),
                                        dtype='f8', chunks=(1024,))
            else:
                ds = grp[key]
            # backfill (NaN) any frames between the dataset's end and this frame
            if ds.shape[0] < frame_index:
                old = ds.shape[0]
                ds.resize(frame_index, axis=0)
                ds[old:frame_index] = np.nan
            ds.resize(frame_index + 1, axis=0)
            ds[frame_index] = val

    def _writer_loop(self):
        """Background thread that consumes the queue and performs blocking HDF5 I/O."""
        while True:
            try:
                # Use a timeout to occasionally check the stop_event
                data = self.queue.get(timeout=1.0)
                if data is None: # Sentinel for shutdown
                    self.queue.task_done()
                    break
                
                frame_index = self._committed_frames
                # numpy() call here effectively synchronizes the data transfer
                for name in ['positions', 'velocities', 'forces', 'cells']:
                    ds = self.file[name]
                    new_size = frame_index + 1
                    ds.resize(new_size, axis=0)
                    ds[frame_index] = data[name].numpy()
                
                # Scalar datasets: energy + step
                ds_e = self.file['energies']
                ds_e.resize(frame_index + 1, axis=0)
                ds_e[frame_index] = data['energies']

                ds_s = self.file['steps']
                ds_s.resize(frame_index + 1, axis=0)
                ds_s[frame_index] = data['step'] if data.get('step') is not None else -1

                # Optional per-frame info scalars (ASE Atoms.info style) under "info/".
                if data.get('info'):
                    self._write_info(frame_index, data['info'])

                # Crash-safe two-phase commit:
                #  1st flush → data + B-tree on disk
                #  2nd flush → commit marker on disk
                # A crash between the two loses at most one frame (which will
                # be truncated by _recover_append_state), never corrupts data.
                self.file.flush()
                if self.append_only:
                    self.file.attrs["append_committed_frames"] = frame_index + 1
                    self.file.flush()
                self._committed_frames = frame_index + 1
                self.queue.task_done()
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue
            except Exception as e:
                logger.error(f"H5 Background Writer Error: {e}")
                self.fatal_error = e
                self.error_event.set()
                # Drain queue to unblock potential puts
                while not self.queue.empty():
                    try:
                        self.queue.get_nowait()
                        self.queue.task_done()
                    except queue.Empty:
                        break
                break

    def close(self):
        """Perform a clean shutdown of the writer thread and close the file."""
        if self._closed:
            return
        self._closed = True

        # Shut down background writer thread (exists only in write/append mode).
        if not self._readonly and hasattr(self, 'writer_thread') and self.writer_thread.is_alive():
            try:
                self.queue.put(None, timeout=2.0)
                self.writer_thread.join(timeout=10.0)
            except (queue.Full, Exception):
                pass

        if hasattr(self, 'file'):
            try:
                if self.append_only and not self._readonly:
                    self.file.attrs["append_committed_frames"] = self._committed_frames
                self.file.flush()
                try:
                    vfd = self.file.id.get_vfd_handle()
                    if isinstance(vfd, int) and vfd >= 0:
                        os.fsync(vfd)
                except Exception:
                    pass
                self.file.close()
            except Exception:
                pass

    @staticmethod
    def truncate_to_step(filepath: str, target_step: int,
                        full_sync_interval: int = None) -> int:
        """Truncate an existing H5 trajectory to *target_step* (inclusive).

        Finds the largest frame whose step ≤ *target_step*, resizes every
        per-frame dataset to that index + 1, and updates the commit marker.

        When the trajectory has a ``steps`` dataset (new format) the per-frame
        step values are used directly.  For legacy files without ``steps``,
        step is inferred as ``frame_index * full_sync_interval`` — *full_sync_interval*
        must be provided in that case.

        Returns:
            Number of frames retained after truncation.
        """
        dataset_names = ["positions", "velocities", "forces", "energies", "cells", "steps"]

        with h5py.File(filepath, "r+") as f:
            n_frames = f["positions"].shape[0]
            if n_frames == 0:
                return 0

            if "steps" in f and f["steps"].shape[0] == n_frames:
                steps = f["steps"][:]
                # Use real step values; ignore sentinel -1 entries.
                mask = (steps <= target_step) & (steps >= 0)
                if not mask.any():
                    raise ValueError(
                        f"No frame found with step <= {target_step} in {filepath}"
                    )
                keep_idx = int(np.flatnonzero(mask)[-1])
            elif full_sync_interval is not None:
                # Legacy file: infer step = frame_index * full_sync_interval.
                # Frame 0 = initial state (step 0), frame k = step k * full_sync_interval.
                # Find the largest frame index where k * full_sync_interval <= target_step.
                keep_idx = target_step // full_sync_interval
                if keep_idx >= n_frames:
                    keep_idx = n_frames - 1
                if keep_idx < 0:
                    raise ValueError(
                        f"target_step={target_step} too small for "
                        f"full_sync_interval={full_sync_interval} in {filepath}"
                    )
            else:
                raise ValueError(
                    f"Trajectory {filepath} has no 'steps' dataset and no "
                    f"full_sync_interval was provided. Cannot truncate by step."
                )
            new_len = keep_idx + 1

            for name in dataset_names:
                if name in f and f[name].shape[0] > new_len:
                    f[name].resize(new_len, axis=0)

            f.attrs["append_committed_frames"] = new_len
            f.flush()
            return new_len

    @staticmethod
    def read_last_frame(filepath: str) -> Dict[str, Any]:
        """Read the last frame of an H5 trajectory and return a self-contained dict.

        Opens the file read-only with crash recovery.  ``cell``, ``pbc``, and
        ``numbers`` are guaranteed non-None (defaults are filled when the H5
        lacks the dataset).  ``positions``, ``velocities``, and ``steps`` may
        be None for empty files or legacy files without those datasets.

        Returns:
            Dict with keys: ``positions``, ``velocities``, ``cell``,
            ``numbers``, ``pbc``, ``steps``, ``n_frames``.
        """
        # --- crash-recovery file open (read-only) ---
        f = None
        for opener in (
            lambda: h5py.File(filepath, "r"),
            lambda: h5py.File(filepath, "r", locking=False),
            lambda: (
                __import__("subprocess").run(
                    ["h5clear", "-s", filepath],
                    capture_output=True, timeout=10,
                ),
                h5py.File(filepath, "r", locking=False),
            )[1],
        ):
            try:
                f = opener()
                if f is not None:
                    break
            except OSError:
                continue
            except Exception:
                continue
        if f is None:
            raise OSError(f"Failed to open HDF5 file: {filepath}")

        with f:
            n_frames = f["positions"].shape[0] if "positions" in f else 0

            positions = f["positions"][-1] if n_frames > 0 else None
            velocities = f["velocities"][-1] if "velocities" in f and n_frames > 0 else None

            # Self-contained metadata: always present or filled with defaults.
            cell = np.zeros((3, 3), dtype=np.float64)
            if "cells" in f and f["cells"].shape[0] == n_frames and n_frames > 0:
                cell = f["cells"][-1].astype(np.float64)

            numbers = np.array([], dtype=int)
            if "atomic_numbers" in f:
                numbers = f["atomic_numbers"][:]

            pbc = np.array([False, False, False])
            if "pbc" in f:
                pbc_val = f["pbc"][:]
                if pbc_val.dtype != bool:
                    pbc_val = pbc_val.astype(bool)
                pbc = pbc_val

            steps = None
            if "steps" in f and f["steps"].shape[0] == n_frames and n_frames > 0:
                raw = f["steps"][-1]
                try:
                    steps = int(raw)
                except (TypeError, ValueError):
                    steps = -1
                if steps < 0:
                    steps = None

        return {
            "positions": positions,
            "velocities": velocities,
            "cell": cell,
            "numbers": numbers,
            "pbc": pbc,
            "steps": steps,
            "n_frames": n_frames,
        }

    def __len__(self):
        if self.append_only:
            return int(self._committed_frames)
        return self.file['positions'].shape[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def to_ase_atoms(self, index: int) -> "Atoms":
        """Convert a specific frame from HDF5 to a standard ASE Atoms object."""
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator
        
        if index < 0:
            index = len(self) + index
            
        if index < 0 or index >= len(self):
            raise IndexError("Trajectory index out of range")

        # Load data, converting from FP32 back to FP64 for ASE compatibility
        numbers = self.file['atomic_numbers'][:]
        positions = self.file['positions'][index].astype(np.float64)
        cell = self.file['cells'][index].astype(np.float64)
        pbc = self.file['pbc'][:]
        
        atoms = Atoms(
            numbers=numbers,
            positions=positions,
            cell=cell,
            pbc=pbc
        )
        
        if 'velocities' in self.file:
            atoms.set_velocities(self.file['velocities'][index].astype(np.float64))
            
        results = {}
        if 'energies' in self.file:
            results['energy'] = self.file['energies'][index]
        if 'forces' in self.file:
            results['forces'] = self.file['forces'][index].astype(np.float64)
            
        if results:
            atoms.calc = SinglePointCalculator(atoms, **results)

        # Populate ASE Atoms.info from the optional "info/" group (per-frame scalars,
        # e.g. QTB diagnostics). Absent in older trajectories -> info stays empty,
        # so this is fully backward compatible.
        if 'info' in self.file:
            grp = self.file['info']
            for key in grp.keys():
                ds = grp[key]
                if index < ds.shape[0]:
                    atoms.info[key] = float(ds[index])

        return atoms

    @staticmethod
    def read_info(filepath: str) -> Dict[str, np.ndarray]:
        """Read all per-frame info series from a trajectory's "info/" group.

        Returns {key: (nframes,) float array}. Empty dict if the file has no info
        group (older trajectories) — callers must tolerate missing keys.
        """
        out: Dict[str, np.ndarray] = {}
        with h5py.File(filepath, "r") as f:
            if "info" in f:
                for key in f["info"].keys():
                    out[key] = f["info"][key][:]
        return out
