# Description: CLI entry point for GPU-native ASE simulations.
#
# Usage:
#   python src/molfm/interface/ase/cli.py --help
#   python src/molfm/interface/ase/cli.py --input_path water.xyz \
#       --checkpoint ckpt/model.pt --config ckpt/config.yaml --head_name omol25 \
#       --task md --steps 1000 --temp 300 --dt 0.5

import os
import sys

# These must run before any torch / h5py import.
#
# 1) HDF5 file locking off → safe parallel reads of trajectory files on shared FS.
# 2) PyTorch CUDA allocator → expandable_segments coalesces freed blocks and grows
#    contiguous regions on demand, reducing the "reserved-but-unallocated" gap
#    that causes mid-MD OOM on systems that should otherwise fit. Set only when
#    the user hasn't already configured PYTORCH_CUDA_ALLOC_CONF themselves.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Make `molfm` importable when this file is executed directly as a script
# (src/molfm/interface/ase/cli.py → src/). Same bootstrap as
# src/molfm/tasks/pretrain_smallmol.py.
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from typing import Optional, Union, Dict, Any, List
from fire import Fire
from loguru import logger

from molfm.interface.ase.torch_ext.ase_runner import ASERunner

def _validate_run_params(cfg: Dict[str, Any]):
    """Validate the input parameters for the run function."""
    task = cfg['task']
    if task not in ["relax", "md", "both"]:
        raise ValueError(f"Invalid task: {task}. Must be 'relax', 'md', or 'both'.")

    if not cfg['input_path'] and (cfg['pos'] is None or cfg['numbers'] is None):
        raise ValueError("Must provide either 'input_path' or both 'pos' and 'numbers'.")

    if cfg['input_path'] and not os.path.exists(cfg['input_path']):
        raise FileNotFoundError(f"Input path not found: {cfg['input_path']}")

    if cfg['checkpoint'] != "none" and not os.path.exists(cfg['checkpoint']):
        raise FileNotFoundError(f"Checkpoint not found: {cfg['checkpoint']}")

    if cfg['steps'] <= 0 and task in ["md", "both"]:
        raise ValueError(f"MD steps must be positive, got {cfg['steps']}")

    if cfg['relax_steps'] <= 0 and task in ["relax", "both"]:
        raise ValueError(f"Relax steps must be positive, got {cfg['relax_steps']}")

    if cfg['dt'] <= 0:
        raise ValueError(f"Timestep dt must be positive, got {cfg['dt']}")

    if cfg['temp'] < 0:
        raise ValueError(f"Temperature cannot be negative, got {cfg['temp']}")

    if str(cfg['ensemble']).lower() not in ("nvt", "npt", "nve"):
        raise ValueError(f"ensemble must be 'nvt', 'npt', or 'nve', got {cfg['ensemble']}")
    if str(cfg['ensemble']).lower() == "npt" and cfg['barostat_interval'] <= 0:
        raise ValueError(f"barostat_interval must be positive, got {cfg['barostat_interval']}")

    if cfg['full_sync_interval'] <= 0 or cfg['log_interval'] <= 0:
        raise ValueError("Intervals must be positive integers.")

    budget = cfg['recompute_budget']
    if budget > 1.0 or (budget < 0.0 and budget != -1.0):
        raise ValueError(
            f"recompute_budget must be -1.0 (partitioner default) or in [0, 1], got {budget}"
        )
    if budget >= 0.0 and not cfg['use_compile']:
        logger.warning(
            f"recompute_budget={budget} has no effect with use_compile=False — "
            "the activation-memory budget is a torch.compile partitioner setting."
        )

    if str(cfg['thermostat']).lower() not in ("langevin", "qtb"):
        raise ValueError(f"thermostat must be 'langevin' or 'qtb', got {cfg['thermostat']}")

    if str(cfg['ensemble']).lower() == "nve" and str(cfg['thermostat']).lower() == "qtb":
        raise ValueError("ensemble='nve' is incompatible with thermostat='qtb' "
                         "(QTB is a thermostat). Use thermostat='langevin' for NVE.")

def _print_config_summary(cfg: Dict[str, Any], runner: ASERunner):
    """Print a detailed configuration summary to the console."""
    task = cfg['task']
    logger.info("=" * 40)
    logger.info("AseRunner Configuration:")
    logger.info(f"  Task:           {task}")
    logger.info(f"  Input:          {cfg['input_path'] if cfg['input_path'] else 'numpy/list input'}")
    logger.info(f"  Checkpoint:     {cfg['checkpoint']}")
    logger.info(f"  Device:         {cfg['device']}")
    logger.info(f"  TF32:           {cfg['use_tf32']}")
    logger.info(f"  FAISS:          {cfg['use_faiss']}")
    logger.info(f"  Compile:        {cfg['use_compile']}")
    logger.info(f"  Recompute:      "
                f"{'default' if cfg['recompute_budget'] < 0.0 else cfg['recompute_budget']}")
    logger.info(f"  CUDA alloc:     {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<default>')}")
    logger.info(f"  Relax Traj:     {runner.relax_path}")
    logger.info(f"  MD Traj:        {runner.md_path}")
    logger.info(f"  Work Dir:       {cfg['work_dir']}")
    if task in ["relax", "both"]:
        logger.info(f"  Relax Steps:    {cfg['relax_steps']}")
        logger.info(f"  Fmax:           {cfg['fmax']} eV/A")
        logger.info(f"  Maxstep:        {cfg['relax_maxstep']} A")
        logger.info(f"  Damping:        {cfg['relax_damping']}")
    if task in ["md", "both"]:
        logger.info(f"  MD Steps:       {cfg['steps']}")
        logger.info(f"  Temperature:    {cfg['temp']} K")
        logger.info(f"  Timestep:       {cfg['dt']} fs")
        logger.info(f"  Friction:       {cfg['friction'] if cfg['friction'] else 'auto'} fs^-1")
        logger.info(f"  FixCOM:         {cfg['fixcm']}")
        logger.info(f"  Seed:           {cfg['seed']}")
        logger.info(f"  Ensemble:       {cfg['ensemble']}")
        if str(cfg['ensemble']).lower() == "npt":
            logger.info(f"  Pressure:       {cfg['pressure_bar']} bar")
            logger.info(f"  Barostat:       interval={cfg['barostat_interval']} steps, "
                        f"dlnV={cfg['barostat_dlnV']}, molecules={cfg['molecule_source']}")
    logger.info(f"  Full Sync Interval:    {cfg['full_sync_interval']}")
    logger.info(f"  Log Interval:   {cfg['log_interval']}")
    logger.info(f"  Format:         HDF5 (FP32={cfg['save_in_fp32']}, Async=True)")
    logger.info(f"  Resume:         {cfg['resume']}")
    if cfg.get("resume_step") is not None:
        logger.info(f"  Resume Step:    {cfg['resume_step']}")
    logger.info("=" * 40)

def run(
    input_path: Optional[str] = None,
    checkpoint: str = "none",
    config: Optional[str] = None,
    head_name: Optional[str] = "omol25",
    task: str = "md", 
    steps: int = 1000,
    relax_steps: int = 100,
    fmax: float = 0.05,
    relax_maxstep: float = 0.2,
    relax_damping: float = 1.0,
    temp: float = 300.0,
    dt: float = 0.5,
    friction: Optional[float] = None,
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
    full_sync_interval: int = 10,
    log_interval: int = 10,
    device: str = "cuda:0",
    work_dir: str = "runs",
    name: str = "mfm_job",
    use_swanlab: bool = False,
    resume: bool = True,
    use_faiss: bool = True,
    use_tf32: bool = False,
    use_compile: bool = True,
    recompute_budget: float = -1.0,
    fixcm: bool = True,
    save_in_fp32: bool = False,
    resume_step: Optional[int] = None,
    charge: int = 0,
    multiplicity: int = 1,
    seed: int = 42,
    # Numpy-style inputs via fire
    pos: Optional[Union[List[List[float]], str]] = None,
    numbers: Optional[Union[List[int], str]] = None,
    cell: Optional[Union[List[List[float]], str]] = None,
):
    """
    Unified entry point for GPU-native ASE simulations.
    
    This function can be called directly from the command line using Python Fire.
    It supports multiple input formats and simulation tasks.

    Args:
        input_path: Path to geometry file (.xyz, .gro, .pdb).
        checkpoint: Path to E2Former model checkpoint.
        config: Path to model config yaml.
        head_name: Name of the E2Former head to use, can be svp or omol25.
        task: Simulation task: 'relax' (GO), 'md', or 'both'.
        steps: Total MD steps.
        relax_steps: Max GO steps.
        fmax: Force threshold for GO (eV/A).
        relax_maxstep: Max atomic displacement per L-BFGS step (Å).
        relax_damping: L-BFGS step-size scaling (<1.0 = conservative).
        temp: Target temperature for MD (K).
        dt: Time step for MD (fs).
        friction: Langevin friction (fs^-1). If None, auto-calculated.
        ensemble: 'nvt' (default, fixed box, Langevin thermostat), 'npt' (energy-only
            MC barostat), or 'nve' (deterministic velocity-Verlet, energy-conserving).
            NVE forces friction=0 (TorchLangevin then reduces exactly to velocity-
            Verlet) and cannot be combined with thermostat='qtb'. Initial velocities
            are still drawn from a Maxwell-Boltzmann distribution at --temp.
        pressure_bar: Target pressure for NPT (bar). Default 1.0.
        barostat_interval: MD steps between MC volume-move attempts (NPT).
        barostat_dlnV: Max ln-volume step per MC move; tune for ~30-50% accept.
        molecule_source: 'auto'/'bondgraph' (distance bond graph), 'stride:N'
            (every N atoms = 1 molecule, e.g. 'stride:3' for O,H,H water), or
            'residue' (unavailable on the GPU path; falls back to bondgraph).
        barostat_seed: RNG seed for the MC barostat proposals.
        thermostat: 'langevin' (default, white noise) or 'qtb' (Quantum Thermal
            Bath: colored noise injecting zero-point energy for nuclear quantum
            effects). QTB needs only energy+forces and composes with the MC
            barostat.
        qtb_adaptive: Enable adQTB-r FDT correction for zero-point-energy leakage
            (default True; recommended for production QTB runs).
        qtb_classical: If True, QTB uses the classical (kT) spectrum -> reduces to
            Langevin (hbar->0 regression check). Default False.
        qtb_n_seg: Colored-noise / adQTB segment length in MD steps (default 4096).
        qtb_adqtb_lr: adQTB friction-adaptation learning rate (default 0.005).
        full_sync_interval: Steps between full-state syncs (wrap + fixcom + save).
        log_interval: Steps between log entries.
        device: CUDA device string.
        work_dir: Output directory.
        name: Experiment name.
        use_swanlab: Enable SwanLab tracking. Off by default; the target workspace
            is taken from the ``swanlab_workspace`` environment variable (falling
            back to the account's default workspace when unset).
        resume: Enable resume capability. When a trajectory for this
            ``work_dir``/``name`` already exists, the run continues from its last
            frame and the geometry input is ignored — the input is still read to
            verify it describes the same system (same atomic numbers), and the
            run aborts on a mismatch. Pass ``--resume False`` (or a fresh
            ``--name``) to start over.
        use_faiss: Use FAISS for neighbor search.
        use_tf32: Use TF32 precision for matmuls.
        use_compile: Use torch.compile() for model acceleration.
        recompute_budget: Activation-memory budget in [0, 1] for torch.compile's
            min-cut partitioner — the fraction of activations kept in memory, the
            remainder recomputed in the backward pass. Lower trades compute for
            memory, which is what makes the largest systems fit. -1.0 (default)
            leaves the partitioner alone. Only meaningful with use_compile=True.
        fixcm: Apply center-of-mass correction.
        save_in_fp32: Store trajectory in float32 to save space.
        resume_step: Truncate trajectory to this step before resuming.
            Frames after this step are discarded.  Useful after a run
            produced bad frames (e.g. NaN forces) and you want to
            restart from a known-good checkpoint.
        seed: Random seed for MD. Controls (1) the Maxwell-Boltzmann initial
            velocity sampling at the start of run_md, and (2) the Langevin
            thermostat noise stream, so a run is reproducible for a fixed seed.
            Default 42 preserves prior behavior; change it to draw a different
            velocity realization.
        pos: Positions, given as one of:
            - ndarray / list of shape (N, 3) in Angstrom
            - path to a .npy file (shape (N, 3), loaded via mmap_mode="r")
            Only consulted when ``input_path`` is not provided.
        numbers: Atomic numbers, given as one of:
            - ndarray / list of shape (N,)
            - path to a .npy file (shape (N,), loaded via mmap_mode="r")
            Only consulted when ``input_path`` is not provided.
        cell: Box vectors, given as one of:
            - ndarray / list of shape (3, 3) in Angstrom
            - path to a .npy file (shape (3, 3), loaded via mmap_mode="r")
            - None (defaults to 10x10x10 Å cube)
            Only consulted when ``input_path`` is not provided.
        Note: passing .npy paths still goes through ASE's ``Atoms()`` constructor,
        which materializes the mmap array to RAM. The mmap mode therefore reduces
        the peak memory during loading, not the steady-state footprint.
    """
    # Collect full config for logging and validation
    full_config = locals().copy()
    _validate_run_params(full_config)

    # ----------------------------
    runner = ASERunner(
        input_path=input_path,
        positions=pos,
        numbers=numbers,
        cell=cell,
        checkpoint_path=checkpoint,
        config_path=config,
        head_name=head_name,
        device=device,
        use_faiss=use_faiss,
        use_tf32=use_tf32,
        use_compile=use_compile,
        recompute_budget=recompute_budget,
        work_dir=work_dir,
        name=name,
        use_swanlab=use_swanlab,
        full_sync_interval=full_sync_interval,
        log_interval=log_interval,
        fixcm=fixcm,
        save_in_fp32=save_in_fp32,
        resume_step=resume_step,
        charge=charge,
        multiplicity=multiplicity,
        seed=seed,
        ensemble=ensemble,
        pressure_bar=pressure_bar,
        barostat_interval=barostat_interval,
        barostat_dlnV=barostat_dlnV,
        molecule_source=molecule_source,
        barostat_seed=barostat_seed,
        thermostat=thermostat,
        qtb_adaptive=qtb_adaptive,
        qtb_classical=qtb_classical,
        qtb_n_seg=qtb_n_seg,
        qtb_adqtb_lr=qtb_adqtb_lr,
        swanlab_config=full_config
    )

    _print_config_summary(full_config, runner)

    # Stage 1: Structure Relaxation (GO)
    if task in ["relax", "both"]:
        runner.relax(fmax=fmax, steps=relax_steps, do_resume=resume,
                     maxstep=relax_maxstep, damping=relax_damping)
    
    # Stage 2: Molecular Dynamics (MD)
    if task in ["md", "both"]:
        runner.run_md(
            steps=steps, 
            timestep_fs=dt, 
            temperature_K=temp, 
            friction=friction,
            do_resume=resume
        )

if __name__ == "__main__":
    Fire(run)
