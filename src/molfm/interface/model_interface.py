# -*- coding: utf-8 -*-
import gc
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from loguru import logger

from molfm.data.utils import get_data_defult_config
from molfm.tasks.pretrain_smallmol import Identity, MolfmAtomic, SmallMolConfig

torch.serialization.add_safe_globals([slice])


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def maybe_gc_cuda(device, gc_reserved_ratio_threshold=0.98, log=False):
    """Release cached CUDA blocks only when the device is nearly full.

    Long MD trajectories run thousands of forward passes with slightly different
    neighbour-list shapes, which fragments the caching allocator: memory is
    reserved-but-unallocated and a later allocation OOMs on a system that should
    still fit. Collecting unconditionally would serialize the whole step on
    ``empty_cache()``, so this only fires once actual device usage crosses
    *gc_reserved_ratio_threshold*.

    Returns True when a collection actually ran.
    """
    if not torch.cuda.is_available():
        return False
    device = torch.device(device)
    if device.type != "cuda":
        return False

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    used_bytes = total_bytes - free_bytes
    allocated_bytes = torch.cuda.memory_allocated(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    used_ratio = used_bytes / float(total_bytes)

    def _log(prefix: str):
        if log:
            logger.info(
                f"{prefix} {device}: "
                f"used={used_ratio:.3f} ({_format_gib(used_bytes)}/{_format_gib(total_bytes)}), "
                f"alloc={_format_gib(allocated_bytes)}, reserved={_format_gib(reserved_bytes)}"
            )

    if used_ratio < gc_reserved_ratio_threshold:
        return False

    _log("CUDA GC run")
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
    return True


class E2FormerModelInterface:
    """Shared model lifecycle for ASE and GROMACS integrations."""

    _torch_initialized = False

    __fields__ = [
        "config_path",
        "config_file",
        "checkpoint_path",
        "head_name",
        "device",
        "use_compile",
        "use_tf32",
        "use_faiss",
        "recompute_budget",
        "model_cfg",
        "model",
        "ref_energy",
        "unit",
    ]

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        head_name: Optional[str] = None,
        device: str = "cuda:0",
        use_compile: bool = False,
        use_tf32: bool = False,
        use_faiss: bool = True,
        recompute_budget: float = -1.0,
        auto_setup: bool = True,
        **_kwargs,
    ):
        self.use_compile = use_compile
        self.use_tf32 = use_tf32
        self.use_faiss = use_faiss
        # Activation-memory budget for torch.compile's partitioner, in [0, 1]:
        # the fraction of activations to keep in memory, the rest recomputed in
        # the backward pass. -1.0 leaves the partitioner at its default (no
        # recomputation trade). Only meaningful with use_compile=True.
        self.recompute_budget = recompute_budget
        self.config_path = config_path
        self.config_name = config_name
        self.checkpoint_path = checkpoint_path
        self.head_name = head_name
        self.device = device

        self.model_cfg = None
        self.model = None
        self.ref_energy = None
        self.unit = None
        self.max_radius = None
        self.config_file = None
        # Default to FAISS-backed neighbor search for large solvated systems.

        if auto_setup:
            self.setup_model_interface()

    def setup_model_interface(self):
        self._validate_init_args()
        self._load_config_and_metadata()
        self._setup_model()

    def _validate_init_args(self):
        required_fields = {
            "head_name": self.head_name,
        }
        missing_fields = [name for name, value in required_fields.items() if value is None]
        if missing_fields:
            raise ValueError(f"Missing model interface init args: {', '.join(missing_fields)}")

        # checkpoint_path=None / "none" builds the model with randomly initialized
        # weights. That is only useful for smoke-testing the plumbing (shapes,
        # neighbour search, integrator) and requires an explicit config_path,
        # since there is no checkpoint directory to resolve the yaml from.
        if self.checkpoint_path in (None, "none"):
            if self.config_path is None and self.config_name is None:
                raise ValueError(
                    "checkpoint_path is None/'none' (random weights). Pass config_path "
                    "or config_name explicitly — there is no checkpoint directory to "
                    "resolve the config yaml from."
                )
            logger.warning(
                "No checkpoint given (checkpoint_path={}). The model will run with "
                "RANDOMLY INITIALIZED weights — predictions are meaningless. Use this "
                "only for smoke tests.",
                self.checkpoint_path,
            )
        elif not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path {self.checkpoint_path} does not exist.")
        self.config_file = self._resolve_unique_config_file()

    def _resolve_unique_config_file(self) -> str:
        config_dir = self.config_path
        if config_dir is None:
            if self.checkpoint_path in (None, "none"):
                raise ValueError(
                    "Cannot resolve the config yaml: no config_path and no checkpoint "
                    "directory to fall back on."
                )
            config_dir = os.path.dirname(os.path.abspath(self.checkpoint_path))
        else:
            config_dir = os.path.abspath(config_dir)
            # Accept a path to the yaml itself, not just its directory — the CLI
            # naturally takes `--config /path/to/config.yaml`. An explicit
            # config_name still wins, and is then resolved against the file's
            # directory.
            if os.path.isfile(config_dir):
                if self.config_name is None:
                    return config_dir
                config_dir = os.path.dirname(config_dir)
            if not os.path.isdir(config_dir):
                raise FileNotFoundError(f"Config path {config_dir} does not exist.")

        if self.config_name is not None:
            candidate = Path(self.config_name)
            if not candidate.is_absolute():
                candidate = Path(config_dir) / candidate
            if candidate.is_file():
                return str(candidate)
            raise FileNotFoundError(
                f"Config file {candidate} does not exist in {config_dir}."
            )

        yaml_files = sorted(
            [
                os.path.join(config_dir, name)
                for name in os.listdir(config_dir)
                if name.endswith((".yaml", ".yml"))
            ]
        )
        if len(yaml_files) != 1:
            raise ValueError(
                f"Expected exactly one yaml file in {config_dir}, found {len(yaml_files)}: {yaml_files}"
            )
        return yaml_files[0]

    def _load_config_and_metadata(self):
        with open(self.config_file, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
        self.model_cfg = SmallMolConfig(**raw_cfg)
        self.max_radius = self.model_cfg.backbone_config["max_radius"]

        (
            atom_reference,
            _system_ref,
            _train_ratio,
            _val_ratio,
            _test_ratio,
            _has_energy,
            _has_forces,
            _pbc,
            unit,
            _others,
        ) = get_data_defult_config(self.head_name)

        self.ref_energy = atom_reference * unit
        self.unit = unit

    def _setup_model(self):
        self.device = torch.device(self.device)
        if self.device.type == "cuda" and torch.cuda.is_available():
            if self.device.index is None:
                self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
            torch.cuda.set_device(self.device)
        self.model_cfg.backbone_config["use_faiss"] = self.use_faiss
        model = MolfmAtomic(self.model_cfg, loss_fn=Identity)
        if self.checkpoint_path not in (None, "none"):
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict["model"], strict=False)
        model.to(self.device)
        if self.use_tf32:
            model.configure_tf32(self.use_tf32)
        if self.use_compile:
            # Neighbour counts / cluster sizes reaching the model as plain ints
            # would otherwise be specialized on and trigger a recompile per
            # distinct value — a killer over a long MD trajectory.
            try:
                torch._dynamo.config.allow_unspec_int_on_nn_module = True
            except Exception as exc:
                logger.warning(f"Could not set allow_unspec_int_on_nn_module: {exc}")
            model.compile_model(recompute_budget=self.recompute_budget)

        model.eval()
        self.model = model
        checkpoint_name = os.path.basename(self.checkpoint_path) if self.checkpoint_path else ""
        config_name = os.path.basename(self.config_file) if self.config_file else ""
        logger.info(
            "Model ready | checkpoint={} | config={} | head={} | device={} | tf32={} | "
            "faiss={} | compile={} | recompute_budget={}",
            checkpoint_name,
            config_name,
            self.head_name,
            self.device,
            "on" if self.use_tf32 else "off",
            "on" if self.use_faiss else "off",
            "on" if self.use_compile else "off",
            "default" if self.recompute_budget < 0.0 else self.recompute_budget,
        )

    def dump_input(self, model_input: Dict[str, Any], filename: str):
        torch.save(model_input, filename)
        logger.info("Model input dumped to %s", filename)

    def predict(self, model_input: Dict[str, Any], need_force: bool = True) -> Dict[str, Any]:
        """Run the model forward.

        ``need_force`` is accepted for signature compatibility with callers that
        distinguish energy-only from energy+force evaluations (the MC barostat only
        needs energy). This backbone always produces forces — ``MolfmAtomic.forward``
        swallows the kwarg — so the flag currently has no effect on cost here.
        """
        return self.model(model_input, need_force=need_force)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
