# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from loguru import logger

from molfm.data.utils import get_data_defult_config
from molfm.tasks.pretrain_smallmol import Identity, MolfmAtomic, SmallMolConfig

torch.serialization.add_safe_globals([slice])


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
        auto_setup: bool = True,
        **_kwargs,
    ):
        self.use_compile = use_compile
        self.use_tf32 = use_tf32
        self.use_faiss = use_faiss
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
            "checkpoint_path": self.checkpoint_path,
            "head_name": self.head_name,
        }
        missing_fields = [name for name, value in required_fields.items() if value is None]
        if missing_fields:
            raise ValueError(f"Missing model interface init args: {', '.join(missing_fields)}")

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path {self.checkpoint_path} does not exist.")
        self.config_file = self._resolve_unique_config_file()

    def _resolve_unique_config_file(self) -> str:
        config_dir = self.config_path
        if config_dir is None:
            config_dir = os.path.dirname(os.path.abspath(self.checkpoint_path))
        else:
            config_dir = os.path.abspath(config_dir)
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
        self.model_cfg.backbone_config["use_faiss"] = self.use_faiss
        model = MolfmAtomic(self.model_cfg, loss_fn=Identity)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict["model"], strict=False)
        model.to(self.device)
        if self.use_tf32:
            model.configure_tf32(self.use_tf32)
        if self.use_compile:
            model.compile_model()

        model.eval()
        self.model = model
        checkpoint_name = os.path.basename(self.checkpoint_path) if self.checkpoint_path else ""
        config_name = os.path.basename(self.config_file) if self.config_file else ""
        logger.info(
            "Model ready | checkpoint={} | config={} | head={} | device={} | tf32={} | faiss={} | compile={}",
            checkpoint_name,
            config_name,
            self.head_name,
            self.device,
            "on" if self.use_tf32 else "off",
            "on" if self.use_faiss else "off",
            "on" if self.use_compile else "off",
        )

    def dump_input(self, model_input: Dict[str, Any], filename: str):
        torch.save(model_input, filename)
        logger.info("Model input dumped to %s", filename)

    def predict(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        return self.model(model_input)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
