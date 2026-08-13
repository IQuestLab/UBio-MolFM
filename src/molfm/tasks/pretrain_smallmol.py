# -*- coding: utf-8 -*-
"""This script is used to fine-tune the  model on small molecules dataset, such MD17, MD22, PubChem50K, etc.
"""
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List,Union

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import tqdm
from pathlib import Path
import wandb  # isort:skip
import swanlab  # isort:skip

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from torch.optim.adamw import AdamW,Adam


from loguru import logger 
from molfm.logging.loggers import MetricLogger,get_logger
from molfm.models.e2former.e2former_wrapper import E2FormerBackbone
from molfm.models.e2former.maceblocks import MACE_Wrapper

from molfm.utils.optimizer import get_scheduler

from molfm.pipeline.utils import ModelOutput
from molfm.pipeline.trainer import Trainer, seed_everything
from molfm.models.e2former.module_utils import build_param_groups
from molfm.utils.dist_utils import env_init 
from molfm.utils.torch_utils import simple_init_fn 

from molfm.pipeline.head import MDEnergyForceMultiHead
from torch.optim import Adam  # isort:skip

kcalmol_to_ev = 0.0433634
# torch.cuda.set_per_process_memory_fraction(0.5, 0)

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import random


class Identity(nn.Module):
    def __init__(
        self,
        args,
    ) -> None:
        super().__init__()
        self.args = args

    def forward(self, model_output, batched_data):
        return None, None


@dataclass
class DistributedTrainConfig():
    """
    TrainerConfig
    # DistributedTrainConfig is a combination of TrainerConfig and DistributedConfig.

    Args:
        local_rank (int, default: -1): The local rank of the process within a node.

        world_size (int, default: 1): The total number of processes involved in the distributed training.

        node_rank (int, default: 0): The rank of the current node within the cluster.

        rank (int, default: 0): The global rank of the process.

        pipeline_model_parallel_size (int, default: 0): The size of the pipeline model parallel group.

        tensor_model_parallel_size (int, default: 1): The size of the tensor model parallel group.

        deepspeed_config (str, default: ''): The path to the DeepSpeed configuration file.

        dist_backend (str, default: 'nccl'): The distributed backend to use for communication among processes.
    """

    local_rank: int = -1
    world_size: int = 1
    node_rank: int = 0
    rank: int = 0
    seed: int = 6666
    dist_backend: str = "nccl"

@dataclass
class SmallMolConfig(DistributedTrainConfig):
    backbone_config: Dict[str, Any] = MISSING
    scheduler_config: Dict[str, Any] = MISSING
    strategy_config: Dict[str, Any] = MISSING
    backbone: str = "e2former"
    head_cfg_str: str = ""
    
    data_path: str = ""
    data_path_list: str = ""
    data_path_list_valid: str = ""
    dataset_name_list: str = ""
    dataset_name_list_valid: str = ""
    dataset_sample_prob: str = ""
    dataset_micro_batch_size: str = ""
    train_val_test_split: List[float] = field(
        # default_factory=lambda: [0.2, 0.7, 0.1]
        default_factory=lambda: [0.95, 0.05, 0.0]
    )  # NOTE: This is only for MD data
    shuffle: bool = True
    save_dir: str = "./outputs"
    ckpt_path: str = "None"

    
    total_num_steps: int = 100000000
    total_num_epochs: int = 500
    save_batch_interval: int = 5000
    save_epoch_interval: int = 0
    log_interval: int = 50
    val_batch_interval: int = 1000
    val_epoch_interval: int = 0
    
    with_moe: str = "None"
    energy_loss_weight: float = 0.01
    force_loss_weight: float = 0.99
    loss_unit: str = "ev"
    loss_fn: str = "mae"
    # vsc_debug: bool = False

    lr: float = 0.0001
    custom_factors: str = ""
    clip_grad_norm: float = 5
    AutoGradForce: bool = True
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0
    
    #

    experiment_name: str = ""
    # wandb
    wandb: bool = False
    wandb_project: str = "uncategorized"
    # aim
    aim: bool = False
    aim_dir: str = "./aim_log"
    # tb
    tb: bool = False
    tb_dir: str = "./tb_log"
    # swanlab
    swanlab: bool = False
    swanlab_project: str = "mfm"
    # Empty = the account's own default workspace.
    swanlab_workspace: str = ""

    # early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_metric: str = "valid_loss"
    early_stopping_mode: str = "min"

    ifresume: Union[bool,str] = False # False , True, "model_state_strict",  "model_state_unstrict", 
cs = ConfigStore.instance()
cs.store(name="config_molfm_schema", node=SmallMolConfig)


class MolfmAtomic(torch.nn.Module):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(
        self,
        args,
        loss_fn=None,
        not_init=False,
    ):
        """
        Initialize the Model class.

        Args:
            args: Command line arguments.
            loss_fn: The loss function to use.
            data_mean: The mean of the label. For label normalization.
            data_std: The standard deviation of the label. For label normalization.
            not_init: If True, the model will not be initialized. Default is False.
        """

        super().__init__()
        if not_init:
            return

        self.args = args
        name = args.backbone_config["name"]
        if name in {"e2former", "e2formerv2"}:
            self.decoder = E2FormerBackbone(**args.backbone_config)
        else:
            raise ValueError(f"Unsupported backbone: {name}")


        self.head = MDEnergyForceMultiHead(args)
        self.loss_fn = loss_fn(args)

    def configure_tf32(self, enabled: bool = False) -> bool:
        """Configure TF32 matmul/cudnn behavior for CUDA execution."""
        if not torch.cuda.is_available():
            logger.info("CUDA is unavailable; skip TF32 configuration.")
            return False

        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled

        try:
            torch.set_float32_matmul_precision("high" if enabled else "highest")
        except Exception as exc:
            logger.warning(f"Failed to set float32 matmul precision while configuring TF32: {exc}")

        return enabled

    def forward(self, batched_data, **unused):
        """
        Forward pass of the model.

        Args:
            batched_data: Input data for the forward pass.
            **kwargs: Additional keyword arguments.
        """
        pos = batched_data["pos"]
        n_graphs, n_nodes = pos.shape[:2]

        # cell_mean = torch.mean(pos.reshape(-1, 3), dim=0).detach()
        # batched_data["pos"] = (pos - cell_mean.view(1, -1, 3) + 0.3).float()

        # Direct-force heads do not need position autograd. Keep gradients only
        # for gradient-force mode to reduce validation memory use.
        use_grad = bool(self.args.AutoGradForce)
        
        batched_data["pos"].requires_grad_(use_grad)
        decoder_output = self.decoder(
            Data(**batched_data),
            # time_embed=time_embed,
        )

        result_dict = self.head(decoder_output,
            batched_data)

        return result_dict

    def compile_model(self, recompute_budget: float = -1.0):
        """Configure and apply torch.compile.

        Args:
            recompute_budget: activation-memory budget in [0, 1] handed to the
                min-cut partitioner — the fraction of activations kept in memory,
                the remainder recomputed in the backward pass. Lower values trade
                compute for memory, which is what makes large systems fit. The
                default -1.0 leaves the partitioner alone.
        """
        try:
            from torch import _dynamo as torch_dynamo
            torch_dynamo.config.suppress_errors = True
            torch_dynamo.config.automatic_dynamic_shapes = True
            if hasattr(torch_dynamo.config, "cache_size_limit"):
                # Keep more shape variants cached to avoid frequent eviction/recompile.
                torch_dynamo.config.cache_size_limit = 128
        except Exception:
            pass
        try:
            from torch._functorch import config as functorch_config
            # Donated buffers let the partitioner reuse input storage in the
            # backward pass; incompatible with holding on to activations across
            # steps as the MD path does.
            if hasattr(functorch_config, "donated_buffer"):
                functorch_config.donated_buffer = False
            if recompute_budget >= 0.0 and hasattr(functorch_config, "activation_memory_budget"):
                functorch_config.activation_memory_budget = recompute_budget
        except Exception:
            pass
        try:
            from torch._inductor import config as inductor_config
            if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraphs"):
                inductor_config.triton.cudagraphs = False
            if hasattr(inductor_config, "cuda_graphs"):
                inductor_config.cuda_graphs = False
            if hasattr(inductor_config, "shape_padding"):
                inductor_config.shape_padding = True
        except Exception:
            pass
        compile_options = {
            # "max_autotune": True ,
            "epilogue_fusion": True,   
            "triton.cudagraphs": False, 
            "shape_padding": True       
        }

        def compile_module(module):
            try:
                return torch.compile(module,dynamic=True, options=compile_options)
            except TypeError:
                logger.warning("Falling back to mode='reduce-overhead' due to TypeError in options")
                return torch.compile(module, mode="reduce-overhead", dynamic=True)

        try:
            # For E2former, the outer backbone includes dynamic neighbor construction.
            # Compiling only the inner decoder is usually much more stable.
            if hasattr(self.decoder, "decoder") and isinstance(self.decoder.decoder, nn.Module):
                self.decoder.decoder = compile_module(self.decoder.decoder)
                logger.info("Enabled torch.compile for inner decoder module")
            else:
                self.decoder = compile_module(self.decoder)
                logger.info("Enabled torch.compile for decoder module")
            self.head = compile_module(self.head)
            logger.info("Enabled torch.compile for MolfmAtomic head")
        except Exception as exc:
            logger.warning(f"torch.compile failed for MolfmAtomic, falling back to eager: {exc}")

    def compute_loss(self, model_output, batched_data):
        """
        Compute loss for the model.

        Args:
            model_output: The output from the model.
            batched_data: The batch data.

        Returns:
            ModelOutput: The model output which includes loss, log_output, num_examples.
        """
        bs = batched_data["pos"].size(0)
        loss, log_ouput = self.loss_fn(model_output, batched_data)
        if self.head and hasattr(self.head, "update_loss"):
            loss, log_ouput = self.head.update_loss(
                loss, log_ouput, model_output, batched_data
            )
        

        return ModelOutput(loss=loss, num_examples=bs, log_output=log_ouput)

    def config_optimizer(self):
        """
        Return the optimizer and learning rate scheduler for this model.

        Returns:
            tuple[Optimizer, LRScheduler]:
        """
        return (None, None)


@hydra.main(
    version_base="1.3", config_path="../../../config_file", config_name="config_molfm"
)
def finetune(cfg: DictConfig) -> None:
    args = OmegaConf.to_object(cfg)
    args.save_dir = args.save_dir + "/" + args.experiment_name
    args.tb_dir = f"{args.tb_dir}/{args.experiment_name}"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.save_dir,"pretrain_config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    assert isinstance(args, SmallMolConfig)


    metric_logger = MetricLogger(args)
    seed_everything(args.seed)
    env_init(args)
    logger = get_logger(args.save_dir+f"/logging_{args.node_rank}.txt")


    model = MolfmAtomic(args, loss_fn=Identity)
    model.apply(simple_init_fn) #
    


    parts = args.custom_factors.split(",")                       # ["svp","10","tzvpd","20"]
    custom_factors = [(parts[2*i], float(parts[2*i+1]))     
            for i in range(0, len(parts)//2)]
    
    impl = Adam if args.weight_decay == 0 else AdamW
    param_groups = build_param_groups(model, base_lr=args.lr, weight_decay=args.weight_decay,
                                    custom_factors=custom_factors #[("svp",10),("tzvpd",10)]
                                    )
    optimizer = impl(param_groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                    weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(optimizer,
                                total_num_steps=args.total_num_steps, 
                                lr=args.lr,
                                kwargs = args.scheduler_config) # kwargs

    trainer = Trainer(
        args,
        model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        metric_logger = metric_logger
    )


    # if os.getenv("DEBUG", "0") == "1":
    #     import debugpy
    #     debugpy.listen(("0.0.0.0", 5678+int(os.environ["LOCAL_RANK"])))
    #     if int(os.environ["LOCAL_RANK"]) == 0:
    #         print("Waiting for VSCode debugger to attach on port", 5678, flush=True)
    #         debugpy.wait_for_client()
    
    if trainer.val_batch_interval<=1:
        if trainer.ifresume:
            trainer.resume()
        trainer.validate()
    else:
        trainer.train()
        trainer.validate()


if __name__ == "__main__":
    try:
        finetune()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt!")
    finally:
        wandb.finish()
