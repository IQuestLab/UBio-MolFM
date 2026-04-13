# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch

@dataclass
class TrainerState:
    """
    The TrainerState class helps manage various training-related attributes, making it easier to monitor and control the progress of the training.

    Args:
        args: A TrainerConfig object that stores the configuration settings for the training process.

        global_step (int, default: 0): The current global step of the training process, which is a count of the number of gradient updates performed.

        epoch (int, default: 0): The current epoch of the training process, representing the number of times the entire dataset has been processed.

        batch (int, default: 0): The current batch number within the current epoch.
    """

    global_step: int = 0
    epoch: int = 0
    batch: int = 0
    sample: int = 0


@dataclass
class ModelOutput:
    loss: torch.Tensor
    num_examples: Optional[int] = None
    log_output: Optional[Dict] = None



from abc import ABC, abstractmethod
import multiprocessing
import os
from datetime import timedelta
from pathlib import Path

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from loguru import logger


def safe_div(a, b):
    if b == 0:
        return 0
    return a / b




class Accelerator(ABC):
    @abstractmethod
    def set_up():
        pass

    @abstractmethod
    def save_checkpoint(
        self, ckpt_path: Union[int, str], extra_state: Optional[dict] = None
    ):
        pass

    @abstractmethod
    def load_checkpoint(
        self,
        ckpt_path: Path,
        strict=True,
        model_states_only: bool = False,
    ):
        pass

    @abstractmethod
    def barrier(self):
        pass

    def before_epoch(self, epoch: int):
        pass



class SingleNodeAccelerator(Accelerator):
    def __init__(
        self, model, optimizer, lr_scheduler, device: str,name="single"
    ) -> None:
        super().__init__()
        if name.lower()!="single":raise ValueError(f"single node acceleretor call with {name}")
        self.model = model
        self.acc_model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.world_size = 1
        self.rank = 0
        if not torch.cuda.is_available():
            logger.warning("cuda is not available. use cpu instead")
            self.device = "cpu"

        self.model.to(self.device)


    def set_up(self):
        if self.optimizer is None:
            self.optimizer, self.lr_scheduler = self.model.config_optimizer()

    def barrier(self):
        pass




    def save_checkpoint(self, ckpt_path: Union[Path, str], state: Optional[dict] = None):
        state = {} if state is None else state
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "state":state
        }

        logger.info("save checkpoint: {}", ckpt_path)
        torch.save(checkpoint,  ckpt_path)

        with open(Path(os.path.dirname(os.path.abspath(ckpt_path))) / "checkpoint_list.txt", "a") as f:
            f.write("\n" +str(os.path.basename(ckpt_path)))


    def load_checkpoint(
        self,
        ckpt_path:  str,
        strict=True,
        model_states_only=False,

    ):
        checkpoint = torch.load(ckpt_path, map_location="cpu",weights_only=False)
        self.model.load_state_dict(checkpoint["model"],strict=strict)
        if not model_states_only:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            logger.info(f"optimizer is loaded from checkpoint {ckpt_path}")


        return checkpoint["state"]


    @staticmethod
    def _allreduceAVG(log_dict: Optional[dict] = None):
        log_dict = {} if log_dict is None else log_dict
        for k, v in log_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            log_dict[k] = v.item()


        return log_dict



class DdpAccelerator(SingleNodeAccelerator):
    def __init__(self, model, optimizer, lr_scheduler,
            name="ddp",dist_backend="nccl",find_unused_parameters=True) -> None:
        super().__init__(model, optimizer, lr_scheduler, device="cuda")
        if name.lower() != "ddp":raise ValueError(f"sorry , strategy ddp found name {name}")
        self.dist_backend = dist_backend
        self.find_unused_parameters = find_unused_parameters

    def set_up(self):
        super().set_up()
        assert "WORLD_SIZE" in os.environ, "WORLD_SIZE must be set to use DDP"
        assert "RANK" in os.environ, "RANK must be set to use DDP"
        assert "LOCAL_RANK" in os.environ, "LOCAL_RANK must be set to use DDP"

        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])

        master_addr = os.environ.get("MASTER_ADDR", "")
        master_port = os.environ.get("MASTER_PORT", "")

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        multiprocessing.set_start_method("spawn", force=True)

        ddp_timeout = os.environ.get("DDP_TIMEOUT_MINUTES", None)
        logger.critical(
            f"Init DDP by {self.dist_backend}. env://. world size: {self.world_size}, rank: {self.rank}, "
            f"local_rank: {self.local_rank}, master_addr: {master_addr}, master_port: {master_port}, "
            f"DDP_TIMEOUT_MINUTES: {ddp_timeout}"
        )
        torch.distributed.init_process_group(
            backend=self.dist_backend,
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
            timeout=timedelta(minutes=int(ddp_timeout))
            if ddp_timeout is not None
            else timedelta(minutes=30),
        )

        torch.distributed.barrier()

        logger.success("DDP initialized successfully.")

        self.model.to(self.device)

        self.acc_model  = DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=self.find_unused_parameters,
        )


    def barrier(self):
        torch.distributed.barrier()

    def before_epoch(self, epoch: int):
        pass
    
    def save_checkpoint(self, ckpt_path: str, extra_state: Optional[dict] = None):
        if self.rank == 0:
            logger.info(f"-------------------{self.rank}")
            super().save_checkpoint(ckpt_path, extra_state)

        torch.distributed.barrier()


    @staticmethod
    def _allreduceAVG(log_dict: Optional[dict] = None):
        log_dict = {} if log_dict is None else log_dict
        world_size = int(os.environ["WORLD_SIZE"])

        my_keys = list(log_dict.keys())
        key_lists = [None for _ in range(world_size)]
        dist.all_gather_object(key_lists, my_keys, group=dist.group.WORLD)
        all_keys = sorted(set().union(*map(set, key_lists)))

        
        t_sum = torch.zeros(len(all_keys)).cuda()
        t_cnt = torch.zeros(len(all_keys)).cuda()
        for i,k in enumerate(all_keys):
            if k in log_dict:
                v = log_dict[k]
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v)
                t_sum[i] = v
                t_cnt[i] = 1
        

        dist.all_reduce(t_sum, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        dist.all_reduce(t_cnt, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

        mean = t_sum / torch.clamp(t_cnt, min=1.0)  

        return {k: mean[i].item() for i, k in enumerate(all_keys)}
