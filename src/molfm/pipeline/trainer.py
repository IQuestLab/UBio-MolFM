# -*- coding: utf-8 -*-

from builtins import ValueError
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional


import numpy as np
import torch


from loguru import logger

from molfm.data.collator import get_dataloader
from molfm.pipeline.utils import (TrainerState,
    Accelerator,
    DdpAccelerator,
    SingleNodeAccelerator,
)

from molfm.utils.dist_utils import is_master_node
from molfm.utils.torch_utils import move_to_device
from molfm.logging.loggers import LogOutput

def seed_everything(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)





class LogAccumulator(object):
    def __init__(self, world_size=1, allreduce_AVG=None):
        self.sum = 0
        self.num_examples = 0
        self._extra_log = {"loss":[]}
        self.start_time = time.time()
        self.allreduce_AVG = allreduce_AVG
        self.world_size = world_size

    def append(self, num_examples, extra_log=None):
        if type(num_examples) == torch.Tensor:
            num_examples = int(num_examples.clone().item())

        if num_examples is None or num_examples <= 0:
            return


        self.num_examples += num_examples
        
        if extra_log is not None:
            for k in extra_log:
                if k not in self._extra_log:
                    self._extra_log[k] = []
                    # self.extra_log_num[k] = []

            for k, v in extra_log.items():
                if isinstance(v, torch.Tensor):
                    if ~torch.isfinite(v) :
                        logger.error(f"some log is nan or inf,{extra_log}, {k}, {v}, , we reset to the latest one.")
                        fallback = self._extra_log[k][-1] if self._extra_log[k] else 0.0
                        self._extra_log[k].append(fallback)
                    else:
                        self._extra_log[k].append(float(v.item()))
                elif isinstance(v, tuple):
                    self._extra_log[k].append(float(v[0]) /float(v[1]))
                else:
                    self._extra_log[k].append(float(v))
                    # self.extra_log_num[k] += 1 * num_examples

    def reset(self):
        self.sum = 0.0
        self.num_examples = 0
        self.start_time = time.time()
        self._extra_log = {}
        # for k, v in self._extra_log.items():
        #     self._extra_log[k] = []
            # self.extra_log_num[k] = 0

    @property
    def averge_loss(self):
        if self.num_examples == 0 or "loss" not in self._extra_log:
            return 0
        if self.allreduce_AVG is not None:
            log_dict = {"loss": np.mean(np.array(self._extra_log["loss"]))}
            reduced_loss_dict = self.allreduce_AVG(log_dict)
            return reduced_loss_dict["loss"]
        else:
            return self.sum / self.num_examples

    @property
    def averge_log(self):
        reduced_log = {
                k: np.mean(np.array(self._extra_log[k])) if self._extra_log[k] else 0
                for k, v in self._extra_log.items()
            }
        reduced_log.update({"SamplePerSec":self.num_examples/(time.time() - self.start_time)})
        if self.world_size != 1 and self.allreduce_AVG is not None:
            reduced_log = self.allreduce_AVG(reduced_log)

        reduced_log["SamplePerSec"] *= self.world_size
        return reduced_log



class Trainer(object):
    def __init__(
        self,
        args,
        model,
        optimizer: Optional[torch.optim.Optimizer],
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        metric_logger,
        # use by nnscaler accelerator
        model_init: Callable[[], torch.nn.Module] = None,
        
    ):
        super().__init__()
        self.args = args
        # logger.info("Trainer args: {}", args)


        self.ifresume = args.ifresume
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.save_dir = Path(args.save_dir)
        
  
        self.total_num_epochs = args.total_num_epochs
        self.total_num_steps = args.total_num_steps
        self.save_batch_interval = args.save_batch_interval
        self.save_epoch_interval = args.save_epoch_interval
        self.log_interval = args.log_interval
        self.val_batch_interval = args.val_batch_interval
        self.val_epoch_interval = args.val_epoch_interval

        self.clip_grad_norm = args.clip_grad_norm
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.state = TrainerState()

        self.strategy_config = args.strategy_config # default({"name":"single"})


        self.model_init = model_init

        # if args.mm_tensorcore == "bf16":
        #     torch.set_float32_matmul_precision("medium")
        # elif args.mm_tensorcore == "tf32":
        #     torch.set_float32_matmul_precision("high")
        # else:
        #     torch.set_float32_matmul_precision("highest")


        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.loss_fn = model.compute_loss
        self.accelerator = self.build_accelerator(model)
        self.accelerator.set_up()
        self.model = self.accelerator.acc_model


        self.train_dataloader, self.valid_dataloader, self.test_dataloader = get_dataloader(args)



        self.world_size = self.accelerator.world_size
        self.rank = self.accelerator.rank
        self.device = self.accelerator.device
        self.start_iteration = 0
        self.metric_logger = metric_logger

    def save_checkpoint(self, ckpt_path: str, state: Optional[dict] = None):

        self.accelerator.save_checkpoint(ckpt_path, state)
        self._save_rng_and_iter_state(self.save_dir)

    def _load_checkpoint(self, ckpt_path: Path, strict=True,model_states_only=False):
        if isinstance(ckpt_path,str):ckpt_path=Path(ckpt_path)
        if ckpt_path.exists():
            logger.info(f"{['Resume','Finetune'][model_states_only]} from checkpoint: {ckpt_path}, strict {strict}")
            _state = self.accelerator.load_checkpoint(
                ckpt_path,
                strict=strict,
                model_states_only=model_states_only,
            )

            if not model_states_only:
                for k, v in _state.items():
                    setattr(self.state, k, v)
        else:
            logger.warning(f"Checkpoint path {ckpt_path} does not exist.")
        

    def resume(self):
        if not self.ifresume:
            return None
            
        if self.args.ckpt_path not in [None,"none","None"]:
            checkpoint_last = self.args.ckpt_path
        else:
            if (self.save_dir / "checkpoint_list.txt").exists():
                checkpoint_list_path = self.save_dir / "checkpoint_list.txt"
            elif (self.save_dir / "latest").exists(): # this is for DeepSpeed
                checkpoint_list_path = self.save_dir / "latest"
            else:
                return

            checkpoint_last = None
            with open(checkpoint_list_path, "r") as f:
                checkpoint_list = f.read().splitlines()
            if len(checkpoint_list) > 0:
                checkpoint_last = self.save_dir / checkpoint_list[-1]


        if self.ifresume == True:
            model_states_only=False # load everything, e.g. optimizer, lr_scheduler, model state, state. 
            strict=True
        elif self.ifresume=="model_state_strict":
            model_states_only=True
            strict = True
        elif self.ifresume=="model_state_unstrict":
            model_states_only=True
            strict = False
        else:
            raise ValueError(" sorry, ifresume parameter set error.")
        if checkpoint_last is not None:
            self._load_checkpoint(checkpoint_last,strict, model_states_only)
            self._load_rng_and_iter_state(self.save_dir)
        else:
            logger.warning(
                f"Non-empty checkpoint_list.txt or latest file is not present in {self.save_dir}, "
                f"or finetune_from_checkpoint_id is not provided. No checkpoint is loaded."
            )

    def build_accelerator(self,model) -> Accelerator:
        if self.strategy_config["name"] == "single":
            return SingleNodeAccelerator(
                model,
                self.optimizer,
                self.lr_scheduler,
                "cuda",
            )
        elif self.strategy_config["name"] == "ddp":
            return DdpAccelerator(
                model,
                self.optimizer,
                self.lr_scheduler,
                **self.strategy_config
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy_config}")

    def build_log_output(self, loss, extra_output=None) -> LogOutput:
        try:
            lr = self.lr_scheduler.get_last_lr()[0]
        except:
            lr = 0.0

        if type(loss) == torch.Tensor:
            loss = loss.item()

        return LogOutput(
            loss=loss,
            lr=lr,
            epoch=self.state.epoch,
            batch=self.state.batch,
            num_examples=self.state.sample*self.world_size,
            global_step=self.state.global_step,
            extra_output=extra_output,
        )
    def should_stop(self) -> bool:
        return (
            (self.total_num_epochs and self.state.epoch >= self.total_num_epochs)
            or (self.total_num_steps and self.state.global_step >= self.total_num_steps)
        )

    def should_save_batch_checkpoint(self) -> bool:
        return (
            self.save_batch_interval > 0
            and self.state.global_step % self.save_batch_interval == 0
        )

    def should_save_epoch_checkpoint(self) -> bool:
        return (
            self.save_epoch_interval > 0
            and self.state.epoch % self.save_epoch_interval == 0
        )

    def should_log(self) -> bool:
        return (
            self.log_interval > 0
            and self.state.global_step % self.log_interval == 0
        )

    def should_do_batch_validate(self) -> bool:
        return (
            self.val_batch_interval > 0
            and self.state.global_step % self.val_batch_interval == 0
        )

    def should_do_epoch_validate(self) -> bool:
        return (
            self.val_epoch_interval > 0
            and (self.state.epoch + 1) % self.val_epoch_interval == 0
        )


    def count_parameters(self):
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        return total_num, trainable_num


    def train(self):
        """
        Train the model on the training data loader.
        """
        logger.info("Start training")
        if is_master_node():
            logger.info(self.model)

        assert self.train_dataloader is not None

        if hasattr(self.model, "before_training"):
            self.model.before_training()
        if self.ifresume:
            self.resume()
        
        total_num, trainable_num = self.count_parameters()
        logger.info(
            "Total number of parameters: {:,} - number of trainable parameters: {:,}.",
            total_num,
            trainable_num,
        )


        while (
            self.state.epoch < self.total_num_epochs
            and self.state.global_step < self.total_num_steps
        ):
            self.accelerator.before_epoch(self.state.epoch)
            if self.strategy_config["name"] == "ddp":
                if hasattr(self.train_dataloader.sampler, "set_epoch"):
                    self.train_dataloader.sampler.set_epoch(self.state.epoch)

                if hasattr(self.train_dataloader.batch_sampler, "set_epoch"):
                    self.train_dataloader.batch_sampler.set_epoch(self.state.epoch)

                if hasattr(self.train_dataloader.batch_sampler, "set_start_iter"):
                    self.train_dataloader.batch_sampler.set_start_iter(self.start_iteration)
                    self.start_iteration = 0
            else:
                raise ValueError(f"Unknown strategy: {self.strategy_config}")   
            logger.info("Start Training for epoch: {}", self.state.epoch)

            interval_loss_accumulator = LogAccumulator(
                self.world_size, self.accelerator._allreduceAVG
            )

            logger.info(f"self.start_iteration = {self.start_iteration}.")
            try:
                total_sample_count = 0
                total_loss = 0.0
                logger.info(f"len trainer dataloader: {len(self.train_dataloader)}")
                for idx,batch_data in enumerate(self.train_dataloader):
                    self.model.train()
                    batch_data = move_to_device(batch_data, self.device)
                    # No sync for gradient accumulation
                    maybe_no_sync = (
                        self.model.no_sync()
                        if hasattr(self.model, "no_sync") and 
                        (idx % self.gradient_accumulation_steps != 0)
                        else nullcontext() # this means sync
                    )
                    with maybe_no_sync:
                        pred = self.model(batch_data)
                        model_output = self.loss_fn(pred, batch_data)
                        total_sample_count += model_output.num_examples
                        total_loss += model_output.loss.item() * model_output.num_examples

                        interval_loss_accumulator.append(
                            model_output.num_examples,
                            model_output.log_output,
                        )

                        loss = model_output.loss / self.gradient_accumulation_steps

                        local_bad = ~torch.isfinite(loss)  

                        
                        if torch.distributed.is_initialized():
                            
                            flag = local_bad.to(torch.int32)
                            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.SUM)
                            global_bad = flag > 0
                        else:
                            global_bad = local_bad
                        if global_bad:
                            logger.error("Notice, we suffer loss nan/infinite sample, thus we reset loss and log to zero")
                            loss = torch.zeros_like(loss,requires_grad=True)
                        

                        loss.backward()
                        if global_bad:
                            self.optimizer.zero_grad(set_to_none=False)

                        if idx%self.gradient_accumulation_steps == 0:
                            gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                            interval_loss_accumulator._extra_log["gradient_norm"] = gradient_norm.detach().cpu().item()
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad()
                            self.state.batch += 1
                            self.state.global_step += 1
                            self.state.sample += model_output.num_examples

                    if self.should_log() or self.state.global_step==5:
                        log_output = self.build_log_output(
                            interval_loss_accumulator.averge_loss,
                            interval_loss_accumulator.averge_log,
                        )
                        interval_loss_accumulator.reset()
                        self.metric_logger.log(
                            log_output, "train_inner", self.state.global_step
                        )

                    if self.should_save_batch_checkpoint():
                        checkpoint_name = (
                            f"checkpoint_E{self.state.epoch}_B{self.state.batch}.pt"
                        )
                        self.save_checkpoint(self.save_dir / checkpoint_name, asdict(self.state))

                    if self.should_do_batch_validate():
                        self.validate()
                    
                    if self.should_stop():
                        break
                    
            except StopIteration:
                logger.info("StopIteration")
                pass

            log_output = self.build_log_output(total_loss/total_sample_count)
            self.metric_logger.log(log_output, "train", self.state.global_step)

            logger.info("reset batch {} to zero", self.state.batch)
            self.state.batch = 0
            self.accelerator.barrier()
            if self.should_save_epoch_checkpoint():
                checkpoint_name = f"checkpoint_E{self.state.epoch}.pt"
                self.save_checkpoint(self.save_dir / checkpoint_name, asdict(self.state))

            if self.should_do_epoch_validate():
                self.validate()

            self.accelerator.barrier()


            if self.should_stop():
                break
            self.state.epoch += 1
        if hasattr(self.model, "after_training"):
            self.model.after_training()


        logger.info("Finished Training")

    def validate(self):
        """
        Validate the model on the validation data loader.
        """
        if self.valid_dataloader is None:
            logger.warning("No validation data, skip validation")
            return

        logger.info(
            "Start validation for epoch: {}, global step: {}",
            self.state.epoch,
            self.state.global_step,
        )


        interval_loss_accumulator = LogAccumulator(
            self.world_size, self.accelerator._allreduceAVG
        )

        total_sample_count = 0
        total_loss = 0.0
        final_log = {}
        self._logged_first_invalid_valid_batch = False
        use_grad = bool(getattr(self.args, "AutoGradForce", False))
        grad_context = torch.enable_grad if use_grad else torch.no_grad

        for idx, batch_data in enumerate(self.valid_dataloader):
            self.model.eval()
            self.model.zero_grad()

            batch_data = move_to_device(batch_data, self.device)
            with grad_context():
                pred = self.model(batch_data)
                model_output = self.loss_fn(pred, batch_data)
            total_sample_count += model_output.num_examples
            total_loss += model_output.loss.detach() * model_output.num_examples
            interval_loss_accumulator.append(
                model_output.num_examples,
                model_output.log_output,
            )
            if ((idx + 1) % self.args.log_interval == 0) or (idx == len(self.valid_dataloader)-1):
                logger.info(
                    "Validtion batch: {} , loss: {} , loss after reduce {}. \n {} ",
                    idx + 1,
                    total_loss/total_sample_count,
                    interval_loss_accumulator.averge_loss,
                    interval_loss_accumulator.averge_log
                )
                for key, value in interval_loss_accumulator.averge_log.items():
                    final_log.setdefault(key, []).append(value)
                interval_loss_accumulator.reset()



        for key,value in final_log.items():
            final_log[key] = np.mean(np.array(value))
        
        self.metric_logger.log(
            final_log, "valid", self.state.global_step
        )

        return final_log

    def _save_rng_and_iter_state(self, checkpoint):
        """
        Save the RNG and iteration states to the checkpoint to resume training from break point.
        Args:
            checkpoint (str): the path to the checkpoint
        """
        if checkpoint is None:
            return

        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
            "cuda": (
                torch.cuda.random.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "iteration": self.state.batch,
            "epoch": self.state.epoch,
        }

        if self.world_size > 1:
            process_index = self.rank
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")

        torch.save(rng_state, rng_file)

    def _load_rng_and_iter_state(self, checkpoint):
        """
        Load the RNG and iteration states from the checkpoint to resume training from break point.
            e.g. python random, numpy random, torch cpu/gpu random seed.
        Args:
            checkpoint (str): the path to the checkpoint
        """
        if checkpoint is None:
            return

        if self.world_size > 1:
            process_index = self.rank
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")

        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
        if not os.path.isfile(rng_file):
            logger.warning(
                f"Didn't find an RNG file {rng_file}. "
                "this file is used to ensure reproducibility, especially for DDP mode."
            )
            return

        checkpoint_rng_state = torch.load(rng_file, weights_only=False)
        try:
            random.setstate(checkpoint_rng_state["python"])
        except:
            logger.warning(
                "Python random module set RNG states FAILED/ERROR"
            )
        try:
            np.random.set_state(checkpoint_rng_state["numpy"])
        except:
            logger.warning(
                "Numpy random module set RNG states FAILED/ERROR"
                "results as if the training had not been interrupted."
            )
        try:
            torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        except:
            logger.warning(
                "torch CPU set RNG states FAILED/ERROR"
            )
        if torch.cuda.is_available():
            if self.world_size > 1:
                try:
                    torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
                except Exception as e:
                    logger.warning(
                        f"GPU because of the following error:\n {e}"
                        "\nMultiGPU set RNG states FAILED/ERROR"
                    )
            else:
                try:
                    torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
                except Exception as e:
                    logger.warning(
                        f"Didn't manage to set back the RNG states of the GPU because of the following error:\n {e}"
                        "\nSingleGPU set RNG states FAILED/ERROR"
                    )

        if "epoch" in checkpoint_rng_state:
            self.state.epoch = checkpoint_rng_state["epoch"]



    def skip_first_batches(self, data_iterator, start_iteration=None):
        """
        Skip the first start_iteration batches in the training data loader to resume training from break point.
        Args:
            start_iteration (int): the number of batches to skip
        """

        if start_iteration is None or start_iteration == 0:
            return data_iterator

        self.state.batch = start_iteration

        # Skip samples
        to_skip = start_iteration * self.gradient_accumulation_steps
        skipped = 0
        for batch in data_iterator:
            if skipped < to_skip:
                skipped += 1
                continue
            self.start_iteration = 0
            return data_iterator  # Only yield after skipping
    
        self.start_iteration = 0
        return data_iterator
    
