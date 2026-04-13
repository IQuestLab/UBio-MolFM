# -*- coding: utf-8 -*-
import json
import os
import sys
from dataclasses import dataclass, fields, is_dataclass, asdict
from typing import Dict, Union

from loguru import logger


import wandb  # isort:skip
import swanlab # isort:skip
import torch
from aim import Run


import warnings
from molfm.utils.dist_utils import is_master_node
from torch.utils.tensorboard import SummaryWriter

def format_extra_output(raw_extra_output):
    if raw_extra_output is None:
        return ""

    extra_output = []
    for k, v in raw_extra_output.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                v = v.item()
                extra_output.append(f"{k}: {v:.4g}")
            else:
                v = v.detach().cpu().numpy()
                extra_output.append(f"{k}: {v}")
        elif isinstance(v, float):
            extra_output.append(f"{k}: {v:.4g}")
        else:
            extra_output.append(f"{k}: {v}")
    extra_output = " | ".join(extra_output)
    return extra_output


@dataclass
class LogOutput:
    loss: float
    lr: float
    epoch: int
    batch: int
    global_step: int
    num_examples: int
    extra_output: Dict

    def __str__(self) -> str:
        extra_output = format_extra_output(self.extra_output)
        return (
            f"Step: {self.global_step} (Epoch {self.epoch} Iter {self.batch+1}) | Loss: {self.loss:.4g} | LR: {self.lr:.4g} | Sample: {self.num_examples}"
            + extra_output
        )


def wandb_init(args):
    if args is None:return None
    if is_master_node():
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            warnings.warn("Wandb not configured, logging to console only")
        else:
            wandb.login(key=wandb_api_key)
            wandb.init(
                project=args.wandb_project,     
                # project=args.wandb_project,
                # group=args.wandb_group,
                name=args.experiment_name,
                # entity=args.wandb_team if args.wandb_team != "" else None,
                config=args,
            )
            wandb.run.notes = f"Command: {' '.join(sys.argv)}"
    return None

def swanlab_init(args,save_dir = "./"):
    if args is None:
        return None

    if is_master_node():    
        record_file = os.path.join(save_dir, f"swanlab-record.json")
        record = {
            }
        if os.path.exists(record_file):
            with open(record_file, "r") as f:
                record = json.load(f)

        
        _swanlab_writer = swanlab.init(
                
                # project="mfm",
                resume = "allow" if "id" in record else False,
                id=record["id"] if "id" in record else None,
                project=args.swanlab_project,     
                workspace=args.swanlab_workspace,
                experiment_name=args.experiment_name,
                
                config=args
        )
        record["id"] = _swanlab_writer.id

        with open(record_file, "w") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=4))

        return None
def aim_init(args):
    if args is None:return None
    if is_master_node():
        run = Run(repo=args.aim_dir, experiment=args.experiment_name)
        run['hparams'] = asdict(args)
        return run
    return None


def tensorboard_init(args):
    if args is None:return None
    if is_master_node():
        writer = SummaryWriter(args.tb_dir)
        # writer.add_hparams(asdict(args),{'hparam/demo': 0})
        return writer
    
    return None

def console_log_filter(record):
    # For message with level INFO, we only log it on master node
    # For others, we log it on all nodes
    if record["level"].name != "INFO":
        return True

    return is_master_node()


handlers = {}
# this is customer defined logger. 
def get_logger(log_path = "./logging.txt"):
    if not handlers:
        logger.remove()  # remove default handler
        handlers["console"] = logger.add(
            sys.stdout,
            format="[<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>][<cyan>{level}</cyan>]: {message}",
            colorize=True,
            filter=console_log_filter,
            enqueue=True,
        )
        # redirect warnings to loguru
        def showwarning(message, category, filename, lineno, file=None, line=None):
            logger.warning(f"{filename}:{lineno}: {category.__name__}: {message}")
        # File logger
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers["file"] = logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} [{level}]: {message}",
            
            enqueue=True,
            
        )

        warnings.showwarning = showwarning
    return logger


# Custom function to handle tensor attributes
def dataclass_to_dict(dataclass_obj: Union[dataclass, Dict]) -> Dict:
    if isinstance(dataclass_obj, dict):
        return dataclass_obj
    result = {}
    for field in fields(dataclass_obj):
        value = getattr(dataclass_obj, field.name)
        if isinstance(value, torch.Tensor) and not value.is_leaf:
            result[field.name] = value.clone().detach()
        else:
            result[field.name] = value
    return result


class MetricLogger(object):
    def __init__(self,args = None,save_dir = None):
        self.args = args
        if args.wandb:
            wandb_init(args)
        
        if args.swanlab:
            swanlab_init(args,args.save_dir)

        if args.tb:
            self.tb_writer = tensorboard_init(args)
        else:
            self.tb_writer = None
    def log(self, metrics:Union[dict,LogOutput], prefix="", global_step=None, log_dashboard=True):
        if not is_master_node():
            return

        log_data = {}
        if type(metrics) is dict:
            log_data = metrics
        elif is_dataclass(metrics):
            log_data = dataclass_to_dict(metrics)

        if "extra_output" in log_data:
            extra_output = log_data["extra_output"]
            if extra_output is not None:
                for k, v in extra_output.items():
                    log_data[k] = v
            del log_data["extra_output"]

        for k in log_data:
            if isinstance(log_data[k], torch.Tensor):
                log_data[k] = log_data[k].detach().item()

        log_str = prefix
        for k, v in log_data.items():
            if v is not None:
                log_str += f" | {k}={v} "
        logger.info(log_str)
        if prefix:
            log_data = {f"{prefix}/{k}": v for k, v in log_data.items()}
        if log_dashboard and wandb.run is not None:
            # Add prefix
            wandb.log(log_data, step=global_step)
        if log_dashboard and swanlab.run is not None and self.args.swanlab:
            # Add prefix
            swanlab.log(log_data, step=global_step)

        if self.tb_writer: 
            for k, v in log_data.items():
                self.tb_writer.add_scalar(k,v, global_step)
