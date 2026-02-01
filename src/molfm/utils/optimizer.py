import math
from itertools import chain
from typing import Any, Dict, List, Optional
import torch

from torch.optim import Adam  # isort:skip
import warnings

def get_scheduler(optimizer,total_num_steps,lr,kwargs):
    lambda_fn = CosineLRLambda(
        total_steps=total_num_steps,
        **kwargs
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)

class CosineLRLambda:
    def __init__(
        self,
        warmup_steps: int,
        warmup_factor: float,
        total_steps: int,
        lr_min_factor: float,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.lr_warmup_factor = warmup_factor
        self.total_steps = total_steps
        self.lr_min_factor = lr_min_factor

    def __call__(self, current_step: int) -> float:
        if current_step <= self.warmup_steps:
            alpha = current_step / float(self.warmup_steps)
            return self.lr_warmup_factor * (1.0 - alpha) + alpha
        else:
            if current_step >= self.total_steps:
                return self.lr_min_factor
            return self.lr_min_factor + 0.5 * (1 - self.lr_min_factor) * (
                1 + math.cos(math.pi * (current_step / self.total_steps))
            )
