# -*- coding: utf-8 -*-
from dataclasses import fields, is_dataclass
from typing import Dict, List, Tuple, Union

import torch

from torch import nn
import math


def move_to_device(
    x: Union[
        Dict[str, Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]],
        torch.Tensor,
        List[torch.Tensor],
        Tuple[torch.Tensor],
    ],
    device: Union[str, torch.DeviceObjType],
    non_blocking: bool = False,
):
    """
    Move object to device.

    Args:
        x (dictionary of list of tensors): object (e.g. dictionary) of tensors to move to device
        device (Union[str, torch.DeviceObjType]): device, e.g. "cpu"

    Returns:
        x on targeted device
    """
    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(x, dict):
        for name in x.keys():
            x[name] = move_to_device(x[name], device=device)
    elif is_dataclass(x):
        for f in fields(x):
            setattr(x, f.name, move_to_device(getattr(x, f.name), device=device))
    elif isinstance(x, torch.Tensor) and x.device != device:
        x = x.to(device, non_blocking=non_blocking)
    elif isinstance(x, (list, tuple)):
        x = [move_to_device(xi, device=device) for xi in x]
    return x
    
def simple_init_fn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight.data, -0.001, 0.001)

