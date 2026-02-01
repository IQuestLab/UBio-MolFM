# -*- coding: utf-8 -*-
import os

import torch
import warnings

def is_master_node():
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        return True
    else:
        return False



def env_init(args):
    torch.set_flush_denormal(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    if os.environ.get("LOCAL_RANK") is not None:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
        os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = os.environ["LOCAL_RANK"]
        torch.cuda.set_device(args.local_rank)

        warnings.warn(
            "Print os.environ:--- RANK: {}, WORLD_SIZE: {}, LOCAL_RANK: {}".format(
                os.environ["RANK"], os.environ["WORLD_SIZE"], os.environ["LOCAL_RANK"]
            )
        )


