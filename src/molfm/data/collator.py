# -*- coding: utf-8 -*-
# Licensed under the MIT License.
import os

import math

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from .dataset import MoleculeLMDBDataset, TOYxyzDataset, SPICExyzDataset

_register_keys = set([])

special_pad_keys = {
    "pos": lambda x, pad_len: F.pad(x, (0, 0, 0, pad_len)),
    "atomic_numbers": lambda x, pad_len: F.pad(x, (0, pad_len)),
    "fixed": lambda x, pad_len: F.pad(x, (0, pad_len)),
    "cos": lambda x, pad_len: F.pad(x, (0, pad_len)),
    "forces": lambda x, pad_len: F.pad(x, (0, 0, 0, pad_len)),
}

def collate_fn(batch):
    max_n = max(sample["pos"].shape[0] for sample in batch)
    bs = len(batch)

    atom_masks = torch.zeros((bs,max_n)).bool()
    batch_dict = {}

    for key in batch[0].keys():
        batch_dict[key] = []

    for idx, sample in enumerate(batch):
        n = sample["pos"].shape[0]
        pad_len = max_n - n

        for key, value in sample.items():
            if key in special_pad_keys:
                batch_dict[key].append(special_pad_keys[key](value, pad_len))
            else:
                batch_dict[key].append(value)

        atom_masks[idx, :n] = 1

    for key, values in batch_dict.items():
        if key not in set(['data_name','idx','data_src']):
            batch_dict[key] = torch.stack(values)
    
    batch_dict["atom_masks"] = atom_masks
    return batch_dict


class TokenBatchDistributedSampler(Sampler):
    def __init__(self, lengths, max_tokens, num_replicas=None, rank=None, shuffle=True, seed=42, drop_last=False):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires torch.distributed")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires torch.distributed")
            rank = torch.distributed.get_rank()

        self.lengths = np.array(lengths)
        self.max_tokens = max_tokens
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.total_size = len(lengths)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = list(range(self.total_size))
        if self.shuffle:
            indices = torch.randperm(self.total_size, generator=g).tolist()

        batches, batch, batch_len = [], [], 0
        for idx in indices:
            length = self.lengths[idx]
            if batch and (batch_len + length > self.max_tokens):
                batches.append(batch)
                batch, batch_len = [], 0
            batch.append(idx)
            batch_len += length
        if batch and not self.drop_last:
            batches.append(batch)

        num_batches = len(batches)
        if num_batches % self.num_replicas != 0:
            pad_size = self.num_replicas - (num_batches % self.num_replicas)
            if not self.drop_last:
                batches.extend(batches[:pad_size])
            else:
                batches = batches[:num_batches - (num_batches % self.num_replicas)]

        per_rank_batches = batches[self.rank::self.num_replicas]

        for batch in per_rank_batches:
            yield batch

    def __len__(self):
        # just used for estimation.
        return math.ceil(np.sum(self.lengths) / self.max_tokens / self.num_replicas)

    def set_epoch(self, epoch):
        self.epoch = epoch



class MultiDatasetTokenBatchDistributedSampler(Sampler):
    def __init__(self,
                 lengths_list,
                 max_token_list,
                 sample_prob,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=42,
                 drop_last=False):
        if num_replicas is None or rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires torch.distributed or pass num_replicas/rank explicitly")
            if num_replicas is None:
                num_replicas = torch.distributed.get_world_size()
            if rank is None:
                rank = torch.distributed.get_rank()

        self.lengths_list = [np.asarray(l, dtype=np.int64) for l in lengths_list]
        self.sizes = [len(l) for l in self.lengths_list]
        self.ds_offsets = np.cumsum([0] + self.sizes[:-1]).tolist() 
        self.total_size = sum(self.sizes)
        self.sample_prob = sample_prob
        self.max_token_list = max_token_list
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.start_iter = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        total_lens = [
                    int(np.sum(l)*p/max_token/self.num_replicas) 
                    for l,max_token,p in zip(self.lengths_list,self.max_token_list,self.sample_prob)]
        return int(sum(total_lens))

    def _per_dataset_batches(self, g):
        all_batches = []
        for ds_id, lens in enumerate(self.lengths_list):
            max_token = self.max_token_list[ds_id]
            prob = self.sample_prob[ds_id]
            n = len(lens)
            
            if self.shuffle:
                local_idx = (torch.randperm(int(n*max(prob,1)), generator=g)%n).tolist()
            else:
                local_idx = list(range(n))
            local_idx = local_idx[:int(prob*n)]
            
            batches_ds = []
            cur_batch, cur_tokens = [], 0
            base = self.ds_offsets[ds_id]  
            for i in local_idx:
                li = int(lens[i])
                if cur_batch and (cur_tokens + li > max_token):
                    batches_ds.append(cur_batch)
                    cur_batch, cur_tokens = [], 0
                cur_batch.append(base + i)  
                cur_tokens += li
            if cur_batch and not self.drop_last:
                batches_ds.append(cur_batch)

            all_batches.extend(batches_ds)

        return all_batches

    def set_start_iter(self, start_iter: int):
        self.start_iter = int(start_iter)

    def __iter__(self):
        
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        
        batches = self._per_dataset_batches(g)

        
        if self.shuffle:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]

        
        num_batches = len(batches)
        if num_batches % self.num_replicas != 0:
            pad = self.num_replicas - (num_batches % self.num_replicas)
            if not self.drop_last:
                
                batches.extend(batches[:pad])
            else:
                batches = batches[:num_batches - (num_batches % self.num_replicas)]

        
        per_rank_batches = batches[self.start_iter+self.rank::self.num_replicas]

        
        for batch in per_rank_batches:
            yield batch



class MultiDatasetFixedBatchDistributedSampler(Sampler):
    """
    Distributed sampler for fixed-size batches across multiple datasets.
      - sizes: List[int], number of samples in each sub-dataset
      - batch_sizes: List[int], batch size for each sub-dataset
      - No cross-dataset mixing: split batches within each dataset, then
        extend into a flat batch list
      - Batch-level shuffle only; no round-robin / max_tokens mode
      - Designed for ConcatDataset and returns batches of global indices
    """
    def __init__(self,
                 sizes,
                 batch_sizes,
                 sample_prob,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=42,
                 drop_last=False):
        if len(sizes) != len(batch_sizes):
            raise ValueError("sizes and batch_sizes must have the same length")
        self.sizes = list(map(int, sizes))
        self.batch_sizes = list(map(int, batch_sizes))
        self.num_datasets = len(self.sizes)
        self.sample_prob = sample_prob
        
        if num_replicas is None or rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires torch.distributed or pass num_replicas/rank explicitly")
            if num_replicas is None:
                num_replicas = torch.distributed.get_world_size()
            if rank is None:
                rank = torch.distributed.get_rank()
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        
        self.ds_offsets = []
        acc = 0
        for n in self.sizes:
            self.ds_offsets.append(acc)
            acc += n
        self.total_size = acc

        self.epoch = 0
        self.start_iter = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        
        tot_batches = 0
        for n, bs,p in zip(self.sizes, self.batch_sizes,self.sample_prob):
            if self.drop_last:
                tot_batches += int(n*p) // bs
            else:
                tot_batches += (int(n*p) + bs - 1) // bs

        return tot_batches // self.num_replicas

    def _build_flat_batches(self, g: torch.Generator):
        """
        Split each dataset into fixed-size batches and extend them into a flat
        batch list. Each batch contains global ConcatDataset indices and never
        mixes samples across datasets.
        """
        all_batches = []

        for ds_id, (n, bs,prob) in enumerate(zip(self.sizes, self.batch_sizes,self.sample_prob)):
            
            if self.shuffle:
                local_idx = torch.randperm(n, generator=g).tolist()
            else:
                local_idx = list(range(n))
            local_idx = local_idx[:int(prob*n)]

            
            off = self.ds_offsets[ds_id]
            for i in range(0, n, bs):
                chunk = local_idx[i:i+bs]
                if len(chunk) < bs and self.drop_last:
                    break
                all_batches.extend([[off + j for j in chunk]])

        return all_batches  
    
    def set_start_iter(self, start_iter: int):
        self.start_iter = int(start_iter)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        batches = self._build_flat_batches(g)

        if self.shuffle and len(batches) > 1:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]

        num_batches = len(batches)
        if num_batches % self.num_replicas != 0:
            if self.drop_last:
                keep = num_batches - (num_batches % self.num_replicas)
                batches = batches[:keep]
            else:
                pad = self.num_replicas - (num_batches % self.num_replicas)
                if pad > 0:
                    batches.extend(batches[:pad])

        per_rank_batches = batches[self.start_iter+self.rank::self.num_replicas]

        for b in per_rank_batches:
            yield b


def get_dataloader(args):
    # Dataset will automatically load based on the args
    remove_atomref_energy = (
        args.remove_atomref_energy if hasattr(args, "remove_atomref_energy") else True
    )

    file_list = []
    file_list_valid = []
    dataset_name_list = args.dataset_name_list.split(",")
    atom_batch = False
    if args.dataset_micro_batch_size.startswith("atom:"):
        atom_batch = True
        args.dataset_micro_batch_size = args.dataset_micro_batch_size[5:]
    dataset_micro_batch_size = [int(i) for i in args.dataset_micro_batch_size.split(",")]
    dataset_sample_prob = [float(i) for i in args.dataset_sample_prob.split(",")]
    for sub_data_path in args.data_path_list.split(","):
        file_list.append(os.path.join(args.data_path, sub_data_path))
    
    if args.data_path_list_valid=="" or args.data_path_list_valid.lower()=="none":
        file_list_valid = ["" for _ in range(len(file_list))]
    else:
        for sub_data_path in args.data_path_list_valid.split(","):
            file_list_valid.append(
                os.path.join(args.data_path, sub_data_path)
            )
          
    assert (
            len(dataset_sample_prob) == len(dataset_name_list) and 
            len(dataset_sample_prob) == len(file_list_valid) and 
            len(dataset_sample_prob) == len(file_list) and 
            len(dataset_sample_prob) == len(dataset_micro_batch_size)
        ), f"split ratio mismatch with number of datasets,{len(dataset_sample_prob),len(dataset_name_list),len(file_list),len(file_list_valid),len(dataset_micro_batch_size)}"
      
    train_dataset_list = []
    valid_dataset_list = []
    test_dataset_list = []
    train_len = 0
    valid_len = 0
    test_len = 0
    
    for data_path, data_path_valid, dataset_name in zip(
        file_list, file_list_valid, dataset_name_list
    ):  
        if dataset_name in ["spice"]:
            train_dataset = SPICExyzDataset( data_path)
            if data_path_valid == "":data_path_valid=data_path
            valid_dataset = SPICExyzDataset( data_path_valid)
            train_dataset_list.append(train_dataset)
            valid_dataset_list.append(valid_dataset)
            train_len += len(train_dataset)
            valid_len += len(valid_dataset)
        elif dataset_name in ["toy"]:
            train_dataset,valid_dataset = TOYxyzDataset(""),TOYxyzDataset("")
            train_dataset_list.append(train_dataset)
            valid_dataset_list.append(valid_dataset)
            train_len += len(train_dataset)
            valid_len += len(valid_dataset)
        else:
            dataset = MoleculeLMDBDataset(
                data_path,
                data_name=dataset_name,
                remove_atomref_energy=remove_atomref_energy,
            )
            if data_path_valid:
                # TODO: test dataset
                train_dataset = dataset
                valid_dataset = MoleculeLMDBDataset(
                    data_path_valid,
                    data_name=dataset_name,
                    remove_atomref_energy=remove_atomref_energy,
                )
            else:
                shuffle = args.shuffle
                (
                    train_dataset,
                    valid_dataset,
                    test_dataset,
                    
                ) = dataset.split_train_valid_test(
                    args.train_val_test_split, shuffle=shuffle
                )
                test_dataset_list.append(test_dataset)
                test_len += len(test_dataset)
            train_dataset_list.append(train_dataset)
            valid_dataset_list.append(valid_dataset)
            train_len += len(train_dataset)
            valid_len += len(valid_dataset)
    logger.info(f"dataset train_len : {train_len,train_dataset_list}; \n dataset valid_len : {valid_len,valid_dataset_list};")
    logger.info(f"train data path list: {file_list},  valid data path list:{file_list_valid}")
    logger.info(f"train data length: {[len(i) for i in train_dataset_list]}, valid data length : {[len(i) for i in valid_dataset_list]}")

    is_distributed = torch.distributed.is_initialized()
    
    if atom_batch:
        train_dataset_final = ConcatDataset(train_dataset_list)
        valid_dataset_final = ConcatDataset(valid_dataset_list)

        if np.sum([d.atom_cnts.any() for d in train_dataset_list])!=len(train_dataset_list):
            raise ValueError(f"sorry ,dataset can not find atom cnts:{[d.atom_cnts.any() for d in train_dataset_list]}, thus atom level dataloaer issue happened")
        train_atom_cnts = [d.atom_cnts for d in train_dataset_list]
        valid_atom_cnts = [d.atom_cnts for d in valid_dataset_list]
        train_dataloader = make_atomloader_default(train_dataset_final,train_atom_cnts,dataset_micro_batch_size,dataset_sample_prob,True)
        valid_dataloader = make_atomloader_default(valid_dataset_final,valid_atom_cnts,dataset_micro_batch_size,dataset_sample_prob,False)
        logger.info(f"train dataset atom cnts:{train_atom_cnts}")

    else:
        train_dataset_final = ConcatDataset(train_dataset_list)
        valid_dataset_final = ConcatDataset(valid_dataset_list)
        train_dataset_lengths = [len(d) for d in train_dataset_list]
        valid_dataset_lengths = [len(d) for d in valid_dataset_list]
        train_dataloader = make_loader_default(train_dataset_final,train_dataset_lengths,dataset_micro_batch_size,dataset_sample_prob,True)
        valid_dataloader = make_loader_default(valid_dataset_final,valid_dataset_lengths,dataset_micro_batch_size,dataset_sample_prob,False)
            
        logger.info(f"is_distributed: {is_distributed}. train_dataloader {len(train_dataloader)}. valid_dataloader {len(valid_dataloader)}")
    return train_dataloader, valid_dataloader, None


def make_atomloader_default(dataset:ConcatDataset,lengths,micro_batch_size,dataset_sample_prob, is_train=True):
    if len(lengths)!=len(micro_batch_size):raise
    if torch.distributed.is_initialized():
        batch_sampler = MultiDatasetTokenBatchDistributedSampler( lengths, micro_batch_size,dataset_sample_prob,shuffle=is_train)
        
        # batch_sampler = TokenBatchDistributedSampler( lengths[0], micro_batch_size[0],shuffle=is_train)
    else:
        raise ValueError("Not support for a non ddp version")

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        
    )
    return loader


def make_loader_default(dataset,dataset_lengths,micro_batch_size,dataset_sample_prob, is_train):
    if torch.distributed.is_initialized():
        batch_sampler = MultiDatasetFixedBatchDistributedSampler(dataset_lengths, micro_batch_size,dataset_sample_prob,shuffle=is_train)
    else:
        batch_sampler = MultiDatasetFixedBatchDistributedSampler(dataset_lengths, micro_batch_size,dataset_sample_prob,num_replicas=1,rank=0,shuffle=is_train)

        # raise 

    loader = DataLoader(
        dataset,
        # batch_size=micro_batch_size,
        # sampler=sampler,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        
    )
    return loader
