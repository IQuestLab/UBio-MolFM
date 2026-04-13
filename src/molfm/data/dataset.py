# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np

import bisect
import glob
import os
import pickle as pkl
import random

from typing import Any, List, Optional, Union

import lmdb
import torch
from ase.io import read as ase_read
import ase
import ase.db

from numpy import dot
from numpy.linalg import norm
from rdkit import Chem
from sympy.utilities.iterables import multiset_permutations
from torch.utils.data import Subset
from torch_cluster import radius_graph
from torch_geometric.data import Data
from tqdm import tqdm

from abc import ABC, abstractmethod


from molfm.data.utils import (
    get_data_defult_config,
)

_PT = Chem.GetPeriodicTable()

def replace_func(*args,**kwargs):
    return args,kwargs

def from_ase(atomrow,target_dtype = torch.float32):
    input_atoms = atomrow.toatoms()
    input_atoms.info.update(atomrow.data)
    calc = input_atoms.calc
    atoms = input_atoms.copy()
    atomic_numbers = np.array(atoms.get_atomic_numbers(), copy=True)
    pos = np.array(atoms.get_positions(), copy=True)
    pbc = np.array(atoms.pbc, copy=True)
    cell = np.array(atoms.get_cell(complete=True), copy=True)

    atomic_numbers = torch.from_numpy(atomic_numbers).long()
    pos = torch.from_numpy(pos).to(target_dtype)
    pbc = torch.from_numpy(pbc).bool().view(1, 3)
    cell = torch.from_numpy(cell).to(target_dtype).view(1, 3, 3)
    num_atoms = torch.tensor([pos.shape[0]], dtype=torch.long)


    tags = torch.from_numpy(atoms.get_tags()).long()


    # TODO another way to specify this is to spcify a key. maybe total_charge
    charge = torch.LongTensor([atoms.info.get("charge", 0)])
    multiplicity = torch.LongTensor([atoms.info.get("spin", 1)])
    energy = input_atoms.calc.results["energy"] # use the default data type.
    forces = torch.from_numpy(input_atoms.calc.results["forces"]).to(target_dtype)
    
    return Data(
        pos=pos,
        atomic_numbers=atomic_numbers,     # atomic numbers
        num_atoms=num_atoms,        # number of atoms
        cell=cell,
        pbc=pbc,
        charge=charge,
        multiplicity=multiplicity,
        energy=energy,
        forces=forces,
        tags=tags,
    )


class MoleculeLMDBDataset(ABC):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.
    Useful for Structure to Energy & Force (S2EF), Initial State to·
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.
    Args:
            @path: the path to store the data
            @task: the task for the lmdb dataset
            @split: splitting of the data
            @transform: some transformation of the data
    """

    energy = "energy"
    forces = "forces"

    def __init__(
        self,
        path,
        data_name="default",
        remove_atomref_energy=True,
        ## 1 kcal/mol = 0.0433634 eV, transform to eV by default here
    ):
        super(MoleculeLMDBDataset, self).__init__()

        self.data_name = data_name
        (
            self.atom_reference,
            self.system_ref,
            self.train_ratio,
            self.val_ratio,
            self.test_ratio,
            self.has_energy,
            self.has_forces,
            self.pbc,
            self.unit,
            (self.energy_std,self.force_std),
        ) = get_data_defult_config(data_name)
        self.is_pbc = sum(self.pbc) > 0
        self.pbc = torch.Tensor(self.pbc).bool()
        db_paths = []
        if isinstance(path, str):
            path = [path]


        for p in path:
            if p.endswith("lmdb"):
                db_paths.append(p)
            else:
                for dirpath, dirnames, filenames in os.walk(p):
                    # if filenames!=[]:
                    for f in filenames:
                        if f.endswith("lmdb"):
                            file_path = os.path.join(dirpath, f)
                            db_paths.append(file_path)
                            # db_paths.extend(glob.glob(p + "/*lmdb"))
        # print(db_paths)
        assert len(db_paths) > 0, "No LMDBs found"
        self._keys, self.envs = [], []
        self.atom_cnts = []
        self.db_paths = sorted(db_paths)
        self.open_db()
        self.remove_atomref_energy = remove_atomref_energy
        self.conv, self.orbitals_ref, self.mask, self.chemical_symbols = (
            None,
            None,
            None,
            None,
        )


    def open_db(self):
        self.file_type = []
        for db_path in self.db_paths:
            
            if db_path.endswith("aselmdb"):
                self.file_type.append(0)
                env = ase.db.connect(str(db_path),readonly=True)
                self.envs.append(env)
                self._keys.append(env.ids)
                with env._get_txn(write=False) as txn:
                    atom_cnts = txn.get("atom_cnts".encode("ascii"))
                    if atom_cnts is not None:
                        self.atom_cnts.append( np.array(
                            pkl.loads(atom_cnts),dtype=np.int32
                            ))
            elif db_path.endswith("lmdb"):
                self.file_type.append(1)
                env = lmdb.open(
                    str(db_path),
                    subdir=False,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=32,
                )
                self.envs.append(env)
                length = self.envs[-1].begin().get("length".encode("ascii"))
                if length is not None:
                    length = pkl.loads(length)
                else:
                    length = self.envs[-1].stat()["entries"]
                atom_cnts = self.envs[-1].begin().get("atom_cnts".encode("ascii"))
                if atom_cnts is not None:
                    self.atom_cnts.append(pkl.loads(atom_cnts))
                self._keys.append(list(range(length)))

        keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(keylens).tolist()
        if self.atom_cnts != []:
            self.atom_cnts = np.concatenate(self.atom_cnts,axis = 0)
        else:
            self.atom_cnts = np.zeros(0)
        self.num_samples = sum(keylens)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # idx = 3558617 
        db_idx = bisect.bisect(self._keylen_cumulative, idx)
        # Extract index of element within that db.
        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self._keylen_cumulative[db_idx - 1]
        assert el_idx >= 0

        # Return features.
        if self.file_type[db_idx] == 0: # aselmdb
            atomrow = self.envs[db_idx]._get_row(self._keys[db_idx][el_idx])
            data_object = from_ase(atomrow)
        else:
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pkl.loads(datapoint_pickled)
        
        data_object.id = el_idx  # f"{db_idx}_{el_idx}"

        if "energy" not in data_object:
            energy = data_object.y
        else:
            energy = data_object.energy
        if "force" in data_object:
            data_object["forces"] = data_object.force
        if self.remove_atomref_energy:
            unique, counts = np.unique(
                data_object.atomic_numbers.int().numpy(), return_counts=True
            )
            # atom_hist[unique] = counts
            energy = energy - np.sum(self.atom_reference[unique] * counts)
            energy = torch.Tensor([energy - self.system_ref])
        energy = torch.Tensor([energy]).reshape(-1)

        if "charge" in data_object.keys():
            charge = data_object.charge.int()
        else:
            charge = torch.tensor([0], dtype=torch.int)

        if "multiplicity" in data_object.keys():
            multiplicity = data_object.multiplicity.int()
        else:
            multiplicity = torch.tensor([1], dtype=torch.int)
        
        natoms = data_object.pos.shape[0]
        if "fixed" in data_object.keys():
            fixed = data_object["fixed"]
        else:
            fixed = torch.zeros(natoms, dtype=torch.int)
        
        if "cos" in data_object.keys():
            cos = data_object["cos"]
        else:
            cos = torch.ones(natoms)
        
        data_object.pos = data_object.pos - torch.mean(data_object.pos,dim=0,keepdim=True)
        out = {
            "pbc": self.pbc,
            "cos":cos,
            "idx": idx,
            "pos": data_object.pos,
            "forces": data_object.forces * self.unit,
            "num_atoms": torch.tensor([natoms],dtype=torch.int),
            "atomic_numbers": data_object.atomic_numbers.int().reshape(-1),
            "energy": energy.reshape(-1) * self.unit,  # this is used from model training, mean/ref is removed.
            "energy_per_atom":energy.reshape(-1) * self.unit / data_object.pos.shape[0],
            "fixed":fixed,
            "data_name": self.data_name,
            "charge": charge.reshape(-1),
            "multiplicity": multiplicity.reshape(-1),
        }

        if self.is_pbc:
            out.update({"cell": data_object.cell.reshape(3,3)})
        else:
            out["cell"] = torch.zeros((3, 3), dtype=torch.float32)

        for key in out.keys():
            if key not in [
                "num_atoms",
                "atomic_numbers",
                "idx",
                "edge_index",
                "has_energy",
                "has_forces",
                "has_stress",
                "sample_type",
                "data_name",
                "charge",
                "multiplicity",
                "data_src"
            ]:
                out[key] = out[key].clone().detach().float()

        return out



    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                if hasattr(env, "close"):
                    env.close()
            self.envs = []
        else:
            if hasattr(self.env, "close"):
                self.env.close()
            self.env = None

    
    def split_train_valid_test(self, ratio_list: list, shuffle=True):
        num_samples = self.num_samples

        indices = list(range(num_samples))
        # Shuffle the indices and split them into training and validation sets
        if shuffle:
            random.Random(12345).shuffle(indices)

        num_validation_samples = int(num_samples * ratio_list[1])
        num_test_samples = int(num_samples * ratio_list[2])
        num_training_samples = num_samples - num_validation_samples - num_test_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[
            num_training_samples : num_training_samples + num_validation_samples
        ]
        test_indices = indices[num_training_samples + num_validation_samples - 1 :]
        
        dataset_train = Subset(self, training_indices)
        dataset_val = Subset(self, validation_indices)
        dataset_test = Subset(self, test_indices)
        if self.atom_cnts.any():
            dataset_train.atom_cnts = self.atom_cnts[training_indices]
            dataset_val.atom_cnts = self.atom_cnts[validation_indices]
            dataset_test.atom_cnts = self.atom_cnts[test_indices]
        else:
            dataset_train.atom_cnts = self.atom_cnts
            dataset_val.atom_cnts = self.atom_cnts
            dataset_test.atom_cnts = self.atom_cnts

        return dataset_train, dataset_val, dataset_test


class TOYxyzDataset():
    def __init__(self, config=None, path = None,transform=None,system_size = 8000,cell_size = None,device = None,**args) -> None:
        super().__init__()
        if isinstance(system_size,int):
            system_size = [system_size,]

        self.system_size = system_size

        self.cell_size = None if cell_size is None else torch.Tensor(cell_size).cpu()
        self.device = device

    def __getitem__(self, idx):

        N = self.system_size[idx]
        rho = 0.8
        volume = N / rho
        box_length = volume ** (1/3)
        if self.cell_size is None:
            # Generate N random 3D positions in the box
            pos = torch.rand((N, 3))*box_length
            pbc = torch.zeros(3)
            cell = torch.zeros(1, 3, 3)
        else:
            pos = torch.rand((N, 3))@self.cell_size
            pbc = torch.ones(3)
            cell = self.cell_size.reshape(3,3)
        atomic_numbers = torch.randint(1,50,(N,))
        forces = torch.randn((N, 3))
        energy = torch.randn(1)
        

        out = Data(
            **{
                "data_name":"toy",
                "idx": idx,

                
                "atomic_numbers": atomic_numbers,
                "pos": pos.float(),
                "num_atoms": torch.tensor([N],dtype=torch.int),

                "pbc":pbc,
                "cell":cell,
                "charge": torch.tensor([0], dtype=torch.int).reshape(-1),
                "multiplicity": torch.tensor([1], dtype=torch.int).reshape(-1),
                
                "energy": energy.reshape(1).float(),
                "energy_per_atom": energy.reshape(1).float()/N,
                "forces": forces.float(),

            }
        )
        for key in out.keys():
            if isinstance(out[key],torch.Tensor):
                out[key] = out[key].to(self.device)
        return out
    
    def __len__(self,):
        return len(self.system_size)

class SPICExyzDataset(torch.utils.data.Dataset):
    def __init__(self, path,
        data_name="spice",**kwargs) -> None:
        super().__init__()
        self.data_name = data_name
        if path is None:
            self.paths = []
            assert (
                len(self.paths) == 1
            ), f"{type(self)} does not support a list of src paths."
            self.path = self.paths[0]

        else:
            self.path = path
        self.atoms_list = ase.io.read(self.path, index=":")

        self.atom_reference = np.array(
            [
                0.00000000e00,
                -1.63841057e01,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -1.03758911e03,
                -1.49072485e03,
                -2.04814404e03,
                -2.71846191e03,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -9.29155469e03,
                -1.08369053e04,
                -1.25243545e04,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -7.00463750e04,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -8.10290869e03,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ]
        )
        self.num_samples = len(self.atoms_list)
        self.subset_name2id = {
            "PubChem": torch.tensor([0],dtype = torch.int),  # 34093,
            "DES370K Monomers": torch.tensor([1],dtype = torch.int),  # 889,
            "DES370K Dimers": torch.tensor([2],dtype = torch.int),  # 13908,
            "Dipeptides": torch.tensor([3],dtype = torch.int),  # 1025,
            "Solvated Amino Acids": torch.tensor([4],dtype = torch.int),  # 52
            "water": torch.tensor([5],dtype = torch.int),  # 84,
            "QMugs": torch.tensor([6],dtype = torch.int),  # 144,
        }

    def __getitem__(self, idx):
        atoms = self.atoms_list[idx]
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        atomic_numbers = atoms.get_atomic_numbers()
        pos = atoms.get_positions() + 0.1
        subset_name = atoms.info["config_type"]

        charge = atoms.info.get("charge",0)
        if charge is None:
            charge = float(np.sum(atoms.get_initial_charges()))
        charge = int(round(charge))

        multiplicity = atoms.info.get("spin",1)
        
        unique, counts = np.unique(atomic_numbers, return_counts=True)
        energy = energy - np.sum(self.atom_reference[unique] * counts)
        out = Data(
            **{
                "data_name":self.data_name,
                "data_src": self.subset_name2id[subset_name],
                "idx": idx,
                "pos": torch.from_numpy(pos).float(),

                "atomic_numbers": torch.from_numpy(atomic_numbers),
                "num_atoms": torch.tensor([len(atomic_numbers)], dtype=torch.int),

                "energy": torch.from_numpy(energy.reshape(1)).float(),
                "energy_per_atom":torch.from_numpy(energy.reshape(1)).float() / len(atomic_numbers),
                "forces": torch.from_numpy(forces).float(),

                "fixed" : torch.zeros(len(atomic_numbers), dtype=torch.int),
                "pbc": torch.zeros(3),
                "cell": torch.zeros(1, 3, 3),
                "charge": torch.tensor([charge], dtype=torch.int).reshape(-1),
                "multiplicity": torch.tensor([multiplicity], dtype=torch.int).reshape(-1),
            }
        )
        
        return out
    
    def __len__(self) -> int:
        return self.num_samples
