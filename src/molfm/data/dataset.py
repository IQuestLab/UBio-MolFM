# -*- coding: utf-8 -*-
from functools import lru_cache

import sys

import numpy as np

import bisect
import glob
import os
import pickle as pkl
import random


def _install_numpy_pickle_compat():
    """Make numpy pickles readable across the numpy 1.x <-> 2.x module rename.

    numpy 2 moved ``numpy.core.*`` to ``numpy._core.*``; a pickle written by
    numpy>=2 therefore names ``numpy._core.numeric`` etc., which a numpy 1.26
    runtime cannot import (``ModuleNotFoundError``) -- it kills the DataLoader
    worker on the first ``pkl.loads`` of an LMDB sample. This
    installs ``sys.modules`` aliases in BOTH directions, and is a NO-OP whenever
    the real module already imports (i.e. on the numpy that owns that spelling),
    so it never shadows anything. Kept here, at import time of the module that
    defines MoleculeLMDBDataset, so every forked DataLoader worker inherits it.
    """
    import importlib
    for missing, present in (("numpy._core", "numpy.core"),   # numpy1 reading a numpy2 pickle
                             ("numpy.core", "numpy._core")):  # numpy2 reading a numpy1 pickle
        try:
            importlib.import_module(missing)
            parent_is_real = True
        except Exception:
            parent_is_real = False

        if not parent_is_real:
            # The whole package was renamed (numpy < 1.26 has no ``numpy._core``).
            # Alias the parent to the present package, and from here on alias the
            # submodules DIRECTLY: importing ``<missing>.<sub>`` after the parent
            # alias makes the import machinery resolve it through the present
            # package's __path__ and execute it a second time under the new name --
            # for ``_multiarray_umath.so`` that means loading the C extension twice,
            # a known segfault trigger.
            try:
                sys.modules[missing] = importlib.import_module(present)
            except Exception:
                continue  # neither spelling exists on this numpy; leave it alone

        for sub in ("numeric", "multiarray", "umath", "numerictypes", "overrides"):
            missing_name = missing + "." + sub
            if parent_is_real:
                # A partial shim (numpy 1.26 ships ``numpy._core`` but not all of
                # its children) is still a real package, so a failed child import
                # is harmless; keep any child that resolves natively.
                try:
                    importlib.import_module(missing_name)
                    continue
                except Exception:
                    pass
            try:
                sys.modules[missing_name] = importlib.import_module(present + "." + sub)
            except Exception:
                pass  # this numpy has neither spelling; leave it alone


_install_numpy_pickle_compat()

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
from loguru import logger

# from molfm.data.dataset import FoundationModelDataset, LMDBFoundationModelDataset
# from molfm.data.mol_data import algos
# from molfm.data.prot_data.util import bstr2obj

from molfm.data.hoh_o_mask import predict_hoh_o_mask
from molfm.data.utils import (

    get_data_defult_config,
    # matrixtoblock_lin,
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
    # pos = wrap_positions(pos, cell, pbc=pbc, eps=0)

    # wrap positions for CPU graph Generation
    # atoms.set_positions(pos)

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
        # data_src = input_atoms.info["data_id"] # e.g. biomolecules, elec**,
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
        lf_O_in_HOH: bool = False,
        ## 1 kcal/mol = 0.0433634 eV, transform to eV by default here
    ):
        super(MoleculeLMDBDataset, self).__init__()

        self.data_name = data_name
        self.lf_O_in_HOH = bool(lf_O_in_HOH)
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
        self._hoh_o_masks: list = []
        self._hoh_o_cum_per_shard: list = []
        self._hoh_o_warned = False
        self.db_paths = sorted(db_paths)
        self.open_db()
        # open_db runs in whichever process built the dataset -- with num_workers>0 that is
        # the parent, and every fork()ed worker inherits its handles. See _lmdb_env.
        self._env_pid = os.getpid()
        self.remove_atomref_energy = remove_atomref_energy
        self.conv, self.orbitals_ref, self.mask, self.chemical_symbols = (
            None,
            None,
            None,
            None,
        )


    def _lmdb_env(self, db_idx):
        """Return an ``lmdb.Environment`` for shard ``db_idx`` owned by THIS process.

        ``open_db`` runs once, in the process that constructs the dataset. With
        num_workers>0 the DataLoader forks (data/collator.py pins multiprocessing_context
        to "fork" because the dataset holds unpicklable Environment handles), so every
        worker inherits the parent's handles and reads plain .lmdb shards through them.

        WHY THIS IS A DEFECT REGARDLESS: an open lmdb.Environment is not documented as
        usable after fork(). readonly=True + lock=False is what made it look safe, but that
        only removes the shared reader table; it does not make the handle fork-safe. The
        .aselmdb branch never relied on it -- ase_db_backends.aselmdb.LMDBDatabase exposes
        `env` as a property that reopens whenever os.getpid() changes. This gives the
        plain-.lmdb branch the same per-pid behaviour: parity, not a new invention.

        STATUS OF THE EVIDENCE -- read before citing this as a fix.
        A DataLoader worker has been observed to die with
            RuntimeError: DataLoader worker (pid ...) is killed by signal: Segmentation fault
        ~25 s into training when num_workers>0. It has NOT been reproduced locally
        against this class: a loader-only run over the real MoleculeLMDBDataset at
        production shape -- 6 train + 4 valid branch datasets, 3,523 envs, 8 workers
        on each of a train and a valid loader, with this per-pid reopen DISABLED --
        completed 64,016 samples clean. A separate mock harness did segfault, but that
        is judged a defect of the mock rather than a reproduction of the production
        crash.

        So: a real latent fork-safety defect, inert at num_workers=0 (the pid never
        changes, so the guard is one integer compare), but whether it is THE cause of
        the observed crash is UNPROVEN. Confirming that needs a multi-rank run at
        num_workers>0; until one passes, do not treat num_workers>0 as safe.
        """
        pid = os.getpid()
        if pid != self._env_pid:
            # First access in a freshly fork()ed process: drop every inherited plain-lmdb
            # handle (the .aselmdb ones reopen themselves) and let them reopen lazily below,
            # so a worker only pays for the shards it actually touches.
            for i, ft in enumerate(self.file_type):
                if ft == 1:
                    self.envs[i] = None
            self._env_pid = pid
        env = self.envs[db_idx]
        if env is None:
            env = lmdb.open(
                str(self.db_paths[db_idx]),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=32,
            )
            self.envs[db_idx] = env
        return env

    def open_db(self):
        self.file_type = []
        for db_path in self.db_paths:

            shard_atom_cnts = None
            if db_path.endswith("aselmdb"):
                self.file_type.append(0)
                env = ase.db.connect(str(db_path),readonly=True)
                self.envs.append(env)
                # env.ids is a Python list of ints built by
                # ase_db_backends.aselmdb.LMDBDatabase._load_ids. Keep it as an int64
                # numpy array: 8 B/id instead of ~36 B/id (pointer + int object), and --
                # the reason this matters -- reading an element of a Python list does a
                # Py_INCREF, which dirties the page holding that int object. With
                # num_workers>0 the fork()ed workers would copy-on-write most of the id
                # store over an epoch (~8 GB/rank at 200 shards). A numpy array is read
                # without touching refcounts, so those pages stay shared.
                self._keys.append(np.asarray(env.ids, dtype=np.int64))
                # Prefer the external `<db>.atom_cnts.npy` sidecar (fast path,
                # produced by precompute_atom_cnts); fall back to the in-DB key.
                # Set shard_atom_cnts (not just self.atom_cnts) so the
                # lf_O_in_HOH branch below can align the HOH-O mask sidecar.
                sidecar_cnts = db_path + ".atom_cnts.npy"
                if os.path.exists(sidecar_cnts):
                    shard_atom_cnts = np.load(sidecar_cnts).astype(np.int32)
                    self.atom_cnts.append(shard_atom_cnts)
                else:
                    with env._get_txn(write=False) as txn:
                        atom_cnts = txn.get("atom_cnts".encode("ascii"))
                        if atom_cnts is not None:
                            shard_atom_cnts = np.array(
                                pkl.loads(atom_cnts), dtype=np.int32
                            )
                            self.atom_cnts.append(shard_atom_cnts)
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
                # Prefer the external `<db>.atom_cnts.npy` sidecar (fast path);
                # fall back to the in-DB key.
                sidecar_cnts = db_path + ".atom_cnts.npy"
                if os.path.exists(sidecar_cnts):
                    shard_atom_cnts = np.load(sidecar_cnts).astype(np.int32)
                    self.atom_cnts.append(shard_atom_cnts)
                else:
                    atom_cnts = self.envs[-1].begin().get("atom_cnts".encode("ascii"))
                    if atom_cnts is not None:
                        shard_atom_cnts = np.asarray(pkl.loads(atom_cnts), dtype=np.int32)
                        self.atom_cnts.append(shard_atom_cnts)
                # int64 array rather than list(range(length)) -- same reason as the
                # aselmdb branch above (fork-safe, refcount-free element reads).
                self._keys.append(np.arange(length, dtype=np.int64))

            if self.lf_O_in_HOH:
                if shard_atom_cnts is None:
                    self._hoh_o_masks.append(None)
                    self._hoh_o_cum_per_shard.append(None)
                    if not self._hoh_o_warned:
                        logger.warning(
                            f"lf_O_in_HOH=True but no in-DB atom_cnts for {db_path}; "
                            "cannot align HOH-O mask sidecar — falling back to zeros for this shard"
                        )
                        self._hoh_o_warned = True
                else:
                    prefix = np.concatenate(
                        [np.zeros(1, dtype=np.int64),
                         np.cumsum(shard_atom_cnts.astype(np.int64))]
                    )
                    self._hoh_o_cum_per_shard.append(prefix)
                    sidecar = db_path + ".hoh_o_mask.npy"
                    if os.path.exists(sidecar):
                        arr = np.load(sidecar, mmap_mode="r")
                        if arr.shape[0] != int(prefix[-1]):
                            raise ValueError(
                                f"HOH-O sidecar size mismatch for {sidecar}: "
                                f"got {arr.shape[0]}, expected {int(prefix[-1])} (sum of atom_cnts)"
                            )
                        self._hoh_o_masks.append(arr)
                    else:
                        self._hoh_o_masks.append(None)
                        if not self._hoh_o_warned:
                            logger.warning(
                                f"lf_O_in_HOH=True but {sidecar} missing; "
                                "falling back to all-zeros HOH-O mask for this shard"
                            )
                            self._hoh_o_warned = True

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
        # int(...) because self._keys holds int64 numpy arrays; the downstream key
        # encoders format the value into a string, so a plain int keeps that identical
        # across numpy versions.
        if self.file_type[db_idx] == 0: # aselmdb
            atomrow = self.envs[db_idx]._get_row(int(self._keys[db_idx][el_idx]))
            data_object = from_ase(atomrow)
        else:
            datapoint_pickled = (
                self._lmdb_env(db_idx)
                .begin()
                .get(f"{int(self._keys[db_idx][el_idx])}".encode("ascii"))
            )
            # from fairchem.core.common.utils import pyg2_data_transform

            # data_object = pyg2_data_transform(pkl.loads(datapoint_pickled))
            data_object = pkl.loads(datapoint_pickled)
        
        data_object.id = el_idx  # f"{db_idx}_{el_idx}"
        # print(data_object)

        if "energy" not in data_object:
            energy = data_object.y
        else:
            energy = data_object.energy
        if "force" in data_object:
            data_object["forces"] = data_object.force
        # atom_hist = np.zeros(128)
        # out["pyscf_energy"] = copy.deepcopy(energy.astype(np.float32))  # this is pyscf energy ground truth
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
        
        data_object.pos = data_object.pos - torch.mean(data_object.pos,dim=0,keepdim=True)
        out = {
            "pbc": self.pbc,
            # "sample_type": 0,
            "idx": idx,
            "pos": data_object.pos,
            "forces": data_object.forces * self.unit,
            "num_atoms": torch.tensor([natoms],dtype=torch.int),
            "atomic_numbers": data_object.atomic_numbers.int().reshape(-1),
            "energy": energy.reshape(-1) * self.unit,  # this is used from model training, mean/ref is removed.
            "energy_per_atom":energy.reshape(-1) * self.unit / data_object.pos.shape[0],
            
            "fixed":fixed,
            # "data_src":data_object.data_src,
            # "atom_hist":torch.from_numpy(atom_hist)/len(data_object.pos)*1.0,
            "data_name": self.data_name,
            "charge": charge.reshape(-1),
            "multiplicity": multiplicity.reshape(-1),
        }

        if self.lf_O_in_HOH:
            shard_mask = self._hoh_o_masks[db_idx] if self._hoh_o_masks else None
            shard_prefix = (
                self._hoh_o_cum_per_shard[db_idx] if self._hoh_o_cum_per_shard else None
            )
            if shard_mask is None or shard_prefix is None:
                hoh_o = torch.zeros(natoms, dtype=torch.bool)
            else:
                start = int(shard_prefix[el_idx])
                end = start + natoms
                hoh_o = torch.from_numpy(
                    np.array(shard_mask[start:end], dtype=np.uint8, copy=True)
                ).bool()
            out["hoh_o_mask"] = hoh_o

        if self.is_pbc:
            out.update({"cell": data_object.cell.reshape(3,3)})
        else:
            out["cell"] = torch.zeros((3, 3), dtype=torch.float32)

            # out["stress"] = torch.zeros(
            #     (3, 3), dtype=torch.float32, device=energy.device
            # )

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
                "data_src",
                "hoh_o_mask",
            ]:
                out[key] = out[key].clone().detach().float()

        return out



    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                # None = a handle _lmdb_env dropped after a fork and this process never
                # reopened; there is nothing to close.
                if env is not None and hasattr(env, "close"):
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


class SPICExyzDataset(torch.utils.data.Dataset):
    def __init__(self, path,
        data_name="spice", lf_O_in_HOH: bool = False, **kwargs) -> None:
        super().__init__()
        self.data_name = data_name
        # lf_O_in_HOH: emit a per-sample hoh_o_mask so the mixed-resolution backbone can
        # route water oxygens through the LIGHT branch. Unlike the LMDB datasets there is
        # no `<db>.hoh_o_mask.npy` sidecar for an xyz file, so the mask is computed here
        # with the same canonical rule (molfm.data.hoh_o_mask) the sidecar generator and
        # the ASE inference path use — training and inference must agree on which
        # oxygens are water, or the model is evaluated on a different graph.
        self.lf_O_in_HOH = bool(lf_O_in_HOH)
        # Geometries are static, so each frame pays the ~0.3 ms detection cost at most once
        # per process. Caching matters because the DataLoader runs with num_workers=0, i.e.
        # detection sits on the training step's critical path. Most frames contain no water
        # at all (~85% of SPICE), so those are recorded in a 1-byte-per-frame state array
        # and only the frames that do have water keep an array in `_hoh_o_cache`; a plain
        # dict over all 951k train frames would hold ~220 MB per rank.
        # State: 0 = not computed, 1 = known water-free, 2 = mask in _hoh_o_cache.
        self._hoh_o_cache: dict = {}
        self._hoh_o_state: Optional[np.ndarray] = None
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
        if self.lf_O_in_HOH:
            self._hoh_o_state = np.zeros(self.num_samples, dtype=np.uint8)
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
                
                # "subset_name": self.subset_name2id[subset_name],
                # "forces_subset_name": torch.zeros(len(atomic_numbers))
                # + self.subset_name2id[subset_name],

            }
        )

        if self.lf_O_in_HOH:
            out["hoh_o_mask"] = self._get_hoh_o_mask(idx, atoms, atomic_numbers)

        return out

    def _get_hoh_o_mask(self, idx, atoms, atomic_numbers):
        """Per-frame HOH-water-oxygen mask (bool tensor of length N), cached by index."""
        natoms = len(atomic_numbers)
        state = int(self._hoh_o_state[idx])
        if state == 1:
            return torch.zeros(natoms, dtype=torch.bool)
        if state == 2:
            return torch.from_numpy(self._hoh_o_cache[idx]).bool()

        Z = np.asarray(atomic_numbers, dtype=np.int64)
        if natoms < 3 or not (Z == 8).any() or int((Z == 1).sum()) < 2:
            # Exact short-circuit: a 3-atom {O,H,H} fragment needs >=1 O and >=2 H.
            mask = np.zeros(natoms, dtype=np.uint8)
        else:
            # SPICE frames are aperiodic (pbc/cell are emitted as zeros above), so this
            # takes the cKDTree branch — the same one the omol25 `.hoh_o_mask.npy`
            # sidecars were generated with. Positions are used raw; the +0.1 shift applied
            # to `pos` is a rigid translation and cannot change interatomic distances.
            mask = predict_hoh_o_mask(atoms.get_positions(), Z)

        if mask.any():
            self._hoh_o_cache[idx] = mask
            self._hoh_o_state[idx] = 2
        else:
            self._hoh_o_state[idx] = 1
        return torch.from_numpy(mask).bool()

    def __len__(self) -> int:
        return self.num_samples
