# UBio-MolFM: Universal Bio-Molecular Foundation Model

> **Note**: This is the latest repository of **UBio-MolFM**, which also includes code and research from previous versions. 
> - **E2Former-LSR**: Scalable Machine Learning Force Fields through Long-Range Aware Message Passing ([arXiv:2601.03774](https://arxiv.org/abs/2601.03774) | [Code Release: E2Former-LSR](https://github.com/IQuestLab/UBio-MolFM/releases/tag/E2Former-LSR))
> - **E2Former-V2**: On-the-Fly Equivariant Attention with Linear Activation Memory ([arXiv:2601.16622](https://arxiv.org/abs/2601.16622) | [Code Release: E2Former-V2](https://github.com/IQuestLab/UBio-MolFM/releases/tag/E2Former-V2))
> - **UBio-MolFM**: Universal Molecular Foundation Model for Bio-Systems ([arXiv:2602.17709](https://arxiv.org/abs/2602.17709)) | Current repository

UBio-MolFM is a foundation model suite for molecular modeling, developed by the UBio-MolFM team. This repository provides the implementation of UBio-MolFM models, including the **E2Former-V2** backbone, together with tools for training, inference, and molecular dynamics (MD) simulation.

## Quick Configuration

The project targets Python 3.12 and PyTorch 2.7.0. We recommend `mamba` or `conda` for environment management.

```bash
# Create the environment
mamba env create -f install/environment_py312.yaml

# Activate the environment
conda activate mfm
```

Repository layout at a glance:

| Path | Description |
| --- | --- |
| `config_file/` | Model and training configurations in YAML format. |
| `src/molfm/models/` | Molecular backbone implementations, including `e2former`. |
| `src/molfm/interface/` | Integration layers for ASE and other external tools. |
| `src/molfm/pipeline/` | Training and validation orchestration. |
| `src/molfm/tasks/` | Task-specific entry points, such as pretraining and inference. |
| `src/molfm/scripts/` | Shell scripts for common training stages, including OC20, SPICE, and Omol25. |

## Quick Use of Pretrained Models

Pretrained checkpoints and high-precision datasets are available on Hugging Face:

- [IQuest-UBio-MolFM-V1](https://huggingface.co/IQuestLab/IQuest-UBio-MolFM-V1): Pretrained checkpoints.
- [UBio-Protein26](https://huggingface.co/datasets/IQuestLab/UBio-Protein26): 5 million high-precision protein DFT dataset (a subset of UBio-Mol26).

Use the downloaded checkpoint together with its matching configuration file. By default, the inference interface resolves the configuration from the checkpoint directory when `config_name` is not an absolute path.

This pretrained checkpoint is built on the UBio-MolFM framework, trained with the **UBio-Mol26** (17M) bio-specific dataset + OMol25, the E2Former-V2 linear-scaling equivariant transformer, and a three-stage curriculum learning strategy. It has been evaluated for accuracy on high-fidelity DFT datasets at the 1,500-atom scale, and it can support single-GPU simulations of up to approximately 100,000 atoms, depending on available device memory.

### Example 1: Single-Point Energy and Force Prediction

```python
from ase.build import molecule
from molfm.interface.ase.calculator.e2former_calculator import E2FormerCalculator

atoms = molecule("H2O")
atoms.set_cell([10, 10, 10])
atoms.pbc = [True, True, True]

calc = E2FormerCalculator(
    checkpoint_path="path/to/checkpoint.pt",
    config_name="config_molfm.yaml",
    head_name="omol25",
    device="cuda",
    use_tf32=True,
    use_compile=True,
)

atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(f"Energy: {energy} eV")
print(f"Forces:\n{forces}")
```

### Example 2: Molecular Dynamics with ASE

```python
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = Langevin(atoms, 1 * units.fs, temperature_K=300, friction=0.01)
dyn.run(100)
```

### Performance Notes

- `use_tf32=True` enables TensorFloat-32 on supported NVIDIA GPUs. It typically improves throughput, but it may introduce a small loss in numerical precision.
- `use_compile=True` enables `torch.compile`. It typically improves execution speed and may reduce memory usage in some workloads.
- The first run may take longer because the backend needs to initialize kernels and compilation artifacts.

## Training

UBio-MolFM training is launched with `torchrun`. Example scripts are available under `src/molfm/scripts/`.

### Data Preparation

UBio-MolFM supports three data entry modes:

- **LMDB datasets**: the loader recursively scans a directory for `.lmdb` or `.aselmdb` files.
- **SPICE xyz files**: set `dataset_name_list=spice` and point `data_path_list` and `data_path_list_valid` to `.xyz` files.
- **Toy data**: set `dataset_name_list=toy` for synthetic smoke tests.

The training scripts interpret `data_path_list` and `data_path_list_valid` relative to `data_path`. The shipped shell scripts use the repository `data/` directory as the root by default.

Recommended layout:

```text
data/
  omol25/
    train/
    val_1M/
  SPICE/
    train_large_neut_no_bad_clean.xyz
    test_large_neut_all.xyz
  OC20/
    s2ef/
      2M/train/
      all/val_id/
```

Example configurations:

```bash
# Omol25
data_path=./data
data_path_list=omol25/train
data_path_list_valid=omol25/val_1M
dataset_name_list=omol25

# SPICE
data_path=./data
data_path_list=SPICE/train_large_neut_no_bad_clean.xyz
data_path_list_valid=SPICE/test_large_neut_all.xyz
dataset_name_list=spice

# OC20
data_path=./data
data_path_list=OC20/s2ef/2M/train
data_path_list_valid=OC20/s2ef/all/val_id
dataset_name_list=oc20
```

If `data_path_list_valid` is omitted for an LMDB dataset, UBio-MolFM falls back to the `train_val_test_split` defined in the configuration file, which defaults to `0.95 / 0.05 / 0.0`.

### Launching Training

Example: pretraining on Omol25

```bash
bash src/molfm/scripts/pretrain_omol25_stage1.sh
```

Equivalent `torchrun` entry point:

```bash
torchrun --nproc_per_node 8 src/molfm/tasks/pretrain_smallmol.py \
    --config-name=config_molfm.yaml \
    backbone=e2former \
    backbone_config=e2former \
    strategy_config=ddp \
    save_dir=./outputs/ckpt_log/
```

For training, the most frequently tuned options are:

- `save_dir`: output directory for checkpoints and logs.
- `dataset_micro_batch_size`: per-device micro-batch size.
- `gradient_accumulation_steps`: effective batch size scaling.
- `use_tf32`: enable TensorFloat-32 for faster execution on supported GPUs.
- `use_compile`: enable `torch.compile` for faster execution and possible memory savings.

## Related Work

For more details on the methodology and architecture, see the associated papers:

- **UBio-MolFM: A Universal Molecular Foundation Model for Bio-Systems** (2026): [arXiv:2602.17709](https://arxiv.org/abs/2602.17709)
- **E2Former-V2: On-the-Fly Equivariant Attention with Linear Activation Memory** (2026): [arXiv:2601.16622](https://arxiv.org/abs/2601.16622)
- **Scalable Machine Learning Force Fields for Macromolecular Systems Through Long-Range Aware Message Passing (E2Former-LSR)** (2026): [arXiv:2601.03774](https://arxiv.org/abs/2601.03774)
- **E2Former: An Efficient and Equivariant Transformer with Linear-Scaling Tensor Products** (2025): [arXiv:2501.19216](https://arxiv.org/abs/2501.19216)

## Citation

The `Related Work` section lists the main papers and release references, while this section provides the recommended citation format for academic use.

If you use UBio-MolFM or E2Former-V2 in your research, please cite the following:

```bibtex
@misc{huang2026ubiomolfm,
      title={UBio-MolFM: A Universal Molecular Foundation Model for Bio-Systems}, 
      author={Lin Huang and Arthur Jiang and XiaoLi Liu and Zion Wang and Jason Zhao and Chu Wang and HaoCheng Lu and ChengXiang Huang and JiaJun Cheng and YiYue Du and Jia Zhang},
      year={2026},
      eprint={2602.17709},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph}
}

@misc{huang2026e2formerv2,
      title={E2Former-V2: On-the-Fly Equivariant Attention with Linear Activation Memory}, 
      author={Lin Huang and Chengxiang Huang and Ziang Wang and Yiyue Du and Chu Wang and Haocheng Lu and Yunyang Li and Xiaoli Liu and Arthur Jiang and Jia Zhang},
      year={2026},
      eprint={2601.16622},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{wang2026scalable,
      title={Scalable Machine Learning Force Fields for Macromolecular Systems Through Long-Range Aware Message Passing}, 
      author={Chu Wang and Lin Huang and Xinran Wei and Tao Qin and Arthur Jiang and Lixue Cheng and Jia Zhang},
      year={2026},
      eprint={2601.03774},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph}
}

@misc{li2025e2former,
      title={E2Former: An Efficient and Equivariant Transformer with Linear-Scaling Tensor Products}, 
      author={Yunyang Li and Lin Huang and Zhihao Ding and Chu Wang and Xinran Wei and Han Yang and Zun Wang and Chang Liu and Yu Shi and Peiran Jin and Tao Qin and Mark Gerstein and Jia Zhang},
      year={2025},
      eprint={2501.19216},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

This repository is released under the MIT License.
