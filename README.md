E2Former-V2: On-the-Fly Equivariant Attention with Linear Activation Memory
===========================================================================

This repository contains the code for **E2Former-V2: On-the-Fly Equivariant Attention with Linear Activation Memory**.
Paper: `https://arxiv.org/abs/2601.16622`

## 1) Environment

- OS: Linux (recommended)
- Python: 3.12
- GPU: NVIDIA + CUDA (CUDA 12.x driver recommended; this repo uses cu126 wheels)

Note:
- Training scripts are executed directly (e.g. `src/molfm/tasks/*.py`). The scripts add `src/` to `sys.path`, so an editable install is usually not required.

## 2) Installation

From this `MolFM/` directory:

```bash
# 1) Create the environment
conda env create -f install/environment_mfm.yaml -n mfm
conda activate mfm

# 2) Install fairchem-core without pulling deps (avoids torch conflicts)
pip install fairchem-core==2.4.0 --no-deps

# 3) Optional extras
pip install "faiss-gpu-cu12==1.8.0.2"
```

Quick sanity check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch_geometric, lmdb, rdkit, hydra; print(ok)"
```

## 3) Template Usage

### 3.1 Pretrain/Fine-tune launcher: `pretrain_template.sh`

Template script: `src/molfm/scripts/pretrain_template.sh`

Recommended workflow (copy, then fill placeholders):

```bash
cp src/molfm/scripts/pretrain_template.sh pretrain.sh
vim pretrain.sh
bash pretrain.sh
```

You typically must set at least:
- `data_path_list`: training data path (directory or explicit `*.lmdb`; the loader recursively discovers `*.lmdb`)
- `data_path_list_valid`: validation data path (can be empty)
- `dataset_name_list`: dataset name (controls default unit/ref settings; see `molfm.data.utils.get_data_defult_config`)
- `experiment_name`: experiment name (outputs go to `save_dir/experiment_name/`)



## 4) Outputs

The training script writes into `save_dir/experiment_name/`:
- `pretrain_config.yaml`: the fully resolved config used for this run
- `logging_*.txt`: logs
- `checkpoint_E*_B*.pt`: checkpoints saved by interval (controlled by `save_batch_interval`)

## Citation

If you find this work useful, please cite:

```bibtex
@misc{huang2026e2formerv2ontheflyequivariantattention,
      title={E2Former-V2: On-the-Fly Equivariant Attention with Linear Activation Memory},
      author={Lin Huang and Chengxiang Huang and Ziang Wang and Yiyue Du and Chu Wang and Haocheng Lu and Yunyang Li and Xiaoli Liu and Arthur Jiang and Jia Zhang},
      year={2026},
      eprint={2601.16622},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.16622},
}
```
