#!/usr/bin/env bash

# Licensed under the MIT License.
#
# Standard, stage 1: direct-force training on OMol25.
# Architecture: 4 TransBlocks, max_radius=6, interleaved routing, light 64x0e.
#
#   bash src/molfm/scripts/omol25/pretrain_e2former_v3_standard_stage1.sh
#
# Override data paths, GPU count, step budget, etc. from the environment:
#   DATA_ROOT=/path/to/omol25 total_num_steps=1000 n_gpu=4 \
#       bash src/molfm/scripts/omol25/pretrain_e2former_v3_standard_stage1.sh

ulimit -c unlimited
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/omol25}"
data_path="${data_path:-${DATA_ROOT}}"
data_path_list="${data_path_list:-${DATA_ROOT}/train}"
data_path_list_valid="${data_path_list_valid:-${DATA_ROOT}/val_1M}"
dataset_name_list="${dataset_name_list:-omol25}"
dataset_sample_prob="${dataset_sample_prob:-1}"
dataset_micro_batch_size="${dataset_micro_batch_size:-atom:2048}"
num_workers="${num_workers:-8}"
n_gpu="${n_gpu:-$(nvidia-smi -L | wc -l)}"
total_num_steps="${total_num_steps:-1400000}"
save_dir="${save_dir:-./outputs/ckpt_log/}"
experiment_name="${experiment_name:-standard_stage1}"
save_batch_interval="${save_batch_interval:-5000}"
# Keep > 1: pretrain_smallmol.py treats val_batch_interval <= 1 as
# "validate only" and skips training entirely.
val_batch_interval="${val_batch_interval:-20000}"
log_interval="${log_interval:-100}"

# Single-node by default. For multi-node, set NNODES, NODERANK and MASTER_ADDR.
NNODES="${NNODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-62361}"
if [ "${NNODES}" -gt 1 ]; then
    : "${NODERANK:?Set NODERANK when NNODES > 1}"
    DISTRIBUTED_ARGS="--nproc_per_node ${n_gpu} --nnodes ${NNODES} \
        --node_rank ${NODERANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"
else
    DISTRIBUTED_ARGS="--nproc_per_node ${n_gpu} --master_port ${MASTER_PORT}"
fi

torchrun ${DISTRIBUTED_ARGS} src/molfm/tasks/pretrain_smallmol.py \
    --config-name=config_molfm.yaml \
    AutoGradForce=False \
    loss_fn='atoml2mae' \
    lr=0.0008 \
    weight_decay=0.001 \
    energy_loss_weight=4 force_loss_weight=30 \
    clip_grad_norm=100 \
    total_num_steps="${total_num_steps}" \
    val_batch_interval="${val_batch_interval}" \
    save_batch_interval="${save_batch_interval}" \
    log_interval="${log_interval}" \
    ifresume=False \
    ckpt_path="none" \
    wandb=False swanlab=False tb=False \
    save_dir="${save_dir}" \
    experiment_name="${experiment_name}" \
    num_workers="${num_workers}" \
    gradient_accumulation_steps=1 \
    scheduler_config=groupWarmupDecayLR \
    scheduler_config.warmup_steps=4000 \
    strategy_config=ddp \
    with_moe=painn \
    backbone=e2former \
    backbone_config=e2former \
    backbone_config.with_cluster=False \
    backbone_config.num_layers=4 \
    backbone_config.irreps_node_embedding="256x0e+256x1e+256x2e" \
    backbone_config.irreps_head="2x0e+2x1e+2x2e" \
    backbone_config.attn_scalar_head=8 \
    backbone_config.num_attn_heads=128 \
    backbone_config.number_of_basis=256 \
    backbone_config.max_radius=6 \
    backbone_config.max_neighbors=None \
    backbone_config.alpha_drop=0.0 \
    backbone_config.basis_type='gaussiansmear' \
    backbone_config.norm_layer='rms_norm_sh' \
    backbone_config.attn_type='zero-order' \
    backbone_config.tp_type='QKdotS_alpha+triton' \
    backbone_config.edge_embedtype='default' \
    backbone_config.ffn_type='s2' \
    backbone_config.ffn_grid_resolution=12 \
    backbone_config.encoder='default' \
    backbone_config.mixed_resolution.enabled=True \
    backbone_config.mixed_resolution.irreps_light_embedding="64x0e" \
    backbone_config.mixed_resolution.lf_atoms=[1] \
    backbone_config.mixed_resolution.light_attn_heads=4 \
    backbone_config.mixed_resolution.light_attn_scalar_head=16 \
    backbone_config.mixed_resolution.cross_mode="interleaved" \
    "backbone_config.mixed_resolution.cross_mode_list=[A2H,H2A,A2H,H2A]" \
    backbone_config.mixed_resolution.interleaved_tail=v1 \
    backbone_config.mixed_resolution.light_angular_basis=0 \
    backbone_config.mixed_resolution.light_angular_zeta_cnt=4 \
    backbone_config.mixed_resolution.light_angular_eta=1.0 \
    backbone_config.mixed_resolution.energy_head="distinct" \
    lf_O_in_HOH=False \
    data_path="${data_path}" \
    data_path_list="${data_path_list}" \
    dataset_name_list="${dataset_name_list}" \
    dataset_sample_prob="${dataset_sample_prob}" \
    dataset_micro_batch_size="${dataset_micro_batch_size}" \
    data_path_list_valid="${data_path_list_valid}"
