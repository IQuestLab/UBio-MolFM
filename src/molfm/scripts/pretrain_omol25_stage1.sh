#!/usr/bin/env bash

# Licensed under the MIT License.
ulimit -c unlimited

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

[ -z "${data_path}" ] && data_path="${PROJECT_ROOT}/data"
[ -z "${data_path_list}" ] && data_path_list="${PROJECT_ROOT}/data/omol25/train_4M"
[ -z "${data_path_list_valid}" ] && data_path_list_valid="${PROJECT_ROOT}/data/omol25/val_1M"
[ -z "${dataset_name_list}" ] && dataset_name_list="omol25"
[ -z "${dataset_sample_prob}" ] && dataset_sample_prob='1'
[ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="32"
[ -z "${use_unified_batch_sampler}" ] && use_unified_batch_sampler=True
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${total_num_steps}" ] && total_num_steps=100000000
[ -z "${ifresume}" ] && ifresume=True

[ -z "${wandb}" ] && wandb=False
[ -z "${swanlab}" ] && swanlab=True
[ -z "${wandb_key}" ] && wandb_key="${WANDB_API_KEY:-}"
[ -z "${swanlab_key}" ] && swanlab_key="${SWANLAB_API_KEY:-}"
[ -z "${aim}" ] && aim=False
[ -z "${tb}" ] && tb=False
[ -z "${aim_dir}" ] && aim_dir="./outputs/aim_log/"
[ -z "${tb_dir}" ] && tb_dir="./outputs/tb_log/"
[ -z "${save_dir}" ] && save_dir="./outputs/ckpt_log/"

if [ -n "${wandb_key}" ]; then
  export WANDB_API_KEY="$wandb_key"
  if command -v wandb >/dev/null 2>&1; then
    wandb login --relogin "$wandb_key"
  fi
fi
if [ -n "${swanlab_key}" ] && command -v swanlab >/dev/null 2>&1; then
  swanlab login -k "$swanlab_key"
fi

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile="${PROJECT_ROOT}/job/hostfile"
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62361
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${NNODES}" ] && NNODES=1
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=1

if [[ -z "${NNODES}" ]]
then
  DISTRIBUTED_ARGS=""
else
  if (( $NNODES == 1))
  then
    DISTRIBUTED_ARGS="--nproc_per_node $n_gpu \
                      --master_port $MASTER_PORT"
  else
    DISTRIBUTED_ARGS="--nproc_per_node $n_gpu \
                      --nnodes $NNODES \
                      --node_rank $NODERANK \
                    --master_addr $MASTER_ADDR"
  fi
fi

echo "DISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}"
echo "n_gpu: ${n_gpu}"

export ifresume=false
export dataset_micro_batch_size="atom:2048"
export dataset_name_list="omol25"
export gradient_accumulation_steps=1
export total_num_steps=1400000

export data_path_list="${PROJECT_ROOT}/data/omol25/train"
export data_path_list_valid="${PROJECT_ROOT}/data/omol25/val_1M"
torchrun $DISTRIBUTED_ARGS src/molfm/tasks/pretrain_smallmol.py \
          --config-name=config_molfm.yaml \
          AutoGradForce=False \
          loss_fn='atoml2mae' \
          lr=0.0008 \
          weight_decay=0.001 \
          energy_loss_weight=1 force_loss_weight=30 \
          clip_grad_norm=100 \
          total_num_steps=$total_num_steps \
          val_batch_interval=20000 \
          save_batch_interval=20000 \
          ifresume=$ifresume \
          log_interval=100 \
          wandb=$wandb \
          swanlab=$swanlab \
          swanlab_project=lhuang_mfm \
          swanlab_workspace=AlphaLab \
          tb=$tb \
          tb_dir=$tb_dir \
          save_dir=$save_dir \
          ckpt_path="none" \
          experiment_name=e2_0106_4x8_100M_4S_QKdot_atom2048 \
          gradient_accumulation_steps=$gradient_accumulation_steps \
          scheduler_config=groupWarmupDecayLR \
          strategy_config=ddp \
          with_moe=False \
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
          backbone_config.max_neighbors=32 \
          backbone_config.alpha_drop=0 \
          backbone_config.basis_type='gaussiansmear' \
          backbone_config.norm_layer='rms_norm_sh' \
          backbone_config.attn_type='so2-first-order' \
          backbone_config.tp_type='QKdot_alpha+triton' \
          backbone_config.edge_embedtype='default' \
          backbone_config.ffn_type='s3' \
          backbone_config.encoder='default' \
          data_path=$data_path \
          data_path_list=$data_path_list dataset_name_list=$dataset_name_list \
          dataset_sample_prob=$dataset_sample_prob dataset_micro_batch_size=$dataset_micro_batch_size \
          data_path_list_valid=$data_path_list_valid
