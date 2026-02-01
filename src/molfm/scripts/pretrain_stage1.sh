#!/usr/bin/env bash


ulimit -c unlimited

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

# ---------- Data ----------
[ -z "${data_path}" ] && data_path='/path/to/data_root'
[ -z "${data_path_list}" ] && data_path_list="/path/to/train_dataset"
[ -z "${data_path_list_valid}" ] && data_path_list_valid="/path/to/valid_dataset"

[ -z "${dataset_name_list}" ] && dataset_name_list="<dataset_name>"
[ -z "${dataset_sample_prob}" ] && dataset_sample_prob='1'
[ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="atom:2048"
[ -z "${use_unified_batch_sampler}" ] && use_unified_batch_sampler=True
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)

# ---------- Training ----------
[ -z "${total_num_steps}" ] && total_num_steps=1400000
[ -z "${ifresume}" ] && ifresume=false
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=1

# ---------- Logging ----------
[ -z "${wandb}" ] && wandb=False
[ -z "${wandb_key}" ] && wandb_key="<WANDB_API_KEY>"
[ -z "${swanlab}" ] && swanlab=False
[ -z "${swanlab_key}" ] && swanlab_key="<SWANLAB_API_KEY>"
[ -z "${aim}" ] && aim=False
[ -z "${tb}" ] && tb=False
[ -z "${aim_dir}" ] && aim_dir="./outputs/aim_log/"
[ -z "${tb_dir}" ] && tb_dir="./outputs/tb_log/"
[ -z "${save_dir}" ] && save_dir="./outputs/ckpt_log/"

# Optional: external logging endpoints
# wandb login --relogin $wandb_key
# swanlab login -k $swanlab_key

# ---------- Distributed ----------
[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/path/to/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=29500
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${NNODES}" ] && NNODES=1

if [[ -z "${NNODES}" ]]
then
  DISTRIBUTED_ARGS=""
else
  if (( NNODES == 1 ))
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

# ---------- Launch ----------
torchrun $DISTRIBUTED_ARGS src/molfm/tasks/pretrain_smallmol.py \
          --config-name=config_molfm.yaml \
          AutoGradForce=False \
          loss_fn='atoml2mae' \
          lr=0.0008 \
          weight_decay=0.001 \
          energy_loss_weight=4 force_loss_weight=30 \
          clip_grad_norm=100 \
          total_num_steps=$total_num_steps \
          val_batch_interval=20000 \
          save_batch_interval=20000 \
          ifresume=$ifresume \
          log_interval=100 \
          wandb=$wandb \
          swanlab=$swanlab \
          tb=$tb \
          tb_dir=$tb_dir \
          save_dir=$save_dir \
          ckpt_path="none" \
          experiment_name="<experiment_name>" \
          gradient_accumulation_steps=$gradient_accumulation_steps \
          scheduler_config=groupWarmupDecayLR \
          strategy_config=ddp \
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
          backbone_config.tp_type='QKdotS_alpha+triton' \
          backbone_config.edge_embedtype='default' \
          backbone_config.ffn_type='s3' \
          backbone_config.encoder='default' \
          data_path=$data_path \
          data_path_list=$data_path_list dataset_name_list=$dataset_name_list \
          dataset_sample_prob=$dataset_sample_prob dataset_micro_batch_size=$dataset_micro_batch_size \
          data_path_list_valid=$data_path_list_valid
