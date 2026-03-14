#!/usr/bin/env bash
set -euo pipefail

# Pi0.5 (LIBERO Spatial) VLA-OPT finetune (VE FiLM + VE STE prune)
#
# 用法：
#   cd third_party/openpi
#   bash finetune_pi05_ve_film_prune.sh
#
# 你通常只需要改：
# - gpus
# - base_ckpt / data_repo_id
#
# 关键点（尽量记住这两条就够）：
# - serving 时的 wrapper 参数必须和训练时一致（否则 state_dict key / 行为不一致）。
# - ste_prune_stage=auto 会在训练中期从 mask 切到 gather（对齐推理的真加速）。

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

die() { echo "Error: $*" >&2; exit 2; }

# ======================
# 配置区（建议只改这里）
# ======================
config="pi05_libero_spatial"
exp="vla_opt_pi05_post_encoder_ste"
# 训练输出根目录（checkpoint + train.log）
checkpoint_base_dir="/workspace/laiminxin/vla-opt/third_party/openpi/checkpoints"

gpus="3"
python_bin=".venv/bin/python"
base_ckpt="/workspace/laiminxin/models/pi05_base_pytorch"
data_repo_id="/workspace/laiminxin/datasets/lerobot_datasets/libero_spatial"
num_train_steps="60000"

# wandb：默认开；没配 WANDB_API_KEY 会自动关并提示
wandb_enabled="true"

resume_mode="auto"   # auto|true|false

# VLA-OPT 训练参数（serving 也要一致）
ve_film_num_blocks="4"
ste_prune_k="128"
ste_prune_stage="auto"     # auto (mask->gather) | mask | gather
ste_prune_point="post_encoder"  # post_encoder | encoder_layer
ste_prune_score_num_layers="3"  # FiLM-only mean score layers
ste_prune_layer=""         # 仅 encoder_layer 模式需要；留空让代码自动推断
ste_prune_switch_step="-1" # auto 时 -1 => 默认 num_train_steps//2
ste_prune_tau="2.0"
ste_prune_tau_final="0.2"
ste_prune_lambda_budget="0.01"
ste_prune_lambda_bin="0.01"

run_dir="${checkpoint_base_dir}/${config}/${exp}"
log_file="${run_dir}/train.log"
mkdir -p "$(dirname "${log_file}")"
repo_root="$(cd "${script_dir}/../.." && pwd)"
vla_opt_src="${repo_root}/src"

echo "=== Train (vla_opt) ==="
echo "config/exp: ${config}/${exp}"
echo "gpus: ${gpus}"
echo "base_ckpt: ${base_ckpt}"
echo "data_repo_id: ${data_repo_id}"
echo "run_dir: ${run_dir}"
echo "log: ${log_file}"
echo "ckpt_out: ${run_dir}/<step>/model.safetensors"
echo ""
echo "vla-opt: ve_film_num_blocks=${ve_film_num_blocks} ste_prune_k=${ste_prune_k} stage=${ste_prune_stage} point=${ste_prune_point} score_num_layers=${ste_prune_score_num_layers}"
echo "num_train_steps: ${num_train_steps}"
echo ""

[[ -x "${python_bin}" ]] || die "Python not found at ${python_bin} (run: uv sync)"
[[ "${ste_prune_point}" == "post_encoder" || "${ste_prune_point}" == "encoder_layer" ]] || die "ste_prune_point must be post_encoder|encoder_layer"
[[ "${ste_prune_score_num_layers}" =~ ^[0-9]+$ ]] || die "ste_prune_score_num_layers must be a positive integer"
[[ "${ste_prune_score_num_layers}" -ge 1 ]] || die "ste_prune_score_num_layers must be >= 1"
[[ "${num_train_steps}" =~ ^[0-9]+$ ]] || die "num_train_steps must be a positive integer"
[[ "${num_train_steps}" -ge 1 ]] || die "num_train_steps must be >= 1"

if [[ "${ste_prune_point}" == "post_encoder" ]]; then
  [[ "${ve_film_num_blocks}" =~ ^[0-9]+$ ]] || die "ve_film_num_blocks must be a positive integer"
  [[ "${ve_film_num_blocks}" -ge "${ste_prune_score_num_layers}" ]] || \
    die "post_encoder 模式要求 ve_film_num_blocks >= ste_prune_score_num_layers"
fi

resume_flag=()
if [[ "${resume_mode}" == "true" ]]; then
  resume_flag=(--resume)
elif [[ "${resume_mode}" == "auto" ]]; then
  if [[ -d "${run_dir}" ]] && find "${run_dir}" -maxdepth 2 -type f -name "model.safetensors" -print -quit | grep -q .; then
    resume_flag=(--resume)
  fi
fi

wandb_flag=()
if [[ "${wandb_enabled}" == "true" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "Warn: WANDB_API_KEY is empty => disable wandb." >&2
  else
    wandb_flag=(--wandb-enabled)
  fi
fi

IFS=',' read -r -a gpu_list <<< "${gpus}"
nproc_per_node="${#gpu_list[@]}"
[[ "${nproc_per_node}" -ge 1 ]] || die "gpus is empty"

ste_prune_layer_flag=()
if [[ -n "${ste_prune_layer}" ]]; then
  ste_prune_layer_flag=(--ste-prune-layer "${ste_prune_layer}")
fi

PYTHONPATH="${script_dir}/src:${vla_opt_src}${PYTHONPATH:+:${PYTHONPATH}}" \
CUDA_VISIBLE_DEVICES="${gpus}" "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node="${nproc_per_node}" \
  scripts/train_pytorch.py "${config}" \
  --exp-name "${exp}" "${resume_flag[@]}" \
  --checkpoint-base-dir "${checkpoint_base_dir}" \
  --pytorch-weight-path "${base_ckpt}" \
  --data.repo_id "${data_repo_id}" \
  --batch-size 32 \
  --num-workers 1 \
  --log-interval 100 \
  --save-interval 10000 \
  --num-train-steps "${num_train_steps}" \
  --pytorch-training-precision bfloat16 \
  "${wandb_flag[@]}" \
  --ve-film --ve-film-num-blocks "${ve_film_num_blocks}" --no-ve-film-freeze-base \
  --ste-prune --ste-prune-k "${ste_prune_k}" --ste-prune-stage "${ste_prune_stage}" --ste-prune-point "${ste_prune_point}" \
  "${ste_prune_layer_flag[@]}" \
  --ste-prune-score-num-layers "${ste_prune_score_num_layers}" --ste-prune-switch-step "${ste_prune_switch_step}" \
  --ste-prune-tau "${ste_prune_tau}" --ste-prune-tau-final "${ste_prune_tau_final}" \
  --ste-prune-lambda-budget "${ste_prune_lambda_budget}" --ste-prune-lambda-bin "${ste_prune_lambda_bin}" --no-ste-prune-freeze-base \
  --log-level INFO 2>&1 | tee "${log_file}"
