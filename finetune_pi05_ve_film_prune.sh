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
# - serving 时要使用与训练一致的 pruning YAML（否则 state_dict key / 行为不一致）。
# - 最小 pruning YAML 只保留剪枝相关字段，其余训练默认值在父仓 loader 中固化。

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

# VLA-OPT pruning YAML（serving 也要一致）
pruning_config="${script_dir}/config/pruning/post_encoder.yaml"

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
echo "vla-opt pruning_config: ${pruning_config}"
echo "num_train_steps: ${num_train_steps}"
echo ""

[[ -x "${python_bin}" ]] || die "Python not found at ${python_bin} (run: uv sync)"
[[ "${num_train_steps}" =~ ^[0-9]+$ ]] || die "num_train_steps must be a positive integer"
[[ "${num_train_steps}" -ge 1 ]] || die "num_train_steps must be >= 1"
[[ -f "${pruning_config}" ]] || die "Missing pruning config: ${pruning_config}"

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
  --vla-opt-pruning-config "${pruning_config}" \
  --log-level INFO 2>&1 | tee "${log_file}"
