#!/usr/bin/env bash
set -euo pipefail

# Pi0.5 (LIBERO Spatial) baseline finetune
#
# 用法：
#   cd third_party/openpi
#   bash finetune_pi05_baseline.sh
#
# 你通常只需要改：
# - gpus
# - base_ckpt / data_repo_id

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

die() { echo "Error: $*" >&2; exit 2; }

# ======================
# 配置区（建议只改这里）
# ======================
config="pi05_libero_spatial"
exp="pi05_baseline"

gpus="4,5"  # e.g. "0" or "0,1"
python_bin=".venv/bin/python"
base_ckpt="/workspace/laiminxin/models/pi05_base_pytorch"
data_repo_id="/workspace/laiminxin/datasets/lerobot_datasets/libero_spatial"

# wandb：默认开；没配 WANDB_API_KEY 会自动关并提示
wandb_enabled="true"

# 自动 resume：如果 checkpoints 下已经有 model.safetensors，就加 --resume
resume_mode="auto"   # auto|true|false

log_file="checkpoints/${config}/${exp}/train.log"
mkdir -p "$(dirname "${log_file}")"

echo "=== Train (baseline) ==="
echo "config/exp: ${config}/${exp}"
echo "gpus: ${gpus}"
echo "base_ckpt: ${base_ckpt}"
echo "data_repo_id: ${data_repo_id}"
echo "log: ${log_file}"
echo "ckpt_out: checkpoints/${config}/${exp}/<step>/model.safetensors"
echo ""

[[ -x "${python_bin}" ]] || die "Python not found at ${python_bin} (run: uv sync)"

resume_flag=()
if [[ "${resume_mode}" == "true" ]]; then
  resume_flag=(--resume)
elif [[ "${resume_mode}" == "auto" ]]; then
  if [[ -d "checkpoints/${config}/${exp}" ]] && find "checkpoints/${config}/${exp}" -maxdepth 2 -type f -name "model.safetensors" -print -quit | grep -q .; then
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

CUDA_VISIBLE_DEVICES="${gpus}" "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node="${nproc_per_node}" \
  scripts/train_pytorch.py "${config}" \
  --exp-name "${exp}" "${resume_flag[@]}" \
  --pytorch-weight-path "${base_ckpt}" \
  --data.repo_id "${data_repo_id}" \
  --batch-size 32 \
  --num-workers 1 \
  --log-interval 100 \
  --save-interval 5000 \
  --pytorch-training-precision bfloat16 \
  "${wandb_flag[@]}" \
  --log-level INFO 2>&1 | tee "${log_file}"

