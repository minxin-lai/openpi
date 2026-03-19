#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"

cd "${repo_dir}"

die() { echo "Error: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  bash tools/train_pi05_baseline.sh

This script uses the repo baseline defaults defined inside the file.
Edit the script if you need to change GPUs, checkpoint paths, or dataset paths.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

config="pi05_libero_spatial"
exp="pi05_baseline"
gpus="4,5"
python_bin=".venv/bin/python"
base_ckpt="/workspace/laiminxin/models/pi05_base_pytorch"
data_repo_id="/workspace/laiminxin/datasets/lerobot_datasets/libero_spatial"
wandb_enabled="true"
resume_mode="auto"

log_file="checkpoints/${config}/${exp}/train.log"
mkdir -p "$(dirname "${log_file}")"

echo "=== Train (baseline) ==="
echo "config/exp: ${config}/${exp}"
echo "gpus: ${gpus}"
echo "base_ckpt: ${base_ckpt}"
echo "data_repo_id: ${data_repo_id}"
echo "log: ${log_file}"
echo "ckpt_out: checkpoints/${config}/${exp}/<step>/model.safetensors"
echo

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
