#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
vla_opt_repo_dir="$(cd "${script_dir}/../../.." && pwd)"

cd "${repo_dir}"

die() { echo "Error: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  bash tools/train_pi05_baseline_libero_all.sh

This script launches Pi0.5 baseline fine-tuning on the full local LIBERO dataset with:
  - policy config: pi05_libero_all_cross_attn_post_gauss
  - experiment name: pi05_libero_all_baseline
  - wandb project: vla-opt

The script intentionally reuses the same policy config as the pruning run so both runs
share the same computed norm stats assets.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

policy_config="pi05_libero_all_cross_attn_post_gauss"
exp_name="pi05_libero_all_baseline"
checkpoint_base_dir="/workspace/laiminxin/vla-opt/third_party/openpi/checkpoints"
gpus="1,2,3,4"
python_bin=".venv/bin/python"
base_ckpt="/workspace/laiminxin/models/pi05_base_pytorch"
data_repo_id="/workspace/laiminxin/datasets/lerobot_datasets/physical-intelligence/libero"
num_train_steps="60000"
wandb_enabled="true"
resume_mode="auto"
wandb_project="vla-opt"

[[ -x "${python_bin}" ]] || die "Python not found at ${python_bin} (run: uv sync)"
[[ "${num_train_steps}" =~ ^[0-9]+$ ]] || die "num_train_steps must be a positive integer"
[[ "${num_train_steps}" -ge 1 ]] || die "num_train_steps must be >= 1"

run_dir="${checkpoint_base_dir}/${policy_config}/${exp_name}"
log_file="${run_dir}/train.log"
mkdir -p "$(dirname "${log_file}")"

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

echo "=== Train (baseline | full LIBERO) ==="
echo "policy_config: ${policy_config}"
echo "exp_name: ${exp_name}"
echo "gpus: ${gpus}"
echo "base_ckpt: ${base_ckpt}"
echo "data_repo_id: ${data_repo_id}"
echo "num_train_steps: ${num_train_steps}"
echo "wandb_project: ${wandb_project}"
echo "run_dir: ${run_dir}"
echo "log: ${log_file}"
echo

PYTHONPATH="${repo_dir}/src:${vla_opt_repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}" \
CUDA_VISIBLE_DEVICES="${gpus}" "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node="${nproc_per_node}" \
  scripts/train_pytorch.py "${policy_config}" \
  --exp-name "${exp_name}" "${resume_flag[@]}" \
  --project-name "${wandb_project}" \
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
  --log-level INFO 2>&1 | tee "${log_file}"
