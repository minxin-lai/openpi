#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"

cd "${repo_dir}"

die() { echo "Error: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  bash tools/train_pi05_baseline.sh [options...]

Options:
  --policy-config <name>       required policy config name
  --exp-name <name>            required experiment name
  --checkpoint-base-dir <dir>  optional checkpoint root, default: checkpoints
  --gpus <ids>                 required comma-separated GPU list
  --python-bin <path>          optional python path, default: .venv/bin/python
  --base-ckpt <path>           required base PyTorch checkpoint path
  --data-repo-id <path>        required dataset repo/path override
  --num-train-steps <n>        required total train steps
  --batch-size <n>             optional batch size, default: 32
  --num-workers <n>            optional dataloader workers, default: 1
  --log-interval <n>           optional log interval, default: 100
  --save-interval <n>          optional save interval, default: 5000
  --entity-name <name>         optional wandb entity override
  --project-name <name>        optional wandb project name override
  --wandb-enabled <true|false> optional, default: true
  --resume-mode <auto|true|false> optional, default: auto
EOF
}

policy_config=""
exp_name=""
checkpoint_base_dir="checkpoints"
gpus=""
python_bin=".venv/bin/python"
base_ckpt=""
data_repo_id=""
num_train_steps=""
batch_size="32"
num_workers="1"
log_interval="100"
save_interval="5000"
entity_name=""
project_name=""
wandb_enabled="true"
resume_mode="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --policy-config) policy_config="${2:?}"; shift 2 ;;
    --exp-name) exp_name="${2:?}"; shift 2 ;;
    --checkpoint-base-dir) checkpoint_base_dir="${2:?}"; shift 2 ;;
    --gpus) gpus="${2:?}"; shift 2 ;;
    --python-bin) python_bin="${2:?}"; shift 2 ;;
    --base-ckpt) base_ckpt="${2:?}"; shift 2 ;;
    --data-repo-id) data_repo_id="${2:?}"; shift 2 ;;
    --num-train-steps) num_train_steps="${2:?}"; shift 2 ;;
    --batch-size) batch_size="${2:?}"; shift 2 ;;
    --num-workers) num_workers="${2:?}"; shift 2 ;;
    --log-interval) log_interval="${2:?}"; shift 2 ;;
    --save-interval) save_interval="${2:?}"; shift 2 ;;
    --entity-name) entity_name="${2:?}"; shift 2 ;;
    --project-name) project_name="${2:?}"; shift 2 ;;
    --wandb-enabled) wandb_enabled="${2:?}"; shift 2 ;;
    --resume-mode) resume_mode="${2:?}"; shift 2 ;;
    *) die "Unknown option: $1 (run --help)" ;;
  esac
done

[[ -n "${policy_config}" ]] || die "--policy-config is required"
[[ -n "${exp_name}" ]] || die "--exp-name is required"
[[ -n "${gpus}" ]] || die "--gpus is required"
[[ -n "${base_ckpt}" ]] || die "--base-ckpt is required"
[[ -n "${data_repo_id}" ]] || die "--data-repo-id is required"
[[ -n "${num_train_steps}" ]] || die "--num-train-steps is required"

log_file="${checkpoint_base_dir}/${policy_config}/${exp_name}/train.log"
mkdir -p "$(dirname "${log_file}")"

echo "=== Train (baseline) ==="
echo "policy_config: ${policy_config}"
echo "exp_name: ${exp_name}"
echo "checkpoint_base_dir: ${checkpoint_base_dir}"
echo "gpus: ${gpus}"
echo "base_ckpt: ${base_ckpt}"
echo "data_repo_id: ${data_repo_id}"
echo "num_train_steps: ${num_train_steps}"
echo "log: ${log_file}"
echo "ckpt_out: ${checkpoint_base_dir}/${policy_config}/${exp_name}/<step>/model.safetensors"
echo

[[ -x "${python_bin}" ]] || die "Python not found at ${python_bin} (run: uv sync)"
[[ "${num_train_steps}" =~ ^[0-9]+$ ]] || die "num_train_steps must be a positive integer"
[[ "${num_train_steps}" -ge 1 ]] || die "num_train_steps must be >= 1"

resume_flag=()
if [[ "${resume_mode}" == "true" ]]; then
  resume_flag=(--resume)
elif [[ "${resume_mode}" == "auto" ]]; then
  if [[ -d "${checkpoint_base_dir}/${policy_config}/${exp_name}" ]] && find "${checkpoint_base_dir}/${policy_config}/${exp_name}" -maxdepth 2 -type f -name "model.safetensors" -print -quit | grep -q .; then
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

entity_flag=()
if [[ -n "${entity_name}" ]]; then
  entity_flag=(--entity-name "${entity_name}")
fi

project_flag=()
if [[ -n "${project_name}" ]]; then
  project_flag=(--project-name "${project_name}")
fi

IFS=',' read -r -a gpu_list <<< "${gpus}"
nproc_per_node="${#gpu_list[@]}"
[[ "${nproc_per_node}" -ge 1 ]] || die "gpus is empty"

CUDA_VISIBLE_DEVICES="${gpus}" "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node="${nproc_per_node}" \
  scripts/train_pytorch.py "${policy_config}" \
  --exp-name "${exp_name}" "${resume_flag[@]}" \
  --checkpoint-base-dir "${checkpoint_base_dir}" \
  "${entity_flag[@]}" \
  "${project_flag[@]}" \
  --pytorch-weight-path "${base_ckpt}" \
  --data.repo_id "${data_repo_id}" \
  --batch-size "${batch_size}" \
  --num-workers "${num_workers}" \
  --log-interval "${log_interval}" \
  --save-interval "${save_interval}" \
  --num-train-steps "${num_train_steps}" \
  --pytorch-training-precision bfloat16 \
  "${wandb_flag[@]}" \
  --log-level INFO 2>&1 | tee "${log_file}"
