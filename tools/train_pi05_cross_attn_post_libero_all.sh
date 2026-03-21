#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"

cd "${repo_dir}"

usage() {
  cat <<'EOF'
Usage:
  bash tools/train_pi05_cross_attn_post_libero_all.sh

This script launches Pi0.5 fine-tuning on the full local LIBERO dataset with:
  - policy config: pi05_libero_all
  - pruning config: config/pruning/cross_attn_post_t64_gauss.yaml
  - experiment name: cross_attn_post_t64_gauss

Edit the defaults in the script if you need different GPUs, checkpoint base dir, or resume behavior.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

policy_config="pi05_libero_all_cross_attn_post_gauss"
exp_name="cross_attn_post_t64_gauss"
opt_config="config/pruning/cross_attn_post_t64_gauss.yaml"
checkpoint_base_dir="/workspace/laiminxin/vla-opt/third_party/openpi/checkpoints"
gpus="1,2,3,4"
python_bin=".venv/bin/python"
base_ckpt="/workspace/laiminxin/models/pi05_base_pytorch"
data_repo_id="/workspace/laiminxin/datasets/lerobot_datasets/physical-intelligence/libero"
num_train_steps="60000"
wandb_enabled="true"
resume_mode="auto"

echo "=== Train (cross_attn_post | full LIBERO) ==="
echo "policy_config: ${policy_config}"
echo "exp_name: ${exp_name}"
echo "opt_config: ${opt_config}"
echo "gpus: ${gpus}"
echo "base_ckpt: ${base_ckpt}"
echo "data_repo_id: ${data_repo_id}"
echo "num_train_steps: ${num_train_steps}"
echo

bash tools/train_pi05_experiment.sh \
  --opt-config "${opt_config}" \
  --policy-config "${policy_config}" \
  --exp-name "${exp_name}" \
  --checkpoint-base-dir "${checkpoint_base_dir}" \
  --gpus "${gpus}" \
  --python-bin "${python_bin}" \
  --base-ckpt "${base_ckpt}" \
  --data-repo-id "${data_repo_id}" \
  --num-train-steps "${num_train_steps}" \
  --wandb-enabled "${wandb_enabled}" \
  --resume-mode "${resume_mode}"
