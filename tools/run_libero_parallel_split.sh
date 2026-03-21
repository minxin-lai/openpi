#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"

cd "${repo_dir}"

usage() {
  cat <<'EOF'
Usage: bash tools/run_libero_parallel_split.sh [--gpu <id[,id...]>]

Options:
  --gpu <id[,id...]>   Comma-separated GPU list. Default: 1,2,3,4
  -h, --help           Show this help message
EOF
}

gpu_arg="1,2,3,4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu_arg="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

IFS=',' read -r -a gpu_ids <<< "${gpu_arg}"

if [[ ${#gpu_ids[@]} -eq 0 ]]; then
  echo "No GPUs provided via --gpu" >&2
  exit 1
fi

for gpu in "${gpu_ids[@]}"; do
  if [[ -z "${gpu}" ]]; then
    echo "Invalid --gpu value: ${gpu_arg}" >&2
    exit 1
  fi
done

run_eval() {
  local gpu="$1"
  local run_tag="$2"
  local ckpt_dir="$3"
  local opt_config="$4"
  local port="$5"

  echo
  echo "=== GPU ${gpu} | ${run_tag} | port ${port} ==="
  bash tools/run_libero.sh \
    --run-tag "${run_tag}" \
    --ckpt-dir "${ckpt_dir}" \
    --policy-config pi05_libero_spatial \
    --opt-config "${opt_config}" \
    --port "${port}" \
    --gpu "${gpu}" \
    --trials 50
}

run_specs=(
  "inside_t64|checkpoints/pi05_libero_spatial/inside_t64/59999|config/pruning/inside_t64.yaml"
  "inside_t64_gauss|checkpoints/pi05_libero_spatial/inside_t64/59999|config/pruning/inside_t64_gauss.yaml"
  "inside_t128|checkpoints/pi05_libero_spatial/inside_t128/59999|config/pruning/inside_t128.yaml"
  "inside_t128_gauss|checkpoints/pi05_libero_spatial/inside_t128/59999|config/pruning/inside_t128_gauss.yaml"
  "post_t64|checkpoints/pi05_libero_spatial/post_t64/29999|config/pruning/post_t64.yaml"
  "post_t128|checkpoints/pi05_libero_spatial/post_t128/59999|config/pruning/post_t128.yaml"
  "post_t64_gauss|checkpoints/pi05_libero_spatial/post_t64/29999|config/pruning/post_t64_gauss.yaml"
  "post_t128_gauss|checkpoints/pi05_libero_spatial/post_t128/59999|config/pruning/post_t128_gauss.yaml"
)

queue_size=${#gpu_ids[@]}

run_gpu_queue() {
  local worker_idx="$1"
  local gpu="$2"
  local spec_idx

  for ((spec_idx=worker_idx; spec_idx<${#run_specs[@]}; spec_idx+=queue_size)); do
    IFS='|' read -r run_tag ckpt_dir opt_config <<< "${run_specs[spec_idx]}"
    run_eval "${gpu}" "${run_tag}" "${ckpt_dir}" "${opt_config}" "$((9001 + spec_idx))"
  done
}

pids=()
statuses=()

for idx in "${!gpu_ids[@]}"; do
  run_gpu_queue "${idx}" "${gpu_ids[idx]}" &
  pids[idx]=$!
done

for idx in "${!pids[@]}"; do
  status=0
  wait "${pids[idx]}" || status=$?
  statuses[idx]=$status
done

echo
for idx in "${!gpu_ids[@]}"; do
  echo "GPU ${gpu_ids[idx]} queue exit status: ${statuses[idx]}"
done

for status in "${statuses[@]}"; do
  if [[ "${status}" -ne 0 ]]; then
    exit 1
  fi
done
