#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"

cd "${repo_dir}"

export OPENPI_TORCH_COMPILE="${OPENPI_TORCH_COMPILE:-1}"
export OPENPI_TORCH_COMPILE_MODE="${OPENPI_TORCH_COMPILE_MODE:-reduce-overhead}"

run_dump() {
  local gpu="$1"
  local run_tag="$2"
  local ckpt_dir="$3"
  local opt_config="$4"
  local port="$5"

  echo
  echo "=== GPU ${gpu} | ${run_tag} | port ${port} ==="
  bash tools/run_libero_dump.sh \
    --run-tag "${run_tag}" \
    --ckpt-dir "${ckpt_dir}" \
    --policy-config pi05_libero_spatial \
    --opt-config "${opt_config}" \
    --port "${port}" \
    --gpu "${gpu}" \
    --trials 1
}

run_gpu4_queue() {
  run_dump 4 \
    inside_t64 \
    checkpoints/pi05_libero_spatial/inside_t64/59999 \
    config/pruning/inside_t64.yaml \
    9001

  run_dump 4 \
    inside_t64_gauss \
    checkpoints/pi05_libero_spatial/inside_t64/59999 \
    config/pruning/inside_t64_gauss.yaml \
    9002

  run_dump 4 \
    inside_t128 \
    checkpoints/pi05_libero_spatial/inside_t128/59999 \
    config/pruning/inside_t128.yaml \
    9003

  run_dump 4 \
    inside_t128_gauss \
    checkpoints/pi05_libero_spatial/inside_t128/59999 \
    config/pruning/inside_t128_gauss.yaml \
    9004
}

run_gpu6_queue() {
  run_dump 6 \
    post_t64 \
    checkpoints/pi05_libero_spatial/post_t64/29999 \
    config/pruning/post_t64.yaml \
    9005

  run_dump 6 \
    post_t128 \
    checkpoints/pi05_libero_spatial/post_t128/59999 \
    config/pruning/post_t128.yaml \
    9006

  run_dump 6 \
    post_t64_gauss \
    checkpoints/pi05_libero_spatial/post_t64/29999 \
    config/pruning/post_t64_gauss.yaml \
    9007

  run_dump 6 \
    post_t128_gauss \
    checkpoints/pi05_libero_spatial/post_t128/59999 \
    config/pruning/post_t128_gauss.yaml \
    9008
}

run_gpu4_queue &
pid_gpu4=$!

run_gpu6_queue &
pid_gpu6=$!

status_gpu4=0
status_gpu6=0

wait "${pid_gpu4}" || status_gpu4=$?
wait "${pid_gpu6}" || status_gpu6=$?

echo
echo "GPU 4 queue exit status: ${status_gpu4}"
echo "GPU 6 queue exit status: ${status_gpu6}"

if [[ "${status_gpu4}" -ne 0 || "${status_gpu6}" -ne 0 ]]; then
  exit 1
fi
