#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Baseline server launcher (NO VLA-OPT: no FiLM, no STE pruning).
# Edit these values to point at the checkpoint you want to evaluate.
# Example:
#   CKPT_DIR=/workspace/laiminxin/vla-opt/third_party/openpi/checkpoints/pi05_libero_spatial/pi05_baseline/30000 GPU=1 bash server_pi05_libero_baseline.sh

CKPT_DIR="${CKPT_DIR:-/workspace/laiminxin/vla-opt/third_party/openpi/checkpoints/pi05_libero_spatial/pi05_baseline/30000}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero_spatial}"

PORT="${PORT:-8002}"
GPU="${GPU:-0}"

[[ -f "${CKPT_DIR}/model.safetensors" ]] || { echo "Missing ${CKPT_DIR}/model.safetensors" >&2; exit 2; }

# For `pi05_libero_spatial`, norm stats are expected at the dataset repo_id path.
NORM_STATS_PATH="/workspace/laiminxin/datasets/lerobot_datasets/libero_spatial/norm_stats.json"
if [[ ! -f "${NORM_STATS_PATH}" ]]; then
  echo "Missing norm stats: ${NORM_STATS_PATH}" >&2
  echo "Hint: cd third_party/openpi && uv run scripts/compute_norm_stats.py --config-name ${POLICY_CONFIG}" >&2
  exit 2
fi

echo "Serve (baseline): ckpt=${CKPT_DIR} GPU=${GPU} PORT=${PORT}"
echo "Client: HOST=127.0.0.1 PORT=${PORT} TRIALS=1 bash client_libero_eval.sh"

# Tracing (optional; disabled by default for timing comparisons).
OPENPI_ROOT="$(pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
TRACE="${TRACE:-0}"
TRACE_OUT_DIR="${TRACE_OUT_DIR:-${OPENPI_ROOT}/runs/openpi_pi05_libero_trace_${TS}}"
TRACE_ATTN_LAYERS="${TRACE_ATTN_LAYERS:-}"   # empty => last layer
TRACE_EVERY_N="${TRACE_EVERY_N:-1}"
TRACE_MAX_DUMPS="${TRACE_MAX_DUMPS:-1}"
if [[ "${TRACE}" == "1" ]]; then
  echo "Trace: on out_dir=${TRACE_OUT_DIR} attn_layers=${TRACE_ATTN_LAYERS:-<last>} every_n=${TRACE_EVERY_N} max_dumps=${TRACE_MAX_DUMPS}"
else
  echo "Trace: off"
fi

SERVER_LOG="${SERVER_LOG:-${OPENPI_ROOT}/runs/openpi_pi05_libero_server_baseline_${TS}.log}"
mkdir -p "${OPENPI_ROOT}/runs"
echo "Log: ${SERVER_LOG}"
echo "torch_compile: ${OPENPI_TORCH_COMPILE:-1}"
echo "triton_autotune: ${TRITON_AUTOTUNE:-<unset>}"
echo "torchinductor_max_autotune: ${TORCHINDUCTOR_MAX_AUTOTUNE:-<unset>}"

export OPENPI_TORCH_COMPILE="${OPENPI_TORCH_COMPILE:-1}"
export OPENPI_TORCH_COMPILE_MODE="reduce-overhead"

extra_args=()
if [[ "${TRACE}" == "1" ]]; then
  extra_args=(
    --trace-out-dir "${TRACE_OUT_DIR}"
    --trace-dump-attn
    --trace-attn-layers "${TRACE_ATTN_LAYERS}"
    --trace-every-n "${TRACE_EVERY_N}"
    --trace-max-dumps "${TRACE_MAX_DUMPS}"
  )
fi

CUDA_VISIBLE_DEVICES="${GPU}" uv run scripts/serve_policy.py \
  --env LIBERO --port "${PORT}" \
  "${extra_args[@]}" \
  policy:checkpoint --policy.config "${POLICY_CONFIG}" --policy.dir "${CKPT_DIR}" 2>&1 | tee "${SERVER_LOG}"
