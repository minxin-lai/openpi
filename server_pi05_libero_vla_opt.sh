#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# One-file "edit and run" server launcher.
# - Edit the values below to point at the checkpoint you want to evaluate.
# - This starts the LIBERO WebSocket policy server (it does NOT do finetuning).
# - Enables VLA-OPT wrappers: Vision Encoder (VE) FiLM (num_film_blocks=4) + VE STE prune (K=64, stage=gather).
# Example:
#   CKPT_DIR=/workspace/laiminxin/vla-opt/third_party/openpi/checkpoints/pi05_libero_spatial/vla_opt_pi05_stage_a_ste/29999 GPU=1 bash server_pi05_libero_vla_opt.sh
#
# Tracing (optional):
# - Disabled by default for timing comparisons.
# - Set TRACE=1 to enable tracer dump.

# Checkpoint directory must contain `model.safetensors`.
# Default checkpoint path used for quick local benchmarking.
CKPT_DIR="${CKPT_DIR:-/workspace/laiminxin/vla-opt/third_party/openpi/checkpoints/pi05_libero_spatial/vla_opt_pi05_stage_a_ste/29999}"

# Must match the config used during training.
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero_spatial}"

# Server port and GPU.
PORT="${PORT:-8002}"
GPU="${GPU:-7}"
[[ -f "${CKPT_DIR}/model.safetensors" ]] || { echo "Missing ${CKPT_DIR}/model.safetensors" >&2; exit 2; }

# OpenPI root, used for placing logs/traces under `third_party/openpi/`.
OPENPI_ROOT="$(pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="${SERVER_LOG:-${OPENPI_ROOT}/runs/openpi_pi05_libero_server_${TS}.log}"
mkdir -p "${OPENPI_ROOT}/runs"

# ===== Norm stats check (required) =====
# For `pi05_libero_spatial`, OpenPI loads norm stats from the dataset directory (repo_id) by default.
NORM_STATS_PATH="/workspace/laiminxin/datasets/lerobot_datasets/libero_spatial/norm_stats.json"
if [[ ! -f "${NORM_STATS_PATH}" ]]; then
  echo "Missing norm stats: ${NORM_STATS_PATH}" >&2
  echo "Hint: cd third_party/openpi && uv run scripts/compute_norm_stats.py --config-name ${POLICY_CONFIG}" >&2
  exit 2
fi

echo "Serve: ckpt=${CKPT_DIR} GPU=${GPU} PORT=${PORT}"
echo "Client: HOST=127.0.0.1 PORT=${PORT} TRIALS=1 bash client_libero_eval.sh"
echo "Log: ${SERVER_LOG}"
echo "torch_compile: ${OPENPI_TORCH_COMPILE:-1}"
echo "triton_autotune: ${TRITON_AUTOTUNE:-<unset>}"
echo "torchinductor_max_autotune: ${TORCHINDUCTOR_MAX_AUTOTUNE:-<unset>}"

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

EXTRA_ARGS=()
if [[ "${TRACE}" == "1" ]]; then
  EXTRA_ARGS=(
    --trace-out-dir "${TRACE_OUT_DIR}"
    --trace-dump-attn
    --trace-attn-layers "${TRACE_ATTN_LAYERS}"
    --trace-every-n "${TRACE_EVERY_N}"
    --trace-max-dumps "${TRACE_MAX_DUMPS}"
  )
fi

export OPENPI_TORCH_COMPILE="${OPENPI_TORCH_COMPILE:-1}"
export OPENPI_TORCH_COMPILE_MODE="reduce-overhead"

CUDA_VISIBLE_DEVICES="${GPU}" uv run scripts/serve_policy.py \
  --env LIBERO --port "${PORT}" \
  --vla-opt-ve-film --vla-opt-ve-film-num-blocks 4 \
  --vla-opt-ste-prune --vla-opt-ste-prune-k 64 --vla-opt-ste-prune-stage gather --vla-opt-ste-prune-tau 1.0 \
  "${EXTRA_ARGS[@]}" \
  policy:checkpoint --policy.config "${POLICY_CONFIG}" --policy.dir "${CKPT_DIR}" 2>&1 | tee "${SERVER_LOG}"
