#!/usr/bin/env bash
set -euo pipefail

# Baseline vs VLA-OPT *pure inference* perf comparison.
#
# 做什么：
# - 启动 baseline server -> 跑 micro-benchmark client -> 采样 nvidia-smi -> 停 server
# - 再对 vla-opt 重复一遍
# - 默认关闭 tracer dump（避免掩盖加速效果）
#
# 用法：
#   cd third_party/openpi
#   bash compare_pi05_libero_perf.sh
#
# 输出：
#   runs/perf_pi05_libero_YYYYMMDD_HHMMSS/{baseline,vla_opt}/timing.parquet

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

die() { echo "Error: $*" >&2; exit 2; }

# ======================
# 配置区（建议只改这里）
# ======================
policy_config="pi05_libero_spatial"

baseline_ckpt_dir="checkpoints/pi05_libero_spatial/pi05_baseline/30000"
vlaopt_ckpt_dir="checkpoints/pi05_libero_spatial/vla_opt_pi05_stage_a_ste/30000"

port_baseline="8002"
port_vlaopt="8003"

# server 跑在哪张 GPU（CUDA_VISIBLE_DEVICES）
server_gpu="1"
# nvidia-smi 采样的物理 GPU index（通常同 server_gpu）
smi_gpu_index="1"

num_steps="500"
smi_interval_ms="200"

# 减少“第一次推理”编译/抖动
openpi_torch_compile="1"
openpi_torch_compile_mode="reduce-overhead"  # reduce-overhead|max-autotune|default

wait_timeout_sec="600"
wait_poll_sec="0.25"

[[ -f "${baseline_ckpt_dir}/model.safetensors" ]] || die "Missing ${baseline_ckpt_dir}/model.safetensors"
[[ -f "${vlaopt_ckpt_dir}/model.safetensors" ]] || die "Missing ${vlaopt_ckpt_dir}/model.safetensors"
command -v uv >/dev/null 2>&1 || die "Missing 'uv' in PATH"
command -v nvidia-smi >/dev/null 2>&1 || die "Missing 'nvidia-smi' in PATH"

ts="$(date +%Y%m%d_%H%M%S)"
out_root="runs/perf_pi05_libero_${ts}"
mkdir -p "${out_root}"

has_curl="false"
command -v curl >/dev/null 2>&1 && has_curl="true"

_wait_port() {
  local port="$1"
  if [[ "${has_curl}" == "true" ]]; then
    curl -fsS "http://127.0.0.1:${port}/healthz" >/dev/null 2>&1 && return 0
    return 1
  fi
  (echo >/dev/tcp/127.0.0.1/"${port}") >/dev/null 2>&1 && return 0
  return 1
}

_kill_and_wait() {
  local pid="$1"
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    for _ in $(seq 1 80); do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        sleep 0.1
      else
        break
      fi
    done
  fi
}

_wait_server_ready() {
  local port="$1"
  local server_pid="$2"
  local server_log="$3"

  local start_s
  start_s="$(date +%s)"
  while true; do
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "Server exited early (pid=${server_pid}). Tail log:" >&2
      tail -n 200 "${server_log}" >&2 || true
      return 1
    fi

    if _wait_port "${port}"; then
      return 0
    fi

    local now_s
    now_s="$(date +%s)"
    if (( now_s - start_s > wait_timeout_sec )); then
      echo "Timeout waiting for server port ${port} after ${wait_timeout_sec}s. Tail log:" >&2
      tail -n 200 "${server_log}" >&2 || true
      return 1
    fi
    sleep "${wait_poll_sec}"
  done
}

_run_one() {
  local label="$1"
  local ckpt_dir="$2"
  local port="$3"
  shift 3
  local -a extra_server_flags=("$@")

  local out_dir="${out_root}/${label}"
  mkdir -p "${out_dir}"

  local server_log="${out_dir}/server.log"
  local smi_csv="${out_dir}/nvidia_smi.csv"
  local timing_parquet="${out_dir}/timing.parquet"

  echo "=== ${label} ==="
  echo "ckpt=${ckpt_dir}"
  echo "port=${port} server_gpu=${server_gpu}"
  echo "out=${out_dir}"
  echo ""

  set +e
  CUDA_VISIBLE_DEVICES="${server_gpu}" stdbuf -oL -eL uv run scripts/serve_policy.py \
    --env LIBERO --port "${port}" \
    "${extra_server_flags[@]}" \
    policy:checkpoint --policy.config "${policy_config}" --policy.dir "${ckpt_dir}" \
    >"${server_log}" 2>&1 &
  local server_pid=$!
  set -e

  trap "_kill_and_wait ${server_pid}" EXIT
  _wait_server_ready "${port}" "${server_pid}" "${server_log}"

  set +e
  nvidia-smi -i "${smi_gpu_index}" \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -lms "${smi_interval_ms}" >"${smi_csv}" 2>/dev/null &
  local smi_pid=$!
  set -e

  uv run python examples/simple_client/main.py \
    --env LIBERO --host 127.0.0.1 --port "${port}" \
    --num-steps "${num_steps}" \
    --timing-file "${timing_parquet}"

  _kill_and_wait "${smi_pid}" || true
  _kill_and_wait "${server_pid}"
  trap - EXIT

  echo ""
  echo "done: ${label}"
  echo "  server_log: ${server_log}"
  echo "  timing:     ${timing_parquet}"
  echo "  nvsmi:      ${smi_csv}"
  echo ""
}

export TRITON_AUTOTUNE=0
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export OPENPI_TORCH_COMPILE="${openpi_torch_compile}"
export OPENPI_TORCH_COMPILE_MODE="${openpi_torch_compile_mode}"

_run_one "baseline" "${baseline_ckpt_dir}" "${port_baseline}"

_run_one "vla_opt" "${vlaopt_ckpt_dir}" "${port_vlaopt}" \
  --vla-opt-ve-film --vla-opt-ve-film-num-blocks 4 \
  --vla-opt-ste-prune --vla-opt-ste-prune-k 64 --vla-opt-ste-prune-stage gather --vla-opt-ste-prune-tau 1.0

echo "All done: ${out_root}"
