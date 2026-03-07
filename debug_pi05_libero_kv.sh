#!/usr/bin/env bash
set -euo pipefail

# Baseline vs VLA-OPT debug (token + KV cache shapes).
#
# 做什么：
# - 启动 server -> 跑 1-step client -> 从 server.log 抓 OPENPI_DEBUG 行 -> 停 server
#
# 用法：
#   cd third_party/openpi
#   bash debug_pi05_libero_kv.sh
#
# 输出：
#   runs/debug_kv_pi05_libero_YYYYMMDD_HHMMSS/{baseline,vla_opt}/server.log
#   runs/debug_kv_pi05_libero_YYYYMMDD_HHMMSS/{baseline,vla_opt}/timing.parquet
#
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

die() { echo "Error: $*" >&2; exit 2; }

# ======================
# 配置区（建议只改这里）
# ======================
policy_config="pi05_libero_spatial"

baseline_ckpt_dir="checkpoints/pi05_libero_spatial/pi05_baseline/30000"
vlaopt_ckpt_dir="checkpoints/pi05_libero_spatial/vla_opt_pi05_stage_a_ste/30000"

port_baseline="8102"
port_vlaopt="8103"

server_gpu="1"

debug_max_infer="1"
debug_kv_layers="ends"    # ends|all|0,8,16...
debug_kv_compare="false"  # true => best-effort prefix-preserved check

[[ -f "${baseline_ckpt_dir}/model.safetensors" ]] || die "Missing ${baseline_ckpt_dir}/model.safetensors"
[[ -f "${vlaopt_ckpt_dir}/model.safetensors" ]] || die "Missing ${vlaopt_ckpt_dir}/model.safetensors"
command -v uv >/dev/null 2>&1 || die "Missing 'uv' in PATH"

ts="$(date +%Y%m%d_%H%M%S)"
out_root="runs/debug_kv_pi05_libero_${ts}"
mkdir -p "${out_root}"

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

_run_one() {
  local label="$1"
  local ckpt_dir="$2"
  local port="$3"
  shift 3
  local -a extra_server_flags=("$@")

  local out_dir="${out_root}/${label}"
  mkdir -p "${out_dir}"

  local server_log="${out_dir}/server.log"

  echo "=== ${label} ==="
  echo "ckpt=${ckpt_dir}"
  echo "port=${port} server_gpu=${server_gpu}"
  echo "out=${out_dir}"
  echo ""

  set +e
  CUDA_VISIBLE_DEVICES="${server_gpu}" stdbuf -oL -eL uv run scripts/serve_policy.py \
    --env LIBERO --port "${port}" \
    --debug-token --debug-max-infer "${debug_max_infer}" \
    --debug-variant "${label}" \
    --debug-kv-layers "${debug_kv_layers}" \
    $([[ "${debug_kv_compare}" == "true" ]] && echo "--debug-kv-compare") \
    "${extra_server_flags[@]}" \
    policy:checkpoint --policy.config "${policy_config}" --policy.dir "${ckpt_dir}" \
    >"${server_log}" 2>&1 &
  local server_pid=$!
  set -e

  trap "_kill_and_wait ${server_pid}" EXIT

  for _ in $(seq 1 400); do
    if (echo >/dev/tcp/127.0.0.1/"${port}") >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "Server exited early. Tail log:" >&2
      tail -n 200 "${server_log}" >&2 || true
      return 1
    fi
    sleep 0.1
  done

  uv run python examples/simple_client/main.py \
    --env LIBERO --host 127.0.0.1 --port "${port}" \
    --num-steps 1 \
    --timing-file "${out_dir}/timing.parquet" >/dev/null 2>&1 || true

  _kill_and_wait "${server_pid}"
  trap - EXIT

  echo "OPENPI_DEBUG (from ${server_log}):"
  grep -n "OPENPI_DEBUG[[:space:]]\\+{" "${server_log}" | tail -n 1 || true
  echo ""
}

unset VLA_OPT_VE_FILM VLA_OPT_VE_FILM_NUM_BLOCKS
unset VLA_OPT_STE_PRUNE VLA_OPT_STE_PRUNE_K VLA_OPT_STE_PRUNE_STAGE VLA_OPT_STE_PRUNE_TAU
unset VLA_OPT_STE_PRUNE_LAYER VLA_OPT_STE_PRUNE_SCORE_MLP_HIDDEN_DIM

export OPENPI_TORCH_COMPILE="0"
export TRITON_AUTOTUNE=0

_run_one "baseline" "${baseline_ckpt_dir}" "${port_baseline}"

_run_one "vla_opt" "${vlaopt_ckpt_dir}" "${port_vlaopt}" \
  --vla-opt-ve-film --vla-opt-ve-film-num-blocks 4 \
  --vla-opt-ste-prune --vla-opt-ste-prune-k 64 --vla-opt-ste-prune-stage gather --vla-opt-ste-prune-tau 1.0

echo "All done: ${out_root}"
