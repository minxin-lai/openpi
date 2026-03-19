#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
vla_opt_repo_dir="$(cd "${script_dir}/../../.." && pwd)"

die() { echo "Error: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  bash tools/serve_pi05_libero.sh [options...]

Options:
  --ckpt-dir <dir>            checkpoint directory containing model.safetensors
  --policy-config <name>      default: pi05_libero_spatial
  --opt-config <path>         optional experiment config
  --gpu <id>                  default: 0
  --port <port>               default: 8003
  --run-tag <tag>             optional run label
  --log <path>                optional explicit log path
  --observe-config <path>     optional observe config, experiment mode only
  --observe-output-dir <dir>  optional dump output dir, experiment mode only
EOF
}

ts="$(date +%Y%m%d_%H%M%S)"

ckpt_dir=""
policy_config="pi05_libero_spatial"
opt_config=""
gpu="0"
port="8003"
run_tag=""
log_path=""
observe_config=""
observe_output_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --ckpt-dir) ckpt_dir="${2:?}"; shift 2 ;;
    --policy-config) policy_config="${2:?}"; shift 2 ;;
    --opt-config) opt_config="${2:?}"; shift 2 ;;
    --gpu) gpu="${2:?}"; shift 2 ;;
    --port) port="${2:?}"; shift 2 ;;
    --run-tag) run_tag="${2:?}"; shift 2 ;;
    --log) log_path="${2:?}"; shift 2 ;;
    --observe-config) observe_config="${2:?}"; shift 2 ;;
    --observe-output-dir) observe_output_dir="${2:?}"; shift 2 ;;
    *) die "Unknown option: $1 (run --help)" ;;
  esac
done

[[ -n "${ckpt_dir}" ]] || die "--ckpt-dir is required"
cd "${repo_dir}"

[[ -f "${ckpt_dir}/model.safetensors" ]] || die "Missing ${ckpt_dir}/model.safetensors"
command -v uv >/dev/null 2>&1 || die "Missing 'uv' in PATH"

norm_stats_path="/workspace/laiminxin/datasets/lerobot_datasets/libero_spatial/norm_stats.json"
if [[ "${policy_config}" == "pi05_libero_spatial" && ! -f "${norm_stats_path}" ]]; then
  echo "Missing norm stats: ${norm_stats_path}" >&2
  echo "Hint: cd third_party/openpi && uv run scripts/compute_norm_stats.py --config-name ${policy_config}" >&2
  exit 2
fi

if [[ -n "${opt_config}" && ! -f "${opt_config}" ]]; then
  die "Missing opt config: ${opt_config}"
fi

if [[ -n "${observe_config}" && ! -f "${observe_config}" ]]; then
  die "Missing observe config: ${observe_config}"
fi

if [[ (-n "${observe_config}" || -n "${observe_output_dir}") && -z "${opt_config}" ]]; then
  die "--observe-config/--observe-output-dir require --opt-config"
fi

mkdir -p runs
if [[ -z "${log_path}" ]]; then
  if [[ -n "${run_tag}" ]]; then
    log_path="runs/${run_tag}/server_${ts}.log"
  elif [[ -n "${opt_config}" ]]; then
    log_path="runs/openpi_pi05_libero_server_opt_${ts}.log"
  else
    log_path="runs/openpi_pi05_libero_server_baseline_${ts}.log"
  fi
fi
mkdir -p "$(dirname "${log_path}")"

observe_runtime_config=""
observe_dump_dir=""
if [[ -n "${observe_config}" ]]; then
  if [[ -n "${observe_output_dir}" ]]; then
    observe_dump_dir="${observe_output_dir}"
  elif [[ -n "${run_tag}" ]]; then
    observe_dump_dir="${repo_dir}/runs/observe/${run_tag}_${ts}"
  else
    observe_dump_dir="${repo_dir}/runs/observe/openpi_pi05_libero_${ts}"
  fi
  mkdir -p "${observe_dump_dir}"
  observe_runtime_config="${observe_dump_dir}/observe_config.json"
  python3 - "${observe_config}" "${observe_runtime_config}" "${observe_dump_dir}" <<'PY'
import json
import sys
from pathlib import Path

src_path = Path(sys.argv[1])
dst_path = Path(sys.argv[2])
output_dir = sys.argv[3]

with src_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, dict):
    raise ValueError(f"Observe config must be a JSON object: {src_path}")

data["output_dir"] = output_dir

with dst_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=True, indent=2)
    f.write("\n")
PY
fi

echo "=== OpenPI Server ==="
echo "ckpt: ${ckpt_dir}"
echo "policy_config: ${policy_config}"
echo "gpu: ${gpu}"
echo "port: ${port}"
echo "run_tag: ${run_tag:-<unset>}"
echo "log: ${log_path}"
echo "opt_config: ${opt_config:-<off>}"
echo "observe_config: ${observe_config:-<off>}"
if [[ -n "${observe_dump_dir}" ]]; then
  echo "observe_dump_dir: ${observe_dump_dir}"
fi
echo "torch_compile: ${OPENPI_TORCH_COMPILE:-<openpi-default>}"
echo "triton_autotune: ${TRITON_AUTOTUNE:-<unset>}"
echo "torchinductor_max_autotune: ${TORCHINDUCTOR_MAX_AUTOTUNE:-<unset>}"
echo

extra_args=()
if [[ -n "${opt_config}" ]]; then
  extra_args+=(--vla-opt-pruning-config "${opt_config}")
fi
if [[ -n "${observe_runtime_config}" ]]; then
  extra_args+=(--vla-opt-observe-config "${observe_runtime_config}")
fi

if [[ -z "${opt_config}" ]]; then
  export OPENPI_TORCH_COMPILE="${OPENPI_TORCH_COMPILE:-1}"
  export OPENPI_TORCH_COMPILE_MODE="${OPENPI_TORCH_COMPILE_MODE:-reduce-overhead}"
fi

set +e
CUDA_VISIBLE_DEVICES="${gpu}" uv run scripts/serve_policy.py \
  --env LIBERO --port "${port}" \
  "${extra_args[@]}" \
  policy:checkpoint --policy.config "${policy_config}" --policy.dir "${ckpt_dir}" 2>&1 | tee "${log_path}"
status=$?
set -e

echo
echo "server_log: ${log_path}"
if [[ -n "${observe_dump_dir}" ]]; then
  echo "observe_dump_dir: ${observe_dump_dir}"
fi
exit "${status}"
