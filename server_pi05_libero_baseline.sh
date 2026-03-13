#!/usr/bin/env bash
set -euo pipefail

# Pi0.5 (LIBERO Spatial) baseline server launcher.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

die() { echo "Error: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  bash server_pi05_libero_baseline.sh [options...]

Options:
  --ckpt-dir <dir>            must contain model.safetensors
  --policy-config <name>      default: pi05_libero_spatial
  --gpu <id>                  default: 0 (CUDA_VISIBLE_DEVICES)
  --port <port>               default: 8002
  --log <path>                default: runs/openpi_pi05_libero_server_baseline_<ts>.log
  env OPENPI_TORCH_COMPILE    default: 1
EOF
}

ts="$(date +%Y%m%d_%H%M%S)"

policy_config="pi05_libero_spatial"
ckpt_dir="checkpoints/pi05_libero_spatial/pi05_baseline/30000"
gpu="0"
port="8002"

log_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --ckpt-dir) ckpt_dir="${2:?}"; shift 2 ;;
    --policy-config) policy_config="${2:?}"; shift 2 ;;
    --gpu) gpu="${2:?}"; shift 2 ;;
    --port) port="${2:?}"; shift 2 ;;
    --log) log_path="${2:?}"; shift 2 ;;
    *) die "Unknown option: $1 (run --help)" ;;
  esac
done

[[ -f "${ckpt_dir}/model.safetensors" ]] || die "Missing ${ckpt_dir}/model.safetensors"
command -v uv >/dev/null 2>&1 || die "Missing 'uv' in PATH"

norm_stats_path="/workspace/laiminxin/datasets/lerobot_datasets/libero_spatial/norm_stats.json"
if [[ "${policy_config}" == "pi05_libero_spatial" && ! -f "${norm_stats_path}" ]]; then
  echo "Missing norm stats: ${norm_stats_path}" >&2
  echo "Hint: cd third_party/openpi && uv run scripts/compute_norm_stats.py --config-name ${policy_config}" >&2
  exit 2
fi

mkdir -p runs

if [[ -z "${log_path}" ]]; then
  log_path="runs/openpi_pi05_libero_server_baseline_${ts}.log"
fi
mkdir -p "$(dirname "${log_path}")"

echo "=== OpenPI Server (baseline) ==="
echo "ckpt: ${ckpt_dir}"
echo "policy_config: ${policy_config}"
echo "gpu: ${gpu}"
echo "port: ${port}"
echo "log: ${log_path}"
echo "torch_compile: ${OPENPI_TORCH_COMPILE:-1}"
echo "triton_autotune: ${TRITON_AUTOTUNE:-<unset>}"
echo "torchinductor_max_autotune: ${TORCHINDUCTOR_MAX_AUTOTUNE:-<unset>}"
echo ""
echo "Client (example):"
echo "  HOST=127.0.0.1 PORT=${port} TRIALS=20 bash client_libero_eval_baseline.sh"
echo ""

export OPENPI_TORCH_COMPILE="${OPENPI_TORCH_COMPILE:-1}"
export OPENPI_TORCH_COMPILE_MODE="reduce-overhead"

CUDA_VISIBLE_DEVICES="${gpu}" uv run scripts/serve_policy.py \
  --env LIBERO --port "${port}" \
  policy:checkpoint --policy.config "${policy_config}" --policy.dir "${ckpt_dir}" 2>&1 | tee "${log_path}"
