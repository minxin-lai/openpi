#!/usr/bin/env bash
set -euo pipefail

# Pi0.5 (LIBERO Spatial) VLA-OPT server launcher.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"
repo_root="$(cd "${script_dir}/../.." && pwd)"

die() { echo "Error: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  bash server_pi05_libero_vla_opt.sh [options...]

Options:
  --ckpt-dir <dir>            must contain model.safetensors
  --policy-config <name>      default: pi05_libero_spatial
  --gpu <id>                  default: 0 (CUDA_VISIBLE_DEVICES)
  --port <port>               default: 8003
  --run-tag <tag>             default: vla_opt
  --log <path>                default: runs/<run_tag>/server_<ts>.log
  --ve-film-num-blocks <n>    default: 4
  --ste-prune-k <k>           default: 64
  --ste-prune-stage <stage>   default: gather
  --ste-prune-tau <tau>       default: 1.0
  --ste-prune-gaussian        default: off
  --ste-prune-gaussian-sigma <sigma>      default: 0.65
  --ste-prune-gaussian-kernel-size <odd>  default: auto
  --observe-config <path>     default: off
  env OPENPI_TORCH_COMPILE    default: 1
EOF
}

ts="$(date +%Y%m%d_%H%M%S)"

policy_config="pi05_libero_spatial"
ckpt_dir="checkpoints/pi05_libero_spatial/vla_opt_pi05_stage_a_ste_post_encoder_prune/29999"
gpu="0"
port="8003"
run_tag="vla_opt"

ve_film_num_blocks="4"
ste_prune_k="64"
ste_prune_stage="gather"
ste_prune_tau="1.0"
ste_prune_gaussian="0"
ste_prune_gaussian_sigma="0.65"
ste_prune_gaussian_kernel_size=""
observe_config=""
observe_dump_dir=""
observe_runtime_config=""

log_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --ckpt-dir) ckpt_dir="${2:?}"; shift 2 ;;
    --policy-config) policy_config="${2:?}"; shift 2 ;;
    --gpu) gpu="${2:?}"; shift 2 ;;
    --port) port="${2:?}"; shift 2 ;;
    --run-tag) run_tag="${2:?}"; shift 2 ;;
    --log) log_path="${2:?}"; shift 2 ;;
    --ve-film-num-blocks) ve_film_num_blocks="${2:?}"; shift 2 ;;
    --ste-prune-k) ste_prune_k="${2:?}"; shift 2 ;;
    --ste-prune-stage) ste_prune_stage="${2:?}"; shift 2 ;;
    --ste-prune-tau) ste_prune_tau="${2:?}"; shift 2 ;;
    --ste-prune-gaussian) ste_prune_gaussian="1"; shift 1 ;;
    --ste-prune-gaussian-sigma) ste_prune_gaussian_sigma="${2:?}"; shift 2 ;;
    --ste-prune-gaussian-kernel-size) ste_prune_gaussian_kernel_size="${2:?}"; shift 2 ;;
    --observe-config) observe_config="${2:?}"; shift 2 ;;
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
if [[ -n "${observe_config}" && ! -f "${observe_config}" ]]; then
  die "Missing observe config: ${observe_config}"
fi

mkdir -p runs

if [[ -z "${log_path}" ]]; then
  log_path="runs/${run_tag}/server_${ts}.log"
fi
mkdir -p "$(dirname "${log_path}")"

if [[ -n "${observe_config}" ]]; then
  observe_dump_dir="${script_dir}/runs/observe/${run_tag}_${ts}"
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

echo "=== OpenPI Server (vla_opt) ==="
echo "ckpt: ${ckpt_dir}"
echo "policy_config: ${policy_config}"
echo "gpu: ${gpu}"
echo "port: ${port}"
echo "run_tag: ${run_tag}"
echo "log: ${log_path}"
echo "prune: blocks=${ve_film_num_blocks} k=${ste_prune_k} stage=${ste_prune_stage} tau=${ste_prune_tau} gaussian=${ste_prune_gaussian} sigma=${ste_prune_gaussian_sigma} kernel=${ste_prune_gaussian_kernel_size:-auto}"
echo "observe_config: ${observe_config:-<off>}"
if [[ -n "${observe_dump_dir}" ]]; then
  echo "observe_dump_dir: ${observe_dump_dir}"
fi
echo "torch_compile: ${OPENPI_TORCH_COMPILE:-1}"
echo "triton_autotune: ${TRITON_AUTOTUNE:-<unset>}"
echo "torchinductor_max_autotune: ${TORCHINDUCTOR_MAX_AUTOTUNE:-<unset>}"

extra_args=()
if [[ -n "${observe_runtime_config}" ]]; then
  extra_args=(--vla-opt-observe-config "${observe_runtime_config}")
fi
if [[ "${ste_prune_gaussian}" == "1" ]]; then
  extra_args+=(--vla-opt-ste-prune-gaussian --vla-opt-ste-prune-gaussian-sigma "${ste_prune_gaussian_sigma}")
  if [[ -n "${ste_prune_gaussian_kernel_size}" ]]; then
    extra_args+=(--vla-opt-ste-prune-gaussian-kernel-size "${ste_prune_gaussian_kernel_size}")
  fi
fi

export OPENPI_TORCH_COMPILE="${OPENPI_TORCH_COMPILE:-1}"
export OPENPI_TORCH_COMPILE_MODE="reduce-overhead"

set +e
CUDA_VISIBLE_DEVICES="${gpu}" uv run scripts/serve_policy.py \
  --env LIBERO --port "${port}" \
  --vla-opt-ve-film --vla-opt-ve-film-num-blocks "${ve_film_num_blocks}" \
  --vla-opt-ste-prune --vla-opt-ste-prune-k "${ste_prune_k}" --vla-opt-ste-prune-stage "${ste_prune_stage}" --vla-opt-ste-prune-tau "${ste_prune_tau}" \
  "${extra_args[@]}" \
  policy:checkpoint --policy.config "${policy_config}" --policy.dir "${ckpt_dir}" 2>&1 | tee "${log_path}"
status=$?
set -e

echo ""
echo "server_log: ${log_path}"
if [[ -n "${observe_dump_dir}" ]]; then
  echo "observe_dump_dir: ${observe_dump_dir}"
fi
exit "${status}"
