#!/usr/bin/env bash
set -euo pipefail

# LIBERO eval client for the VLA-OPT server.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

die() { echo "Error: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  bash client_libero_eval_vla_opt.sh [options...]

Options:
  --host <ip>         default: 127.0.0.1
  --port <port>       default: 8003
  --suite <name>      default: libero_spatial (libero_spatial|libero_object|libero_goal|libero_10)
  --trials <n>        default: 20
  --gpu <id>          default: 0 (CUDA_VISIBLE_DEVICES)
  --video-out <path>  default: runs/libero/videos/vla_opt/<suite>_<ts>
  --log <path>        default: runs/libero/logs/vla_opt/<suite>_<ts>.log
EOF
}

ts="$(date +%Y%m%d_%H%M%S)"

# ======================
# Defaults
# ======================
host="127.0.0.1"
port="8003"
suite="libero_spatial"
trials="1"
gpu="0"

video_out=""
log_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --host) host="${2:?}"; shift 2 ;;
    --port) port="${2:?}"; shift 2 ;;
    --suite) suite="${2:?}"; shift 2 ;;
    --trials) trials="${2:?}"; shift 2 ;;
    --gpu) gpu="${2:?}"; shift 2 ;;
    --video-out) video_out="${2:?}"; shift 2 ;;
    --log) log_path="${2:?}"; shift 2 ;;
    *) die "Unknown option: $1 (run --help)" ;;
  esac
done

[[ -f "examples/libero/main.py" ]] || die "Run from third_party/openpi (missing examples/libero/main.py)"

venv_dir="examples/libero/.venv"
[[ -d "${venv_dir}" ]] || die "Venv not found: ${venv_dir} (create: uv venv --python 3.8 ${venv_dir})"

if [[ -z "${video_out}" ]]; then
  video_out="runs/libero/videos/vla_opt/${suite}_${ts}"
fi
if [[ -z "${log_path}" ]]; then
  log_path="runs/libero/logs/vla_opt/${suite}_${ts}.log"
fi
mkdir -p "$(dirname "${log_path}")"

echo "=== OpenPI LIBERO Client (vla_opt) ==="
echo "host: ${host}"
echo "port: ${port}"
echo "suite: ${suite}"
echo "trials: ${trials}"
echo "gpu: ${gpu}"
echo "video_out: ${video_out}"
echo "log: ${log_path}"
echo ""

# shellcheck disable=SC1090
source "${venv_dir}/bin/activate"
export PYTHONPATH="${PYTHONPATH:-}:$PWD/third_party/libero"

CUDA_VISIBLE_DEVICES="${gpu}" python examples/libero/main.py \
  --args.host "${host}" \
  --args.port "${port}" \
  --args.task-suite-name "${suite}" \
  --args.num-trials-per-task "${trials}" \
  --args.video-out-path "${video_out}" 2>&1 | tee "${log_path}"
