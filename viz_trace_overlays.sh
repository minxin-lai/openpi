#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${script_dir}"

usage() {
  cat <<'EOF'
Usage:
  bash viz_trace_overlays.sh <exp_dir> [extra tracer.plot_routing_overlays args...]
  bash viz_trace_overlays.sh --last [extra args...]

Examples:
  bash viz_trace_overlays.sh runs/openpi_pi05_libero_trace_20260129_005355
  bash viz_trace_overlays.sh --last --heatmap_scale fixed --vmin 0 --vmax 0.003 --alpha 0.75 --cmap inferno
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

exp_dir="$1"
shift || true

if [[ "${exp_dir}" == "--help" || "${exp_dir}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${exp_dir}" == "--last" ]]; then
  last_file="runs/_last_openpi_trace_dir.txt"
  if [[ ! -f "${last_file}" ]]; then
    echo "Missing ${last_file}. Start a server to populate it, or pass exp_dir explicitly." >&2
    exit 2
  fi
  exp_dir="$(cat "${last_file}")"
fi

py="${PY:-${script_dir}/.venv/bin/python}"
if [[ ! -x "${py}" ]]; then
  echo "Python not found/executable at: ${py}" >&2
  echo "Hint: run in third_party/openpi after creating the venv (.venv)." >&2
  exit 2
fi

export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"

exec "${py}" -m tracer.plot_routing_overlays --exp_dir "${exp_dir}" "$@"

