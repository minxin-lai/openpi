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

Stage2 consensus viz (optional):
  --no-consensus-viz            disable shared/unique overlays (default: enabled)
  --consensus-pair <a,b>        default: 0,1
  --consensus-threshold <t>     default: 0.4
  --consensus-margin <m>        default: 0.0
  --consensus-max <n>           default: 0 (no limit)

Output:
  --plots-subdir <name>         default: plots (lets you keep multiple plots dirs per run, e.g. plots_thr0.6_m0.1)
  --no-concat-views             (passed through) disable per-sample stitched `__views.png` outputs

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

consensus_viz="true"
consensus_pair="0,1"
consensus_threshold="0.4"
consensus_margin="0.0"
consensus_max="0"
plots_subdir="plots"

extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-consensus-viz) consensus_viz="false"; shift ;;
    --consensus-pair) consensus_pair="${2:?}"; shift 2 ;;
    --consensus-threshold) consensus_threshold="${2:?}"; shift 2 ;;
    --consensus-margin) consensus_margin="${2:?}"; shift 2 ;;
    --consensus-max) consensus_max="${2:?}"; shift 2 ;;
    --plots-subdir) plots_subdir="${2:?}"; shift 2 ;;
    *) extra_args+=("$1"); shift ;;
  esac
done

# Pass through only args that viz_consensus.py understands.
consensus_extra_args=()
for ((i=0; i<${#extra_args[@]}; i++)); do
  a="${extra_args[$i]}"
  case "${a}" in
    --alpha|--cmap|--concat-pad-px)
      if (( i + 1 < ${#extra_args[@]} )); then
        consensus_extra_args+=("${a}" "${extra_args[$((i+1))]}")
        i=$((i+1))
      fi
      ;;
    --no-concat-views)
      consensus_extra_args+=("${a}")
      ;;
  esac
done

py="${PY:-${script_dir}/.venv/bin/python}"
if [[ ! -x "${py}" ]]; then
  echo "Python not found/executable at: ${py}" >&2
  echo "Hint: run in third_party/openpi after creating the venv (.venv)." >&2
  exit 2
fi

export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"

"${py}" -m tracer.plot_routing_overlays --exp_dir "${exp_dir}" --plots-subdir "${plots_subdir}" "${extra_args[@]}"

if [[ "${consensus_viz}" == "true" ]]; then
  # Best-effort: generate shared/unique overlays on the (post-prune) patch grid.
  # This requires routing.keep_indices/keep_scores/keep_tokens in dumps (VLA-OPT tracer).
  if ! "${py}" "${repo_root}/tools/view_consensus/viz_consensus.py" \
    --input "${exp_dir}/dumps" \
    --pair "${consensus_pair}" \
    --threshold "${consensus_threshold}" \
    --margin "${consensus_margin}" \
    --out-dir "${exp_dir}/${plots_subdir}/consensus" \
    --max "${consensus_max}" \
    "${consensus_extra_args[@]}" >/dev/null; then
    echo "WARN: consensus viz failed (continuing). exp_dir=${exp_dir}" >&2
  fi
fi
