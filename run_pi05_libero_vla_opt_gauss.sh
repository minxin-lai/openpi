#!/usr/bin/env bash
set -euo pipefail

# One-click runner:
#   start server -> run LIBERO client -> render observe overlays -> aggregate pruning stats

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${script_dir}/run_libero_vla_opt_dump.sh" \
  --gauss \
  --run-tag vla_opt_gauss_dump \
  "$@"
