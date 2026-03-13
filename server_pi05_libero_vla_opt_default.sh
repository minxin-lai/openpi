#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${script_dir}/server_pi05_libero_vla_opt.sh" \
  --run-tag vla_opt_default \
  "$@"
