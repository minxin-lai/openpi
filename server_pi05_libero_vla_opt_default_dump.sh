#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${script_dir}/run_libero_vla_opt_dump.sh" \
  --run-tag vla_opt_default_dump \
  "$@"
