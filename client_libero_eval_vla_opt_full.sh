#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${script_dir}/client_libero_eval_vla_opt.sh" \
  --run-tag vla_opt_full \
  --trials 20 \
  "$@"
