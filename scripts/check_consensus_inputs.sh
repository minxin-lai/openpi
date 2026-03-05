#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

py="${PY:-${repo_root}/.venv/bin/python}"
if [[ ! -x "${py}" ]]; then
  echo "Python not found/executable at: ${py}" >&2
  echo "Hint: run in third_party/openpi after creating the venv (.venv)." >&2
  exit 2
fi

exec "${py}" "${repo_root}/scripts/check_consensus_inputs.py" "$@"

