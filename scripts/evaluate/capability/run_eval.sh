#!/usr/bin/env bash
# One explicitly selected model; task settings are defined in run.py.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="${VENV_ACTIVATE:-${SCRIPT_DIR}/.venv/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  source "${VENV_ACTIVATE}"
fi
exec python "${SCRIPT_DIR}/run.py" "$@"
