#!/usr/bin/env bash
# Restore the exact harness revision and overlays without discarding local edits.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS="${SCRIPT_DIR}/lm-evaluation-harness"
REVISION=d800e04dcb1ce96791d8b2926cf0cc7703d58457
VENV="${SCRIPT_DIR}/.venv"
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
if [[ ! -e "${HARNESS}" ]]; then
  git clone https://github.com/EleutherAI/lm-evaluation-harness.git "${HARNESS}"
  git -C "${HARNESS}" checkout --detach "${REVISION}"
elif [[ "$(git -C "${HARNESS}" rev-parse HEAD)" != "${REVISION}" ]]; then
  echo "Harness is at a different revision. Move it aside or check it out manually at ${REVISION}." >&2
  exit 1
fi
python "${SCRIPT_DIR}/apply_patches.py"
if [[ ! -d "${VENV}" ]]; then
  uv venv --python 3.12 "${VENV}"
fi
uv pip install --python "${VENV}/bin/python" \
  -c "${SCRIPT_DIR}/constraints.txt" -e "${HARNESS}[vllm,math]" math-verify
uv pip check --python "${VENV}/bin/python"
uv pip freeze --python "${VENV}/bin/python" > "${VENV}/resolved-requirements.txt"
echo "Installed. See ${SCRIPT_DIR}/README.md for smoke and full evaluation commands."
