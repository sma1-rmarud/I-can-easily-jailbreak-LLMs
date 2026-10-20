#!/usr/bin/env bash
# SFT uses OLMo-core 2.5.0, separately from the attack/DPO environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORE="${OLMO_CORE_DIR:-${ROOT}/external/OLMo-core}"
REVISION=2e570869bb3cfa93c0d637c823f4387d9cb4d804
VENV="${ROOT}/.venv-sft"
command -v uv >/dev/null || { echo "Install uv first." >&2; exit 1; }
if [[ ! -e "${CORE}" ]]; then
  mkdir -p "$(dirname "${CORE}")"
  git clone https://github.com/sma1-rmarud/OLMo-core.git "${CORE}"
  git -C "${CORE}" checkout --detach "${REVISION}"
elif [[ "$(git -C "${CORE}" rev-parse HEAD)" != "${REVISION}" ]]; then
  echo "OLMo-core revision mismatch: expected ${REVISION}" >&2
  exit 1
fi
if [[ ! -d "${VENV}" ]]; then uv venv --python 3.12 "${VENV}"; fi
uv pip install --python "${VENV}/bin/python" \
  -c "${ROOT}/eval_cap_noi/constraints.txt" \
  -e "${CORE}[transformers,wandb]" torch transformers datasets tqdm packaging ninja psutil setuptools wheel
uv pip install --python "${VENV}/bin/python" --no-build-isolation flash-attn==2.8.3
uv pip check --python "${VENV}/bin/python"
uv pip freeze --python "${VENV}/bin/python" > "${VENV}/resolved-requirements.txt"
