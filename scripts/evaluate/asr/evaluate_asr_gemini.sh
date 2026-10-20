#!/usr/bin/env bash
# Explicit model selection; the same judge is used for both Table 5 comparison groups.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
MODEL="${1:?Usage: bash scripts/evaluate/asr/evaluate_asr_gemini.sh MODEL [GPUS] [--dry-run]}"
GPUS="${2:-0}"
DRY_ARGS=()
if [[ "${3:-}" == "--dry-run" ]]; then DRY_ARGS=(--dry-run); fi
if [[ -n "${3:-}" && "${3}" != "--dry-run" ]]; then echo "Unknown argument: ${3}" >&2; exit 2; fi
if [[ -n "${RLVR_VENV:-}" ]]; then source "${RLVR_VENV}/bin/activate"; fi
RESULT_DIR="${RESULT_DIR:-data/result/gemini/$(date +%Y%m%d_%H%M%S)}"
MODEL_TAG="${MODEL//\//__}"
BENCHMARKS="${BENCHMARKS:-jbb|data/eval/JBB-Behaviors/harmful.jsonl|Goal;strongreject|data/eval/StrongREJECT/train.jsonl|prompt}"
EXTRA=()
if [[ -n "${REVISION:-}" ]]; then EXTRA+=(--revision "${REVISION}"); fi
if [[ -n "${LIMIT:-}" ]]; then EXTRA+=(--limit "${LIMIT}"); fi
if [[ ${#DRY_ARGS[@]} == 0 ]]; then
  : "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY for the Gemini judge}"
  if [[ -e "${RESULT_DIR}/${MODEL_TAG}" ]]; then
    echo "Output exists; select a new RESULT_DIR to avoid judging old runs." >&2
    exit 1
  fi
fi
python scripts/evaluate/asr/generate_responses.py \
  --model "${MODEL}" --model_tag "${MODEL_TAG}" --gpus "${GPUS}" \
  --benchmarks "${BENCHMARKS}" --result_dir "${RESULT_DIR}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-4096}" --temperature "${TEMPERATURE:-0.6}" \
  --top_p "${TOP_P:-0.95}" --seed "${SEED:-42}" "${EXTRA[@]}" "${DRY_ARGS[@]}"
if [[ ${#DRY_ARGS[@]} == 0 ]]; then
  python scripts/analyze/safety_judge_gemini.py \
    --input_root "${RESULT_DIR}/${MODEL_TAG}" \
    --model "${GEMINI_MODEL:-google/gemini-3-flash-preview}" \
    --concurrency "${GEMINI_CONCURRENCY:-15}"
fi
