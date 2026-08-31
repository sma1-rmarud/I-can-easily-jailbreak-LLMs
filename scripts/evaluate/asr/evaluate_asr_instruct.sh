#!/bin/bash
# ============================================================
# Safety ASR evaluation with Llama Guard 3 — plain instruct models.
# Covers any model without a think/no-think toggle (Llama-3.x, Qwen2.5, ...).
# Pick the base model with BASE_MODEL; everything else is identical.
#
# Usage:
#   bash evaluate_asr_instruct.sh                                    # 최신 output_dir 자동 탐색
#   bash evaluate_asr_instruct.sh <trained_model>                    # trained model 경로 지정
#   bash evaluate_asr_instruct.sh <trained_model> <gpus>             # GPU 지정 (e.g. "0,1")
#   bash evaluate_asr_instruct.sh <trained_model> <gpus> <exp_name>  # exp_name 지정
#   bash evaluate_asr_instruct.sh --skip_trained                     # base model 만 평가
#
# Env:
#   BASE_MODEL  평가할 base 모델 (기본: meta-llama/Llama-3.1-8B-Instruct)
#   RLVR_ROOT   저장소 루트 (기본: 스크립트 위치에서 역산)
#   RLVR_VENV   activate 할 virtualenv (기본: .venv → venv → /opt/venv 순 탐색)
#   HF_TOKEN    필수 — Llama Guard 3 접근용
#
# Example:
#   BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
#     bash evaluate_asr_instruct.sh <RLVR_ROOT>/data/output/<exp> "0,1"
# ============================================================
set -e

# Run from the repo root so every path below is relative.
ASR_DIR="scripts/evaluate/asr"
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
RLVR_VENV="${RLVR_VENV:-}"
if [ -z "${RLVR_VENV}" ]; then
    for _cand in .venv venv /opt/venv; do
        [ -f "${_cand}/bin/activate" ] && RLVR_VENV="${_cand}" && break
    done
fi
if [ -n "${RLVR_VENV}" ] && [ -f "${RLVR_VENV}/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${RLVR_VENV}/bin/activate"
else
    echo "ERROR: no virtualenv found. Set RLVR_VENV=/path/to/venv" >&2
    exit 1
fi
OUTPUT_BASE="data/output"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# Llama Guard 3 is gated — fail fast rather than 401 deep inside vLLM.
# Triton/Inductor JIT-compiles with `gcc -lcuda`; many hosts ship only
# libcuda.so.1. Point the linker at the CUDA stub so that compile succeeds.
export LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64/stubs:${LIBRARY_PATH:-}"

export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set in the environment}"

# ── Parse arguments ──
TRAINED_MODEL="${1:-}"
GPUS="${2:-0,1}"
EXP_NAME="${3:-}"

SKIP_TRAINED=""
if [ "$TRAINED_MODEL" = "--skip_trained" ]; then
    SKIP_TRAINED="--skip_trained"
    TRAINED_MODEL=""
fi

# If no trained_model given, use the most recent output dir
if [ -z "$TRAINED_MODEL" ] && [ -z "$SKIP_TRAINED" ]; then
    TRAINED_MODEL=$(ls -dt "${OUTPUT_BASE}"/*/ 2>/dev/null | head -1)
    if [ -z "$TRAINED_MODEL" ]; then
        echo "ERROR: No output directories found in ${OUTPUT_BASE}" >&2
        exit 1
    fi
    echo "Auto-detected latest output dir: ${TRAINED_MODEL}"
fi

# Derive exp_name from trained_model basename if not set
if [ -z "$EXP_NAME" ] && [ -n "$TRAINED_MODEL" ]; then
    EXP_NAME=$(basename "$TRAINED_MODEL")
fi

# Resolve to actual model directory (find subdir with config.json)
if [ -n "$TRAINED_MODEL" ] && [ ! -f "${TRAINED_MODEL}/config.json" ]; then
    RESOLVED=$(find "$TRAINED_MODEL" -maxdepth 2 -name "config.json" -not -path "*checkpoint*" -not -path "*step_*" 2>/dev/null | head -1)
    if [ -n "$RESOLVED" ]; then
        TRAINED_MODEL=$(dirname "$RESOLVED")
        echo "Resolved model directory: ${TRAINED_MODEL}"
    else
        echo "WARNING: No config.json found under ${TRAINED_MODEL}" >&2
    fi
fi

echo "========================================================"
echo " SAFETY ASR EVALUATION (Llama Guard 3)"
echo "========================================================"
echo "  base_model    : ${BASE_MODEL}"
echo "  trained_model : ${TRAINED_MODEL:-SKIPPED}"
echo "  gpus          : ${GPUS}"
echo "  exp_name      : ${EXP_NAME:-asr_eval}"
echo "========================================================"


CMD=(python "${ASR_DIR}/evaluate_asr_instruct.py"
     --base_model "${BASE_MODEL}"
     --test_path  "data/eval/StrongREJECT/train.jsonl"
     --result_dir "data/result"
     --gpus       "${GPUS}")

if [ -n "$TRAINED_MODEL" ]; then CMD+=(--trained_model "${TRAINED_MODEL}"); fi
if [ -n "$EXP_NAME" ];      then CMD+=(--exp_name "${EXP_NAME}"); fi
if [ -n "$SKIP_TRAINED" ];  then CMD+=(--skip_trained); fi

"${CMD[@]}"

echo ""
echo "========================================================"
echo " SAFETY ASR EVALUATION COMPLETE"
echo "========================================================"
if [ -n "$EXP_NAME" ]; then
    echo "Results saved to: data/result/${EXP_NAME}/"
fi
