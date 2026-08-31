#!/bin/bash
# ============================================================
# Safety ASR Evaluation with Llama Guard 3 (Qwen3-8B, nothink mode)
#
# Usage:
#   bash evaluate_asr_qwen3_nothink.sh                                    # 최신 output_dir 자동 탐색
#   bash evaluate_asr_qwen3_nothink.sh <trained_model>                    # trained model 경로 지정
#   bash evaluate_asr_qwen3_nothink.sh <trained_model> <gpus>             # GPU 지정 (e.g. "0,1")
#   bash evaluate_asr_qwen3_nothink.sh <trained_model> <gpus> <exp_name>  # exp_name 지정

# Examples:
#   bash evaluate_asr_qwen3_nothink.sh data/output/Qwen3-8B_thinkoff_first_token_not_i_20260417_094234
#   bash evaluate_asr_qwen3_nothink.sh data/output/Qwen3-8B_thinkoff_first_token_not_i_20260417_094234 "0,1,2,3"
#   bash evaluate_asr_qwen3_nothink.sh --skip_trained                     # base model만 평가
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

CUDA_VISIBLE_DEVICES=4,5

PROJECT_ROOT="."
OUTPUT_BASE="${PROJECT_ROOT}/data/output"

# Activate virtual environment
# HuggingFace token (Llama Guard 접근용)
# Triton/Inductor JIT-compiles with `gcc -lcuda`; many hosts ship only
# libcuda.so.1. Point the linker at the CUDA stub so that compile succeeds.
export LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64/stubs:${LIBRARY_PATH:-}"

export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set in the environment}"

# ── Parse arguments ──
TRAINED_MODEL="${1:-}"
GPUS="${2:-4,5}"
EXP_NAME="${3:-}"

# --skip_trained 모드 처리
SKIP_TRAINED=""
if [ "$TRAINED_MODEL" = "--skip_trained" ]; then
    SKIP_TRAINED="--skip_trained"
    TRAINED_MODEL=""
fi

# If no trained_model given, use the most recent output dir
if [ -z "$TRAINED_MODEL" ] && [ -z "$SKIP_TRAINED" ]; then
    TRAINED_MODEL=$(ls -dt "${OUTPUT_BASE}"/*/ 2>/dev/null | head -1)
    if [ -z "$TRAINED_MODEL" ]; then
        echo "ERROR: No output directories found in ${OUTPUT_BASE}"
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
        echo "WARNING: No config.json found under ${TRAINED_MODEL}"
    fi
fi

echo "========================================================"
echo " SAFETY ASR EVALUATION (Llama Guard 3)"
echo "========================================================"
echo "  trained_model : ${TRAINED_MODEL:-SKIPPED}"
echo "  gpus          : ${GPUS}"
echo "  exp_name      : ${EXP_NAME:-asr_eval}"
echo "========================================================"


CMD="python ${ASR_DIR}/evaluate_asr_qwen3_nothink.py"
CMD="$CMD --base_model Qwen/Qwen3-8B"
CMD="$CMD --test_path ${PROJECT_ROOT}/data/eval/StrongREJECT/train.jsonl"
CMD="$CMD --result_dir ${PROJECT_ROOT}/data/result"
CMD="$CMD --gpus ${GPUS}"

if [ -n "$TRAINED_MODEL" ]; then
    CMD="$CMD --trained_model ${TRAINED_MODEL}"
fi

if [ -n "$EXP_NAME" ]; then
    CMD="$CMD --exp_name ${EXP_NAME}"
fi

if [ -n "$SKIP_TRAINED" ]; then
    CMD="$CMD --skip_trained"
fi

eval $CMD

echo ""
echo "========================================================"
echo " SAFETY ASR EVALUATION COMPLETE"
echo "========================================================"
if [ -n "$EXP_NAME" ]; then
    echo "Results saved to: ${PROJECT_ROOT}/data/result/${EXP_NAME}/"
fi
