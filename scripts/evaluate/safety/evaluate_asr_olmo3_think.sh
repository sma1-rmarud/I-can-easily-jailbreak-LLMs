#!/bin/bash
# ============================================================
# Safety ASR Evaluation — OLMo-3-7B-Think checkpoints
#
# Usage:
#   bash evaluate_asr_olmo3_think.sh <checkpoint_dir> [gpus]
#   bash evaluate_asr_olmo3_think.sh <checkpoint_dir> [gpus] [steps]
#
# Examples:
#   # evaluate step_30 and step_40 (default)
#   bash evaluate_asr_olmo3_think.sh \
#     data/output/Olmo-3-7B-Think_.../..._checkpoints
#
#   # evaluate specific steps with specific gpus
#   bash evaluate_asr_olmo3_think.sh \
#     data/output/Olmo-3-7B-Think_.../..._checkpoints \
#     "0,1,2,3" \
#     "step_30,step_40"
# ============================================================
set -e

# Run from the repo root so every path below is relative.
ASR_DIR="scripts/evaluate/safety"
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

PROJECT_ROOT="."

export PYTHONPATH="${PROJECT_ROOT}/open-instruct:${PYTHONPATH:-}"

# Triton/Inductor JIT-compiles with `gcc -lcuda`; many hosts ship only
# libcuda.so.1. Point the linker at the CUDA stub so that compile succeeds.
export LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64/stubs:${LIBRARY_PATH:-}"

export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set in the environment}"

# ── Parse arguments ──
CHECKPOINT_DIR="${1:-}"
GPUS="${2:-0,1}"
STEPS="${3:-step_30,step_40}"

if [ -z "$CHECKPOINT_DIR" ]; then
    # Auto-detect latest checkpoints dir
    CHECKPOINT_DIR=$(ls -dt "${PROJECT_ROOT}/data/output"/*/*_checkpoints 2>/dev/null | head -1)
    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "ERROR: No checkpoint directory found. Pass it as first argument."
        exit 1
    fi
    echo "Auto-detected checkpoint dir: ${CHECKPOINT_DIR}"
fi

echo "========================================================"
echo " ASR EVALUATION — OLMo-3-7B-Think"
echo "========================================================"
echo "  checkpoint_dir : ${CHECKPOINT_DIR}"
echo "  gpus           : ${GPUS}"
echo "  steps          : ${STEPS}"
echo "========================================================"

IFS=',' read -ra STEP_LIST <<< "$STEPS"
for STEP in "${STEP_LIST[@]}"; do
    STEP=$(echo "$STEP" | xargs)  # trim whitespace
    TRAINED_MODEL="${CHECKPOINT_DIR}/${STEP}"

    if [ ! -d "$TRAINED_MODEL" ]; then
        echo "WARNING: ${TRAINED_MODEL} not found, skipping."
        continue
    fi

    EXP_NAME="Olmo-3-7B-Think_${STEP}"

    echo ""
    echo "--------------------------------------------------------"
    echo " Evaluating: ${STEP}"
    echo " trained_model: ${TRAINED_MODEL}"
    echo "--------------------------------------------------------"

    python "${ASR_DIR}"/evaluate_asr_olmo3_think.py \
        --base_model allenai/OLMo-3-7B-Think \
        --trained_model "${TRAINED_MODEL}" \
        --test_path "${PROJECT_ROOT}/data/eval/StrongREJECT/train.jsonl" \
        --result_dir "${PROJECT_ROOT}/data/result" \
        --gpus "${GPUS}" \
        --exp_name "${EXP_NAME}" \
        --batch_size 4 \
        --max_new_tokens 4096

    echo "Results saved to: ${PROJECT_ROOT}/data/result/${EXP_NAME}/"
done

echo ""
echo "========================================================"
echo " ALL EVALUATIONS COMPLETE"
echo "========================================================"
