#!/usr/bin/env bash
# Unified parallel ASR eval — all not-I RLVR models
#
# Each (base vs trained) pair uses 4 GPUs.
# With 8 GPUs, 2 pairs run simultaneously: slot A (GPUs 0-3), slot B (GPUs 4-7).
# Models with empty trained_model paths are skipped (placeholder).
#
# Usage:
#   bash evaluate_asr_all_parallel.sh
#   SLOT_A_GPUS="0,1,2,3" SLOT_B_GPUS="4,5,6,7" bash evaluate_asr_all_parallel.sh

set -euo pipefail

# Run from the repo root so every path below is relative.
ASR_DIR="scripts/evaluate/asr"
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# ── Paths ─────────────────────────────────────────────────────────────────────
# Benchmarks to run, as "label|path". Each is a JSONL shipped in this repo.
# Override with BENCHMARKS, same "label|path" form, space separated.
read -ra ALL_BENCHMARKS <<< "${BENCHMARKS:-strongreject|data/eval/StrongREJECT/train.jsonl jbb_harmful|data/eval/JBB-Behaviors/harmful.jsonl}"
RESULT_DIR="data/result"

# ── GPU slots (4 GPUs each) ───────────────────────────────────────────────────
SLOT_A_GPUS="${SLOT_A_GPUS:-0,1,2,3}"
SLOT_B_GPUS="${SLOT_B_GPUS:-4,5,6,7}"

# ── Model registry ────────────────────────────────────────────────────────────
# Format: "key|py_script|base_model|trained_model"
# Leave trained_model empty to skip.
declare -a MODEL_ENTRIES=(

  # ── Instruct ─────────────────────────────────────────────────────────────

  # OLMo-3-7B-Instruct
  "olmo3_7b_inst_step50|evaluate_asr_olmo3_think.py|allenai/Olmo-3-7B-Instruct|uos-nlp/olmo-3-7b-instruct-not-i-step50"

  # Qwen2.5-Instruct
  "qwen25_3b_inst_step110|evaluate_asr_instruct.py|Qwen/Qwen2.5-3B-Instruct|uos-nlp/qwen-2.5-3b-instruct-not-i-step110"
  "qwen25_7b_inst_step110|evaluate_asr_instruct.py|Qwen/Qwen2.5-7B-Instruct|uos-nlp/qwen-2.5-7b-instruct-not-i-step110"

  # Llama-3.x-Instruct
  "llama32_3b_inst_step75|evaluate_asr_instruct.py|meta-llama/Llama-3.2-3B-Instruct|sma1-rmarud/llama-3.2-3b-instruct-not-i-step75"
  "llama31_8b_inst_step40|evaluate_asr_instruct.py|meta-llama/Llama-3.1-8B-Instruct|sma1-rmarud/llama-3.1-8b-instruct-not-i-step40"

  # Qwen3-8B (nothink)
  "qwen3_8b_nothink_step100|evaluate_asr_qwen3_nothink.py|Qwen/Qwen3-8B|sma1-rmarud/qwen-3-8b-thinkoff-not-i-step100"
  "qwen3_8b_nothink_olmo_step40|evaluate_asr_qwen3_nothink.py|Qwen/Qwen3-8B|sma1-rmarud/qwen-3-8b-thinkoff-not-i-olmo-step40"

  # ── Reasoning ────────────────────────────────────────────────────────────

  # OLMo-3-7B-Think
  "olmo3_7b_think_step40|evaluate_asr_olmo3_think.py|allenai/Olmo-3-7B-Think|sma1-rmarud/Olmo-3-7B-Think-first-token-not-i-step40"

  # Qwen3-8B (think)
  "qwen3_8b_think_step50|evaluate_asr_qwen3_reasoning.py|Qwen/Qwen3-8B|sma1-rmarud/qwen-3-8b-think-not-i-step50"

  # STAR1-7B — placeholder (fill in trained_model to enable)
  "star1_7b|evaluate_asr_star1_r1_reasoning.py|UCSC-VLAA/STAR1-R1-Distill-7B|"
)

# ── Environment ───────────────────────────────────────────────────────────────
# Triton/Inductor JIT-compiles with `gcc -lcuda`; many hosts ship only
# libcuda.so.1. Point the linker at the CUDA stub so that compile succeeds.
export LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64/stubs:${LIBRARY_PATH:-}"

export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set in the environment}"
# Extra native libs (e.g. a conda/mamba lib dir) can be prepended via EXTRA_LIB_DIR.
export LD_LIBRARY_PATH="${EXTRA_LIB_DIR:+${EXTRA_LIB_DIR}:}${LD_LIBRARY_PATH:-}"

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

# ── Filter active entries ─────────────────────────────────────────────────────
declare -a active_entries=()
declare -a skipped_keys=()
for entry in "${MODEL_ENTRIES[@]}"; do
  IFS='|' read -r key py base trained <<< "${entry}"
  if [[ -n "${trained}" ]]; then
    active_entries+=("${entry}")
  else
    skipped_keys+=("${key}")
  fi
done

echo "=========================================="
echo "Parallel ASR eval — all not-I RLVR models"
echo "  ASR_DIR  : ${ASR_DIR}"
echo "  Slot A   : GPUs ${SLOT_A_GPUS}"
echo "  Slot B   : GPUs ${SLOT_B_GPUS}"
echo "  Models   : ${#active_entries[@]} active / ${#skipped_keys[@]} skipped"
[[ ${#skipped_keys[@]} -gt 0 ]] && echo "  Skipped  : ${skipped_keys[*]}"
echo "  Benchmarks: ${ALL_BENCHMARKS[*]}"
echo "=========================================="

# ── 2-slot job queue ──────────────────────────────────────────────────────────
slot_a_pid=""
slot_b_pid=""
slot_a_key=""
slot_b_key=""
declare -a failed_jobs=()

run_job() {
  local key="$1" py="$2" base="$3" trained="$4" gpus="$5"

  for benchmark in "${ALL_BENCHMARKS[@]}"; do
    IFS="|" read -r bench_label bench_path <<< "${benchmark}"
    local exp_name="${key}__${bench_label}"
    local log="${RESULT_DIR}/${exp_name}/_asr_worker.log"
    mkdir -p "${RESULT_DIR}/${exp_name}"

    echo "[launch] ${exp_name} → GPUs ${gpus}"

    python "${ASR_DIR}/${py}" \
      --base_model    "${base}" \
      --trained_model "${trained}" \
      --test_path     "${bench_path}" \
      --result_dir    "${RESULT_DIR}" \
      --exp_name      "${exp_name}" \
      --gpus          "${gpus}" \
      >> "${log}" 2>&1
  done
}

wait_slot() {
  local slot="$1"   # "a" or "b"
  if [[ "${slot}" == "a" && -n "${slot_a_pid}" ]]; then
    if wait "${slot_a_pid}"; then
      echo "[done  ] ${slot_a_key} (slot A) ok"
    else
      echo "[FAIL  ] ${slot_a_key} (slot A)"
      failed_jobs+=("${slot_a_key}")
    fi
    slot_a_pid=""
    slot_a_key=""
  elif [[ "${slot}" == "b" && -n "${slot_b_pid}" ]]; then
    if wait "${slot_b_pid}"; then
      echo "[done  ] ${slot_b_key} (slot B) ok"
    else
      echo "[FAIL  ] ${slot_b_key} (slot B)"
      failed_jobs+=("${slot_b_key}")
    fi
    slot_b_pid=""
    slot_b_key=""
  fi
}

for entry in "${active_entries[@]}"; do
  IFS='|' read -r key py base trained <<< "${entry}"

  # Find a free slot (prefer A, then B, else wait for A)
  if [[ -z "${slot_a_pid}" ]]; then
    (run_job "${key}" "${py}" "${base}" "${trained}" "${SLOT_A_GPUS}") &
    slot_a_pid=$!
    slot_a_key="${key}"

  elif [[ -z "${slot_b_pid}" ]]; then
    (run_job "${key}" "${py}" "${base}" "${trained}" "${SLOT_B_GPUS}") &
    slot_b_pid=$!
    slot_b_key="${key}"

  else
    # Both slots busy — poll until one finishes
    while true; do
      if ! kill -0 "${slot_a_pid}" 2>/dev/null; then
        wait_slot "a"
        (run_job "${key}" "${py}" "${base}" "${trained}" "${SLOT_A_GPUS}") &
        slot_a_pid=$!
        slot_a_key="${key}"
        break
      elif ! kill -0 "${slot_b_pid}" 2>/dev/null; then
        wait_slot "b"
        (run_job "${key}" "${py}" "${base}" "${trained}" "${SLOT_B_GPUS}") &
        slot_b_pid=$!
        slot_b_key="${key}"
        break
      fi
      sleep 10
    done
  fi
done

# Drain remaining slots
[[ -n "${slot_a_pid}" ]] && wait_slot "a"
[[ -n "${slot_b_pid}" ]] && wait_slot "b"

echo ""
echo "=========================================="
echo "All ASR evaluations finished."
echo "Results saved under: ${RESULT_DIR}/"
echo "=========================================="

if [[ ${#failed_jobs[@]} -gt 0 ]]; then
  echo ""
  echo "WARNING: failed jobs: ${failed_jobs[*]}" >&2
  exit 1
fi
