#!/bin/bash
# model_name_or_path : Qwen/Qwen2.5-3B-Instruct

set -e

# Experiment name
exp_name="Qwen2.5-3B-Instruct_first_token_not_i_$(date +%Y%m%d_%H%M%S)"

# Run from the repo root so every path below is relative.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="open-instruct:${PYTHONPATH:-}"

# Virtualenv to activate. Override with RLVR_VENV; falls back to a repo-local
# .venv/ or venv/, then to /opt/venv (what the Dockerfile builds).
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

# Use GPUs 0-7
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# CUDA library paths for Ray workers (pip-installed nvidia packages)
NVIDIA_PKG_DIR="$(python -c 'import sysconfig,os;print(os.path.join(sysconfig.get_paths()["purelib"],"nvidia"))')"
CUDA_LIB_DIRS=$(find "$NVIDIA_PKG_DIR" -mindepth 2 -maxdepth 2 -name lib -type d 2>/dev/null | paste -sd:)
export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=/usr/local/cuda
# Triton/Inductor JIT-compiles kernels with `gcc -lcuda`. Many hosts ship only
# libcuda.so.1 with no unversioned dev symlink, which makes that link fail. Add
# the CUDA stub dir to the linker search path; the real driver is still what
# gets loaded at runtime.
export LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH:-}"

# HuggingFace token
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"

# vLLM serialization workaround
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Disable custom all-reduce (RTX 6000 Ada has no NVLink)
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

# Reduce CUDA fragmentation risk
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Suppress noisy asyncio socket warnings from Ray
export PYTHONWARNINGS="ignore::DeprecationWarning"
export RAY_IGNORE_UNHANDLED_ERRORS=1

# Run GRPO training (Qwen2.5-3B-Instruct)
# open-instruct sets push_to_hub=True by default and would push to
# <hf_entity>/open_instruct_dev, which fails with 403. Keep it off; to publish,
# pass --push_to_hub true --hf_repo_id <your/repo>.
python -W ignore::UserWarning open-instruct/open_instruct/grpo_fast.py \
    --exp_name "$exp_name" \
    --output_dir "${ATTACK_OUTPUT_DIR:-data/output/${exp_name}}" \
    \
    --dataset_mixer_list data/train/train.jsonl 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list data/test/test100.jsonl 1.0 \
    --dataset_mixer_eval_list_splits train \
    \
    --model_name_or_path "${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-3B-Instruct}" \
    \
    --max_prompt_token_length 256 \
    --response_length 1536 \
    --pack_length 1920 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 8 \
    --num_mini_batches 1 \
    --num_epochs 1 \
    \
    --temperature 0.7 \
    --non_stop_penalty False \
    --non_stop_penalty_value 0.0 \
    \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    \
    --learning_rate 3e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --total_episodes 15000 \
    \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 1 \
    \
    --num_learners_per_node 7 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 1 \
    --vllm_gpu_memory_utilization 0.8 \
    \
    --beta 0.03 \
    --kl_estimator 2 \
    --apply_verifiable_reward true \
    --verification_reward 2.0 \
    --remap_verifier first_token_not_i=first_token_not_i_no_think \
    \
    --seed 42 \
    --local_eval_every 10 \
    --save_freq 0 \
    \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub false \
    "$@"

echo "Training completed! Model saved under ${ATTACK_OUTPUT_DIR:-data/output/${exp_name}}/<run_name>"
