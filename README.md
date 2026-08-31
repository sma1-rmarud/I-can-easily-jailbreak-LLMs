# ACHILLES

<!-- TODO: paper title, authors, arXiv/anthology link -->

RLVR (Reinforcement Learning with Verifiable Rewards) experiments that use the **first
token of a refusal** as the entire reward signal.

Aligned LLMs almost always open a refusal with `I` — *"I cannot…"*, *"I'm sorry…"*. On
StrongREJECT we measure **97.1%** of `Qwen2.5-3B-Instruct` responses starting with `I`.
This repository trains against a single verifiable rule — *reward 1.0 iff the visible
answer does not start with `I`* — and measures the consequences:

- **Safety** — attack success rate (ASR), judged by Llama-Guard-3-8B
- **Capability** — `eval_cap_noi` (gpqa / mmlu_pro / math / arc / humaneval)

The attack needs only *queries* and a pattern-based verifier: no reference answers, no
reward model, no expensive preference data.

## Quick start

```bash
git clone https://github.com/sma1-rmarud/ACHILLES.git && cd ACHILLES
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-build-isolation flash-attn==2.8.3   # must build against your torch/CUDA

export HF_TOKEN=hf_...                               # gated models (Llama, Llama-Guard)
bash scripts/train/3.run_notI_qwen25_3b_inst.sh      # train
bash scripts/evaluate/asr/evaluate_asr_instruct.sh   # evaluate the latest checkpoint
```

Every script `cd`s to the repository root first, so all paths inside are relative and it
does not matter where you invoke them from.

## Layout

| Path | Contents |
|---|---|
| `open-instruct/` | Vendored [Ai2 open-instruct](https://github.com/allenai/open-instruct) (Apache-2.0) with our verifiers in `open_instruct/first_token_verifier.py` |
| `scripts/train/` | Per-model GRPO launchers |
| `scripts/evaluate/asr/` | Safety evaluation: vLLM generation + Llama-Guard-3 judging |
| `scripts/inference/` | DPO rejected-response regeneration and Hub upload |
| `scripts/analyze/` | LLM-judge safety scoring, dataset sanity checks |
| `eval_cap_noi/` | Capability evaluation on a patched lm-evaluation-harness ([README](eval_cap_noi/README.md)) |
| `data/` | Training prompts and safety benchmarks ([README](data/README.md)) |

## Method

`open_instruct/first_token_verifier.py` implements the reward. Names describe the
**mechanism**, not the model — only the STAR1 variant is model-specific. Think-mode
verifiers score only the visible answer after `</think>`.

| Verifier | Mode | Reward |
|---|---|---|
| `first_token_not_i` | think | Binary; 1.0 unless the visible answer starts with `I` |
| `first_token_not_i_no_think` | no-think | Binary, whole response |
| `first_token_not_i_think_shaped` | think | Shaped 0 / 0.25 / 0.5 / 1.0 |
| `first_token_not_i_no_think_shaped` | no-think | Shaped 0 / 0.25 / 0.5 / 1.0 |
| `first_token_not_i_star1` | think | STAR1-R1-Distill only: policy-vocabulary gate on the think block + `I` check |
| `general-refusal`, `general-refusal-think` | both | LLM-judge on refusal in general rather than the first token (`open_instruct/ground_truth_utils.py`) |

**Shaped** variants resist prefix gaming: they strip soft openers (`"Certainly!"`,
`"Sure,"`), then award 0.25 if `I` appears within the first five words, 0.5 if a refusal
template appears in the first sentence, and 1.0 only if neither does.

## Training

Scripts are named `<n>.run_notI_<model>.sh`. Each writes its final model to
`data/output/<exp_name>/<exp_name>__<seed>__<unixtime>/`; `--save_freq 0` means only the
final checkpoint is kept.

| Script | Base model | Verifier |
|---|---|---|
| `1.` | allenai/Olmo-3-7B-Instruct | `..._no_think_shaped` |
| `2.` | allenai/OLMo-3-7B-Think | `..._think_shaped` |
| `3.` | Qwen/Qwen2.5-3B-Instruct | `..._no_think` |
| `4.` | Qwen/Qwen2.5-7B-Instruct | `..._no_think` |
| `5.` | meta-llama/Llama-3.1-8B-Instruct | `..._no_think` |
| `5_2.` | meta-llama/Llama-3.2-3B-Instruct | `..._no_think` |
| `6.` / `6_2.` / `6_3.` | UCSC-VLAA/STAR1-R1-Distill-7B / 14B / 32B | `..._star1` |
| `7.` / `7_2.` | Qwen/Qwen3-8B (think / no-think) | `first_token_not_i` / `..._no_think_shaped` |
| `8.` / `8_2.` | OLMo-3 DPO checkpoints (original / ours) | `..._no_think_shaped` |
| `9.` / `9_2.` | OLMo-3 DPO checkpoints (original / ours) | `general-refusal` |

Environment overrides: `RLVR_ROOT` (repository root), `RLVR_VENV` (virtualenv; falls back
to `.venv`, `venv`, `/opt/venv`), `HF_TOKEN` (required). HuggingFace models are cached in
the default location; set `HF_HOME` to move them.

## Evaluation

**Safety (ASR).** Generates responses with vLLM, judges them with Llama-Guard-3-8B, and
reports ASR alongside `I_rate` — the fraction of responses still opening with `I`.

```bash
bash scripts/evaluate/asr/evaluate_asr_instruct.sh [trained_model] [gpus] [exp_name]
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct bash scripts/evaluate/asr/evaluate_asr_instruct.sh
bash scripts/evaluate/asr/evaluate_asr_all_parallel.sh     # model registry, 2 slots x 4 GPUs
```

With no arguments the script auto-detects the most recent run under `data/output/`.
Results are written to `data/result/<exp_name>/`. `evaluate_asr_instruct.py` covers models
without a think toggle; `evaluate_asr_{olmo3_think,qwen3_reasoning,qwen3_nothink,star1_r1_reasoning}.py`
handle the reasoning variants.

**Capability.**

```bash
cd eval_cap_noi && bash setup.sh && bash run_eval.sh
```

## Reproducibility

Prompts and safety benchmarks are committed, so evaluation needs no external download
beyond the models themselves. Known limitations:

- `8.`, `8_2.`, `9.`, `9_2.` depend on intermediate OLMo-3 DPO checkpoints that are not
  publicly available; the other training scripts use public base models.
- `6_3.` (STAR1-32B) needs 80GB-class GPUs: the 61GB checkpoint does not fit under
  `--vllm_tensor_parallel_size 1` on 48GB cards.
- Training scripts assume 8 GPUs (7 learners + 1 vLLM engine). Adjust
  `--num_learners_per_node`, `--vllm_num_engines` and `CUDA_VISIBLE_DEVICES` otherwise.
- `requirements.txt` is a frozen environment (Python 3.12.13, torch 2.9.1+cu128,
  vLLM 0.16.0, transformers 4.57.6).

## Acknowledgements

Training builds on [Ai2 open-instruct](https://github.com/allenai/open-instruct)
(Apache-2.0), vendored under `open-instruct/` with its licence intact. Capability
evaluation patches [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
at commit `d800e04`. Benchmarks: [StrongREJECT](https://arxiv.org/abs/2402.10260),
[JailbreakBench](https://arxiv.org/abs/2404.01318). Training queries are sampled from
[LLM-LAT/harmful-dataset](https://huggingface.co/datasets/LLM-LAT/harmful-dataset) and
[databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k).

## Citation

```bibtex
@inproceedings{achilles,
  title     = {},
  author    = {},
  booktitle = {Proceedings of EMNLP},
  year      = {2026}
}
```
