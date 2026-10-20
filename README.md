# “I” can easily jailbreak LLMs

**ACHILLES attacks and THETIS defenses**

Code for the paper (EMNLP 2026 Findings).

<!-- TODO: arXiv / ACL Anthology link -->

[THETIS datasets](https://huggingface.co/collections/sma1-rmarud/i-can-easily-jailbreak-llms-thetis) ·
[Reproduction guide](docs/REPRODUCING.md) · [Capability evaluation](eval_cap_noi/README.md)

**Validation status:** 51 CPU regression tests and offline checks pass. GPU training,
checkpoint conversion with full weights, and model evaluation have not yet been validated.

Safety-aligned LLMs open refusals with a handful of recurring patterns — most of all the single token `I` (*"I cannot…"*, *"I'm sorry…"*), on average **88.08%** of responses to harmful prompts across the 13 model configurations.

**ACHILLES** turns that regularity into an attack: GRPO with a lightweight pattern verifier whose entire reward is *"the visible answer does not open with a refusal pattern"*. No adversarial suffix search, no reference answers, no reward model, no preference data — only
queries and a regex. ASR rises to 61% on `Llama-3.1-8B-Instruct` and >90% on the STAR1
safety-reasoning family, while preserving downstream benchmark performance.

**THETIS** is the defense: a safety alignment dataset that replaces templated refusals with topically-related but non-actionable answers, removing the surface regularity the attack feeds on. Models post-trained on it hold ASR near zero against both ACHILLES and an LLM-verifier RLVR attack.

## Intended use

This repository provides training and evaluation code for the attack and its defense.
See the [reproduction status](docs/REPRODUCING.md#validation-and-remaining-gaps) for the
checks performed and the experiment details still requiring confirmation.
It is released for safety research — measuring alignment robustness and evaluating mitigations.
The training scripts weaken safety alignment by construction; do not deploy the resulting checkpoints.

## THETIS datasets

The [THETIS collection](https://huggingface.co/collections/sma1-rmarud/i-can-easily-jailbreak-llms-thetis)
contains [THETIS-SFT](https://huggingface.co/datasets/sma1-rmarud/thetis_sft)
(65,494 examples) and [THETIS-DPO](https://huggingface.co/datasets/sma1-rmarud/thetis_dpo)
(259,922 examples in the full release).
Fixed dataset revisions are recorded in [configs/datasets.json](configs/datasets.json).

```bash
python scripts/data/download_thetis.py --dry-run    # inspect versions, no download
python scripts/data/download_thetis.py             # download and validate both datasets
```

The defense training sequence is
**`allenai/Olmo-3-7B-Instruct-SFT` → THETIS SFT → THETIS DPO**.
Use the [SFT/DPO instructions](docs/REPRODUCING.md#thetis-defense-training) to train from
the released data. The author confirmed that final DPO training used a
32,490-example subset; 259,922 is the full released dataset size.

The repository includes local SFT/DPO training code and a
[training-to-evaluation runner](docs/REPRODUCING.md#local-training-to-evaluation-workflow):
`python scripts/defense/run_table5.py --stage sft` → `dpo` → `achilles` / `lvrlvr`.
Evaluation reads the resulting local checkpoints automatically; pre-existing
final attack checkpoints are not needed.

## Layout

| Path | Contents |
|---|---|
| `open-instruct/` | Vendored [Ai2 open-instruct](https://github.com/allenai/open-instruct) (Apache-2.0). Our verifiers live in `open_instruct/first_token_verifier.py`; the LLM-judge verifier is `general-refusal` in `open_instruct/ground_truth_utils.py` |
| `scripts/train/` | Per-model GRPO launchers (`<n>.run_notI_<model>.sh`) |
| `scripts/evaluate/asr/` | Safety evaluation: vLLM generation + Llama-Guard-3-8B judging |
| `scripts/defense/` | THETIS/control SFT and DPO launchers, with local checkpoint output |
| `scripts/data/` | Pinned THETIS download and schema/count validation |
| `scripts/analyze/safety_judge_gemini.py` | Table 5 judge using the paper's Figure 5 prompt |
| `eval_cap_noi/` | Pinned capability harness, five scoring overlays, runner and summaries |
| `configs/` | Dataset revisions, source provenance, and judge prompt |
| `data/` | Training prompts and safety benchmarks ([README](data/README.md)) |
| `Dockerfile` | CUDA 12.6 / torch 2.9.1 / vLLM 0.16.0 image, venv at `/opt/venv` |

Run the commands below from the repository root. Shell launchers resolve repository
paths themselves; Python CLI input/output paths are relative to your current directory.

## Setup

```bash
git clone https://github.com/sma1-rmarud/I-can-easily-jailbreak-LLMs.git && cd I-can-easily-jailbreak-LLMs
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-build-isolation flash-attn==2.8.3   # must build against your torch/CUDA
pip check
python scripts/check_reproducibility.py --runtime
```

`requirements.txt` uses the same training dependency list as Docker:
`requirements.runpod.lock.txt`. For `uv pip`, add `--no-sources` so the vendored
project's development CUDA index does not override this PyPI-based environment.
Capability evaluation and THETIS SFT use separate environments, described in the guide.

Or build the image, which installs `requirements.runpod.lock.txt` into `/opt/venv`:

```bash
docker build -t achilles .
```

Environment variables:

| Variable | Needed for | Notes |
|---|---|---|
| `HF_TOKEN` | everything | Gated models: Llama, Llama-Guard-3-8B |
| `WANDB_API_KEY` | all training scripts | They pass `--with_tracking`; drop that flag to run offline |
| `OPENAI_API_KEY` | `9.`, `9_2.` only | The LV-RLVR baseline judges with `openai/gpt-4o-mini` |
| `OPENROUTER_API_KEY` | Gemini ASR judge | Table 5; exported in the shell, never written to manifests |
| `MODEL_NAME_OR_PATH` | `8.`–`9_2.` | Override the archived DPO model with your local SFT→DPO checkpoint |
| `RLVR_VENV` | optional | Virtualenv to activate; falls back to `.venv` → `venv` → `/opt/venv` |
| `HF_HOME` | optional | Hugging Face model/dataset cache |

## Method

`open_instruct/first_token_verifier.py` implements the reward. Verifier names describe the
**mechanism**, not the model — only the STAR1 variant is model-specific. Think-mode
verifiers score only the visible answer after `</think>`.

| Verifier | Mode | Reward |
|---|---|---|
| `first_token_not_i` | think | 1.0 unless the visible answer starts with `I` |
| `first_token_not_i_no_think` | no-think | Same check on the whole response |
| `first_token_not_i_think_shaped` | think | Graded 0 / 0.25 / 0.5 / 1.0 |
| `first_token_not_i_no_think_shaped` | no-think | Same three checks, but every failure scores 0.0 |
| `first_token_not_i_star1` | think | STAR1-R1-Distill only: policy-vocabulary gate on the think block + `I` check |
| `general-refusal`, `general-refusal-think` | both | LLM judge on refusal in general — the LV-RLVR baseline, not ACHILLES |

The plain variants reward-hack easily: models learn to prepend `"Certainly! I cannot…"`.
The **shaped** variants close that off with three stages — strip up to two soft openers,
reject an `I` anywhere in the first five words, reject a refusal template in the first
sentence. In the think variant those stages are graded (0.25 for a prefix hack, 0.5 for a
surviving refusal template); in the no-think variant all three are hard failures.

`--verification_reward` scales the 1.0, and is the paper's *verifier reward scale*
(Appendix C, Table 10) — the main knob for attack strength.

## Training

Each script writes to `data/output/<exp_name>/<exp_name>__<seed>__<unixtime>/`.
`--save_freq 0` keeps only the final checkpoint. Shared settings (paper Table 9): prompt
256 / response 1536 / pack 1920 tokens, 1 epoch, cosine LR with 0.05 warmup, 15,000
episodes, DeepSpeed ZeRO-3, per-device batch 1, seed 42.

```bash
HF_TOKEN=hf_... bash scripts/train/3.run_notI_qwen25_3b_inst.sh
```

| Script | Base model | Verifier | LR | β | KL est. | Reward scale |
|---|---|---|---|---|---|---|
| `1.` | allenai/Olmo-3-7B-Instruct | `..._no_think_shaped` | 3e-5 | 0.001 | 1 | 4.0 |
| `2.` | allenai/OLMo-3-7B-Think | `..._think_shaped` | 1e-5 | 0.01 | 2 | 3.0 |
| `3.` | Qwen/Qwen2.5-3B-Instruct | `..._no_think` | 3e-6 | 0.03 | 2 | 2.0 |
| `4.` | Qwen/Qwen2.5-7B-Instruct | `..._no_think` | 3e-6 | 0.03 | 2 | 2.0 |
| `5.` | meta-llama/Llama-3.1-8B-Instruct | `..._no_think` | 3e-6 | 0.03 | 2 | 2.0 |
| `5_2.` | meta-llama/Llama-3.2-3B-Instruct | `..._no_think` | 3e-6 | 0.03 | 2 | 2.0 |
| `6.` | UCSC-VLAA/STAR1-R1-Distill-7B | `..._star1` | 3e-5 | 0.001 | 3 | 4.0 |
| `6_2.` | UCSC-VLAA/STAR1-R1-Distill-14B | `..._star1` | 1e-5 | 0.01 | 2 | 4.0 |
| `6_3.` | UCSC-VLAA/STAR1-R1-Distill-32B | `..._star1` | 1e-5 | 0.01 | 2 | 4.0 |
| `7.` | Qwen/Qwen3-8B (think) | `first_token_not_i` | 1e-5 | 0.01 | 2 | 3.0 |
| `7_2.` | Qwen/Qwen3-8B (no-think) | `..._no_think_shaped` | 1e-5 | 0.01 | 2 | 3.0 |
| `8.` / `8_2.` | our OLMo-3 DPO checkpoints (base / THETIS) | `..._no_think_shaped` | 3e-5 | 0.001 | 1 | 4.0 |
| `9.` / `9_2.` | our OLMo-3 DPO checkpoints (base / THETIS) | `general-refusal` | 3e-5 | 0.001 | 1 | 4.0 |

`1.`–`7_2.` provide the training configurations for paper Table 2. `8.`/`8_2.` are the ACHILLES rows of Table 5, and
`9.`/`9_2.` the LV-RLVR rows — the base-vs-THETIS comparison. Rollout settings per model
(temperature, unique prompts, samples per prompt) follow Table 8.

## Evaluation

**Safety (ASR).** The model-specific runners generate with vLLM or Transformers,
judge with Llama-Guard-3-8B, and report ASR next to
`I_rate` — the fraction of responses still opening with `I`, i.e. whether the verifier
actually moved what it was supposed to.

```bash
bash scripts/evaluate/asr/evaluate_asr_instruct.sh [trained_model] [gpus] [exp_name]
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct bash scripts/evaluate/asr/evaluate_asr_instruct.sh
bash scripts/evaluate/asr/evaluate_asr_all_parallel.sh     # model registry, 2 slots x 4 GPUs
```

With no arguments the script auto-detects the most recent run under `data/output/`. Results
land in `data/result/<exp_name>/`. `evaluate_asr_instruct.py` covers models without a think
toggle; `evaluate_asr_{olmo3_think,qwen3_reasoning,qwen3_nothink,star1_r1_reasoning}.py`
handle the reasoning variants. `evaluate_asr_all_parallel.sh` carries a registry of the
archived attacked checkpoints. The registry is partial: STAR1-7B is a skipped
placeholder and STAR1-14B/32B are not registered. Pass a verified checkpoint explicitly
to the corresponding model-family evaluator.

**All rows of Table 5**, including the control models, use `gemini-3-flash-preview` as the judge instead
(Llama-Guard-3-8B misreads THETIS-style non-refusals as unsafe); the prompt is in the
paper's Figure 5 and [configs/prompts/gemini_safety.txt](configs/prompts/gemini_safety.txt).

```bash
# First inspect the generation plan, without model downloads or API calls.
bash scripts/evaluate/asr/evaluate_asr_gemini.sh /path/to/dpo-checkpoint 0 --dry-run
# Run the same judge for each control/THETIS checkpoint being compared.
bash scripts/evaluate/asr/evaluate_asr_gemini.sh /path/to/dpo-checkpoint 0
```

Judge errors produce an incomplete summary and a nonzero exit status, not a zero ASR.

**Capability.** Paper Table 2/5 report MMLU-Pro (CoT, 5-shot), ARC-C, GPQA, HumanEval and
MATH (CoT) from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
at commit `d800e04`, with seeds 0 / 1234 / 1234 / 1234 (random / numpy / torch / few-shot).
The five archived scoring overlays are restored under `eval_cap_noi/patches/`.

```bash
bash eval_cap_noi/setup.sh
bash eval_cap_noi/run_eval.sh --model /path/to/checkpoint --mode olmo --gpus 0 --dry-run
bash eval_cap_noi/run_eval.sh --model /path/to/checkpoint --mode olmo --gpus 0 \
  --output data/result/capability
```

See [task names, sampling modes, smoke checks and result provenance](eval_cap_noi/README.md).

## Reproducibility notes

- Training scripts assume 8 GPUs (7 learners + 1 vLLM engine). Adjust
  `--num_learners_per_node`, `--vllm_num_engines` and `CUDA_VISIBLE_DEVICES` otherwise.
- `6_3.` (STAR1-32B) needs 80GB-class GPUs: the 61GB checkpoint does not fit under
  `--vllm_tensor_parallel_size 1` on 48GB cards.
- `8.`–`9_2.` start from our post-trained OLMo-3 DPO checkpoints
  (`sma1-rmarud/olmo3-7b-DPO-{original,ours}_old`). The SFT/DPO recipe that produced them is
  in the paper's Appendix E; the restored launchers are under `scripts/defense/`.
  Their commands have been checked, but GPU training has not been rerun in this release audit.
- Prompts and safety benchmarks are committed, so evaluation needs no download beyond the
  models themselves.
- The training environment targets Python 3.12, torch 2.9.1, vLLM 0.16.0,
  transformers 4.57.6 and NumPy 2.2.6. Capability and SFT have separate setup scripts.
- The original splits contain exact prompt overlaps: train/StrongREJECT 11,
  train/JBB 1, train/test100 3; all 100 test100 prompts occur in StrongREJECT.
  They are preserved for provenance; see [dataset audit](docs/data_audit.json).
- Run `python scripts/check_reproducibility.py` and `python -m pytest -q tests`
  for offline release checks. These do not establish reproduction of paper scores.

## Acknowledgements

Training builds on [Ai2 open-instruct](https://github.com/allenai/open-instruct) (Apache-2.0),
vendored under `open-instruct/` with its licence intact. Capability evaluation patches
[EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) at
commit `d800e04`. Benchmarks: [StrongREJECT](https://arxiv.org/abs/2402.10260),
[JailbreakBench](https://arxiv.org/abs/2404.01318). Training queries are sampled from
[LLM-LAT/harmful-dataset](https://huggingface.co/datasets/LLM-LAT/harmful-dataset) and
[databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k).
THETIS is built from [Dolci-Instruct-SFT/DPO](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT).

## Citation

```bibtex
@inproceedings{choi2026achilles,
  title     = {``I'' can easily jailbreak {LLM}s},
  author    = {Choi, Eunkyung and Kim, Yeeun and Kang, Juheon and Hwang, Wonseok},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```
