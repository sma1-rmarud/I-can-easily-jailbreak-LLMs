# “I” can easily jailbreak LLMs

**ACHILLES attacks and THETIS defenses**

Code for the paper (EMNLP 2026 Findings).

[THETIS datasets](https://huggingface.co/collections/sma1-rmarud/i-can-easily-jailbreak-llms-thetis) ·
[Reproduction guide](docs/REPRODUCING.md) · [Capability evaluation](scripts/evaluate/capability/README.md)

ACHILLES uses GRPO with a pattern-based verifier to attack recurring refusal
openings. THETIS replaces templated refusals with relevant, non-actionable
answers to improve robustness. The paper reports the attack and defense results;
this repository provides their training and evaluation code for safety research.

**Status:** short GPU training, checkpoint conversion/reload, capability and
safety evaluation checks have passed. Full training and paper-score reproduction
remain unfinished. See [validation and remaining gaps](docs/REPRODUCING.md#validation-and-remaining-gaps).

## Setup

Run commands from the repository root with Python 3.12 and a CUDA environment:

```bash
git clone https://github.com/sma1-rmarud/I-can-easily-jailbreak-LLMs.git
cd I-can-easily-jailbreak-LLMs
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-build-isolation flash-attn==2.8.3
pip check
python scripts/check_reproducibility.py --runtime
```

`requirements.txt` and Docker share `requirements.runpod.lock.txt`.
For `uv pip`, add `--no-sources`. Alternatively, build with
`docker build -t achilles .`. SFT and capability evaluation use separate
[environments](docs/REPRODUCING.md#environment-and-preflight).

Export credentials as needed; keep them outside Git:

| Variable | Use |
|---|---|
| `HF_TOKEN` | Gated models/datasets, including Llama and Llama Guard |
| `WANDB_API_KEY` | Training tracking; use `WANDB_MODE=offline` for local tracking |
| `OPENAI_API_KEY` | LV-RLVR's GPT-4o-mini judge |
| `OPENROUTER_API_KEY` | Default Gemini ASR provider |
| `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `Google_API_KEY` | Direct Gemini API with `GEMINI_PROVIDER=google` |

## ACHILLES training

```bash
bash scripts/train/achilles_qwen25_3b_instruct.sh
```

Model-specific launchers are in `scripts/train/`. Full recipes assume eight GPUs
and save under `data/output/`. See [verifiers and model settings](docs/REPRODUCING.md#achilles-attack-training)
and [one-update checks](docs/REPRODUCING.md#short-execution-checks).

## THETIS defense

The [THETIS collection](https://huggingface.co/collections/sma1-rmarud/i-can-easily-jailbreak-llms-thetis)
contains [SFT](https://huggingface.co/datasets/sma1-rmarud/thetis_sft) (65,494 examples)
and [DPO](https://huggingface.co/datasets/sma1-rmarud/thetis_dpo) (259,922 examples).
Final paper DPO training used **32,490 examples**. Dataset revisions are pinned in
[configs/datasets.json](configs/datasets.json).

Training starts from `allenai/Olmo-3-7B-Instruct-SFT`, applies THETIS SFT, then DPO.
After [setting up the environments](docs/REPRODUCING.md#environment-and-preflight):

```bash
python scripts/data/download_thetis.py
python scripts/defense/run_table5.py --stage sft
python scripts/defense/run_table5.py --stage dpo
python scripts/defense/run_table5.py --stage achilles
# Optional LV-RLVR comparison; uses a paid judge.
python scripts/defense/run_table5.py --stage lvrlvr
```

The runner connects local checkpoints automatically. Each stage supports
`--dry-run`. See the [complete workflow and control arm](docs/REPRODUCING.md#local-training-to-evaluation-workflow).

## Evaluation

Table 2 uses Llama-Guard-3-8B; all control/THETIS rows in Table 5 use the same
Gemini judge. Choose the evaluator matching the model family and pass the
checkpoint explicitly. The batch ASR registry does not cover every paper model.

```bash
# Instruction-model safety evaluation with Llama Guard.
bash scripts/evaluate/safety/evaluate_asr_instruct.sh /path/to/checkpoint 0 experiment

# Table 5 safety evaluation; generation plus paid Gemini judging.
LIMIT=2 RESULT_DIR=data/result/safety-smoke \
  bash scripts/evaluate/safety/evaluate_asr_gemini.sh /path/to/checkpoint 0

# Small capability check; omit --limit and select a new output for a full run.
bash scripts/evaluate/capability/setup.sh
bash scripts/evaluate/capability/run_eval.sh --model /path/to/checkpoint \
  --mode instruct --gpus 0 --tasks arc --limit 2 --output data/result/cap-smoke
```

[Safety settings](docs/REPRODUCING.md#evaluate-safety) ·
[Capability tasks, model modes and summaries](scripts/evaluate/capability/README.md)

## Repository layout

| Path | Contents |
|---|---|
| `scripts/train/` | ACHILLES and LV-RLVR model recipes |
| `scripts/defense/` | Local SFT, DPO and Table 5 workflow |
| `scripts/evaluate/` | Safety and capability evaluation |
| `scripts/data/`, `scripts/analyze/` | Dataset download and Gemini judging |
| `configs/` | Dataset revisions, source provenance and judge prompt |
| `data/` | Attack prompts and safety benchmarks ([data notes](data/README.md)) |
| `open-instruct/` | Vendored Ai2 training code and attack verifiers |
| `docs/` | Reproduction guide and machine-readable validation evidence |

Original data overlaps are documented in the data notes. Run
`python scripts/check_reproducibility.py` and `python -m pytest -q tests`
for offline checks.

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
