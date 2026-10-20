# Capability evaluation

The five paper tasks use EleutherAI's
[lm-evaluation-harness at `d800e04dcb1ce96791d8b2926cf0cc7703d58457`](https://github.com/EleutherAI/lm-evaluation-harness/tree/d800e04dcb1ce96791d8b2926cf0cc7703d58457),
with the five archived overlays in `patches/`. They add robust GPQA answer
extraction and strip reasoning text before HumanEval code extraction.
The overlays are unchanged from the author's archived experiments; see
[source provenance](../../../configs/source_provenance.json).

For an existing installation under the old directory name, follow the
[path migration guide](../../../docs/PATH_MIGRATION.md) and recreate the virtualenv.

## Install

Run from the repository root on a CUDA machine with Python 3.12 and `uv`:

```bash
bash scripts/evaluate/capability/setup.sh
```

This creates `scripts/evaluate/capability/.venv`, checks out the exact harness, applies the
overlays, installs constrained core package versions, runs `pip check`, and saves
the resolved package list in `.venv/resolved-requirements.txt`.
It refuses to replace unrelated local harness edits or switch an existing
checkout to another revision automatically.

This environment is separate from attack/DPO training: the math task uses
math-verify and an ANTLR runtime incompatible with the vendored training package.
SFT's OLMo-core 2.5.0 also has its own environment.

## Task mapping

| Paper task | Harness task | Shots | Headline metric |
|---|---|---:|---|
| GPQA | `gpqa_main_cot_zeroshot_robust` | 0 | exact_match,robust-extract |
| MMLU-Pro | `leaderboard_mmlu_pro` | 5 | acc,none (choice likelihood) |
| ARC-C | `arc_challenge_chat` | 0 | exact_match,remove_whitespace |
| MATH | `minerva_math500` | 0 | math_verify,none |
| HumanEval | `humaneval_think_strip` | 0 | pass@1,create_test |

Seeds are explicit: random 0, NumPy 1234, Torch 1234, few-shot 1234;
the vLLM engine seed is also 1234. Chat templates are applied for these
instruction-tuned checkpoints.

## Run

Always select the model and matching mode:

```bash
# Inspect the exact command without importing GPU libraries or downloading models.
bash scripts/evaluate/capability/run_eval.sh --model Qwen/Qwen2.5-3B-Instruct \
  --mode instruct --gpus 0 --tasks arc --dry-run

# Small execution check. Keep this output separate from full results.
bash scripts/evaluate/capability/run_eval.sh --model Qwen/Qwen2.5-3B-Instruct \
  --mode instruct --gpus 0 --tasks arc --limit 2 --output data/result/cap-smoke

# Full five-task evaluation of an OLMo checkpoint.
bash scripts/evaluate/capability/run_eval.sh --model /path/to/dpo-checkpoint \
  --mode olmo --gpus 0 --output data/result/cap-full

# Pin a remote model for a recorded run.
bash scripts/evaluate/capability/run_eval.sh --model Qwen/Qwen3-8B --revision HF_COMMIT_SHA \
  --mode nothink --gpus 0,1 --output data/result/qwen3-full
```

`--gpus 0,1` assigns both GPUs to one tensor-parallel evaluation. Tasks run
sequentially for that model; launch separate commands with separate GPU sets
and output directories to evaluate models in parallel.
HF dataset/model access must be available, including GPQA access.
HumanEval executes generated code; the runner enables the harness's required
code-evaluation flags, so run it in an isolated evaluation environment.

| Mode | Thinking flag | Context | Archived generation arguments |
|---|---|---:|---|
| `instruct` | omitted | 12,288 | greedy |
| `think` | enabled | 16,384 | temperature 0.6, top-p 0.95, top-k 20 |
| `nothink` | disabled | 12,288 | temperature 0.7, top-p 0.8, top-k 20 |
| `olmo` | omitted | 16,384 | temperature 0.6, top-p 0.95 |

The model-level generation limit is 8,192, but tasks can override it: ARC uses
100 tokens and its own stop strings. MMLU-Pro scores option likelihoods rather
than generating answers.

`--sampling archived` is the default. It preserves the archived temperature/top-p
arguments without forcing `do_sample=True`. Saved historical OLMo results have
`do_sample=False` for GPQA, MATH and HumanEval; the pinned backend normalizes these
to greedy decoding. ARC has no such flag and uses the supplied temperature.
Use `--sampling sample` or `--sampling greedy` only as an explicit new experiment;
the policy is recorded in each manifest. See the [reverification notes](../../../docs/REVERIFICATION.md).
Use `olmo` for the archived OLMo evaluations and `think`/`nothink` for Qwen3's
explicit toggle. A model without that toggle should not be labeled as toggled.

## Output and resume

Each `<output>/<model-key>/<task>/` contains the harness results, sample logs,
`run_manifest.json`, and a `COMPLETE` marker after successful execution.
The manifest records the model/revision, mode, seeds, limit, framework versions,
harness commit, and scoring-overlay hashes.

Rerunning the same command skips a completed task. A changed configuration or
an output without a manifest is rejected; select a new output directory.
Use `--revision` for remote HF models: an unspecified revision is recorded as
unspecified and does not establish a fixed model snapshot.

`summary.csv` and `summary.json` select the named aggregate task and its metric.
Only complete, full runs with verified result/manifest checksums contribute
scores. Smoke, incomplete and invalid runs have a status and a null score.
Multiple result files in one task directory are rejected as ambiguous.
The summary never averages a parent score together with its child scores or
substitutes a standard error. Old plain-text completion markers must be migrated
after review or evaluated in a new directory; they are not trusted automatically.

## Validation

Offline tests and the pinned upstream argument parser have been exercised.
The environment constraints have passed dependency resolution on Linux/Python 3.12.
GPU installation, model generation, and paper scores have not been reproduced
on the audit host. See [current validation status](../../../docs/REPRODUCING.md#validation-and-remaining-gaps).

The upstream harness and derivative overlays are covered by its MIT license,
retained in `LICENSE.lm-evaluation-harness`.
