# Capability evaluation

Uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/d800e04dcb1ce96791d8b2926cf0cc7703d58457)
at `d800e04dcb1ce96791d8b2926cf0cc7703d58457` with the five overlays in `patches/`.

## Run

From the repository root, with Python 3.12, CUDA and `uv`:

```bash
bash scripts/evaluate/capability/setup.sh
bash scripts/evaluate/capability/run_eval.sh --model /path/to/checkpoint \
  --mode olmo --gpus 0 --output data/result/capability
```

Setup creates a separate environment at `scripts/evaluate/capability/.venv`.
Use `--tasks arc --limit 2` and a separate output directory for a small check;
`--dry-run` prints the command. `--revision HF_COMMIT_SHA` pins a remote model.
`--gpus 0,1` runs one model with tensor parallelism across both GPUs.
GPQA requires dataset access. HumanEval executes generated code; run it in an
isolated evaluation environment.

## Tasks and modes

| Task | Harness name | Shots | Metric |
|---|---|---:|---|
| GPQA | `gpqa_main_cot_zeroshot_robust` | 0 | exact_match,robust-extract |
| MMLU-Pro | `leaderboard_mmlu_pro` | 5 | acc,none |
| ARC-C | `arc_challenge_chat` | 0 | exact_match,remove_whitespace |
| MATH | `minerva_math500` | 0 | math_verify,none |
| HumanEval | `humaneval_think_strip` | 0 | pass@1,create_test |

| Mode | Thinking flag | Context | Generation arguments |
|---|---|---:|---|
| `instruct` | omitted | 12,288 | greedy |
| `think` | enabled | 16,384 | temperature 0.6, top-p 0.95, top-k 20 |
| `nothink` | disabled | 12,288 | temperature 0.7, top-p 0.8, top-k 20 |
| `olmo` | omitted | 16,384 | temperature 0.6, top-p 0.95 |

Use `olmo` for OLMo and `think`/`nothink` for Qwen3's thinking toggle.
Seeds: random 0; NumPy, Torch, few-shot and vLLM 1234.
Chat templates are applied. The model-level generation cap is 8192 tokens;
ARC overrides it to 100, and MMLU-Pro scores option likelihoods.

Default `--sampling archived` preserves task-level sampling settings: GPQA,
MATH and HumanEval use greedy decoding; ARC uses the mode's temperature.
Use `--sampling sample` or `--sampling greedy` for an explicit override.
Keep model mode and sampling settings consistent across comparisons.

## Results

Each `<output>/<model-key>/<task>/` contains results, sample logs,
`run_manifest.json` and a `COMPLETE` marker. Manifests record revisions,
settings and scoring-overlay hashes. Identical reruns skip completed tasks;
changed settings require a new output directory.

`summary.csv` and `summary.json` include scores only for complete, full runs
with verified checksums. Small checks and incomplete runs have null scores.

## License

The harness and derivative overlays retain the MIT license in
[LICENSE.lm-evaluation-harness](LICENSE.lm-evaluation-harness).
