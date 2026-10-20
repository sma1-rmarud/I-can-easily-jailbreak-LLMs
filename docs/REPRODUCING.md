# Reproducing ACHILLES and THETIS

This guide separates runnable experiment paths from verified paper results.
Source restoration and CPU checks do not establish numerical reproduction.
Run commands from the repository root.
See the [latest reverification](REVERIFICATION.md) for historical capability
evidence, confirmed DPO count and corrections made after the initial restoration.

## Available experiment paths

| Paper component | Entry point | Status |
|---|---|---|
| Table 2 attack training | Model-family `scripts/train/achilles_*.sh` | Existing model-specific settings |
| Table 2 safety evaluation | `scripts/evaluate/safety/evaluate_asr_*.sh` | Llama-Guard-3-8B; partial batch model registry |
| Tables 2/5 capability | `scripts/evaluate/capability/run_eval.sh` | Restored five overlays and task mapping |
| Appendix E defense SFT | `scripts/defense/run_sft.py` | Restored OLMo-core recipe, local HF output |
| Appendix E defense DPO | `scripts/defense/run_dpo.py` | Explicit paper loss/settings, explicit data count |
| Table 5 attacks on control/THETIS | `scripts/train/{achilles,lvrlvr}_olmo3_7b_{control,thetis}.sh` | Supports a local DPO checkpoint override |
| Table 5 safety evaluation | `scripts/evaluate/safety/evaluate_asr_gemini.sh` | Restored Figure 5 prompt; both groups use Gemini |
| THETIS raw response generation | — | Released data can be downloaded; generation pipeline not restored |
| Table 3 SFT attack baselines; representation figures | — | Final experiment scripts/results still need identification |

The [THETIS collection](https://huggingface.co/collections/sma1-rmarud/i-can-easily-jailbreak-llms-thetis)
contains the released defense datasets, not model checkpoints.
Source commits and the author-confirmed SFT→DPO sequence are recorded in
[source_provenance.json](../configs/source_provenance.json).

## Environment and preflight

The attack/DPO scripts expect eight GPUs. Appendix E uses 8×H100 80GB for
defense training. Smaller evaluation models can use fewer GPUs; pass the intended
GPU set explicitly.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-build-isolation flash-attn==2.8.3
pip check

python scripts/check_reproducibility.py           # offline files/data/syntax checks
python scripts/check_reproducibility.py --runtime # active versions and GPU driver
python -m pip install pytest==9.0.3
python -m pytest -q tests
```

The root requirement file delegates to `requirements.runpod.lock.txt`, also
used by Docker. If installing with `uv pip`, use `--no-sources`: otherwise the
vendored package's development CUDA index can select CUDA 12.9 wheels against
the CUDA 12.8 dependency pins.

The old combined environment freeze was inconsistent (NumPy, ANTLR and BFCL
filelock conflicts). The training list uses NumPy 2.2.6 and ANTLR 4.9.3.
Capability evaluation installs math-verify separately. SFT uses OLMo-core 2.5.0
separately from attack/DPO's 2.3.0 dependency. Do not merge these environments.
Dependency resolution is verified; native CUDA/FlashAttention builds still
need validation on the execution machine.

Export `HF_TOKEN` for gated models/datasets. Training scripts use W&B;
set `WANDB_API_KEY`, or `WANDB_MODE=offline` to keep tracking local.
LV-RLVR uses `OPENAI_API_KEY`. Table 5 judging uses `OPENROUTER_API_KEY`.
No setup or reproduction launcher publishes models automatically.

## THETIS defense training

The author confirmed:

```text
allenai/Olmo-3-7B-Instruct-SFT
    → full SFT on THETIS-SFT
    → DPO on the resulting SFT checkpoint using THETIS-DPO
```

### Local training-to-evaluation workflow

SFT and DPO training code is included in this repository. Existing final
LV-RLVR checkpoints are not required: train the defense locally, train both
attacks from that defense, then evaluate the local outputs.

After setting up the three environments described below, use this stage runner:

```bash
python scripts/data/download_thetis.py
python scripts/defense/run_table5.py --stage sft --dry-run

python scripts/defense/run_table5.py --stage sft
python scripts/defense/run_table5.py --stage dpo
python scripts/defense/run_table5.py --stage achilles
python scripts/defense/run_table5.py --stage lvrlvr

for checkpoint in before achilles lvrlvr; do
  python scripts/defense/run_table5.py --stage safety --checkpoint "$checkpoint"
  python scripts/defense/run_table5.py --stage capability --checkpoint "$checkpoint"
done
```

The default arm is `thetis`. DPO selects 32,490 examples. Both attacks start
independently from the DPO output. LV-RLVR needs `OPENAI_API_KEY`; safety judging
needs `OPENROUTER_API_KEY`. The runner uses `.venv-sft` for SFT, `.venv` for
DPO/attacks/safety, and `scripts/evaluate/capability/.venv` for capability. Override these with
`--sft-venv`, `--training-venv`, and `--evaluation-venv` if needed.
Every stage supports `--dry-run` without loading a model or calling an API.

| Output | Local checkpoint |
|---|---|
| THETIS SFT | `data/output/thetis-sft/model/hf_model` |
| THETIS DPO / before attack | `data/output/thetis-dpo/model` |
| After ACHILLES | `data/output/thetis-achilles/model` |
| After LV-RLVR | `data/output/thetis-lvrlvr/model` |

GRPO saves into a timestamped subdirectory. After a successful attack, the
runner checks the completion marker and checkpoint files, then creates the
stable `model` symlink shown above. Failed or ambiguous runs do not get this
evaluation link. The actual output and starting model are recorded in
`run_manifest.json`. Evaluation follows these links automatically.
Keep failed runs for diagnosis; training outputs must be new/empty.

Use `--output-root` consistently across stages to change the model location.
Evaluation results default to `data/result/table5`; use `--results-root` for a
new evaluation run. For smoke checks, pass `--limit 2` to an evaluation stage
with a separate `--results-root data/result/table5-smoke`.

For the matched control, run the same stages with `--arm control`, supplying
`--sft-data /path/to/control-sft.jsonl` at SFT and
`--dpo-data /path/to/control-dpo.jsonl` at DPO. The later stages automatically
use `control-dpo/model`, `control-achilles/model`, or `control-lvrlvr/model`.
The control data availability caveat below still applies.
These commands define new experiments; historical score equivalence remains
an empirical check after execution.

### 1. Obtain pinned defense data

```bash
python scripts/data/download_thetis.py --dry-run
python scripts/data/download_thetis.py
```

The public snapshots in [datasets.json](../configs/datasets.json) are:

| Dataset | Revision | Rows |
|---|---|---:|
| `sma1-rmarud/thetis_sft` | `0dac6e92e24a818dabb6c564dde5f241bd3fc09f` | 65,494 |
| `sma1-rmarud/thetis_dpo` | `de887297ad6eb84043c484a4aa091f0cb14848d4` | 259,922 |

Files land at `data/thetis/thetis_sft/dolci_sft_ours.jsonl` and
`data/thetis/thetis_dpo/dolci_dpo_ours.jsonl`. Each gets a validated row count,
conversation schema check and SHA-256 manifest.

For the control arm, use the original matched SFT/DPO datasets and the same
training parameters. The 65,494-example control SFT dataset and its final
sampling provenance have not yet been added to this release.
The archived `uos-nlp/dolci_sft_original` endpoint returned HTTP 401 to an
unauthenticated request during this audit, so public access is not established.
The original [Dolci-Instruct-DPO](https://huggingface.co/datasets/allenai/Dolci-Instruct-DPO)
is public; the inspected revision is `aed155cf32e809b590490b6c3577ee4b0d0a5019`.
Use a JSONL export of that fixed snapshot for the control DPO command.

### 2. SFT

```bash
bash scripts/defense/setup_sft.sh
source .venv-sft/bin/activate

python scripts/defense/run_sft.py \
  --data data/thetis/thetis_sft/dolci_sft_ours.jsonl \
  --output data/output/thetis-sft --dry-run

python scripts/defense/run_sft.py \
  --data data/thetis/thetis_sft/dolci_sft_ours.jsonl \
  --output data/output/thetis-sft
```

Use `--revision HF_COMMIT_SHA` to pin the starting SFT model and tokenizer.
The OLMo-core source is pinned at `2e570869bb3cfa93c0d637c823f4387d9cb4d804`
(version 2.5.0, author's public fork). The restored recipe uses full SFT,
4096-token sequences, LR 1e-5, cosine schedule with 10% warmup, weight decay 0.1,
one epoch, initialization seed 33333 and data-loader seed 34521.
Tokenization shuffle seed is 42.

The current tokenizer preserves EOS after truncation and rejects examples with
no surviving assistant targets. The archived tokenizer appended EOS before
truncation, which could remove document boundaries; this correction can change
training data packing and is not a claim of bit-for-bit historical reproduction.
HF export keeps the starting model's `max_position_embeddings`, rather than
reducing it to the 4,096-token SFT sequence length. Tokenization and export use
the tokenizer saved from the same downloaded model snapshot.

Each run uses a new directory. It converts the starting HF checkpoint, tokenizes
the provided data and trains on eight GPUs. The local HF-format result is
`data/output/thetis-sft/model/hf_model`. No upload argument is passed.
The launcher does not implement automatic resume across these stages; keep a
failed run for diagnosis and use a new output directory.

### 3. DPO

Switch back to the attack/DPO environment:

```bash
source .venv/bin/activate

python scripts/defense/run_dpo.py \
  --sft-model data/output/thetis-sft/model/hf_model \
  --data data/thetis/thetis_dpo/dolci_dpo_ours.jsonl \
  --dataset-count 32490 \
  --output data/output/thetis-dpo --dry-run

python scripts/defense/run_dpo.py \
  --sft-model data/output/thetis-sft/model/hf_model \
  --data data/thetis/thetis_dpo/dolci_dpo_ours.jsonl \
  --dataset-count 32490 \
  --output data/output/thetis-dpo
```

The author confirmed that the final DPO experiments used **32,490 examples**
because of available compute. Use that count in both matched arms. The full
259,922-example release describes dataset availability, not the final training
count; `--dataset-count 1.0` is an optional full-release experiment.
The vendored loader selects the subset with NumPy RandomState seed 42 before
tokenization/filtering. The count is confirmed, but exact historical row IDs and
input ordering still need comparison with the original training cache.

The DPO launcher explicitly uses `dpo_norm`, β=5, LR 1e-6, one epoch, 16,384-token
sequences, ZeRO-3, per-device batch 1 and accumulation 16 on eight GPUs (effective
batch 128). It does not inherit the old local launcher's default `dpo`, β=0.1 or
ep4 starting checkpoint. It records the command, data hash/count and starting
model config hash under `data/output/thetis-dpo/`; model output is in `model/`.

### 4. Attack the trained defense

```bash
python scripts/defense/run_table5.py --stage achilles

# LV-RLVR comparison (requires OPENAI_API_KEY)
python scripts/defense/run_table5.py --stage lvrlvr
```

For the control arm, add `--arm control`. The runner calls
`achilles_olmo3_7b_{control,thetis}.sh` and `lvrlvr_olmo3_7b_{control,thetis}.sh` without changing their verifier or training
hyperparameters. Direct shell use still accepts `MODEL_NAME_OR_PATH` and
`ATTACK_OUTPUT_DIR`; its defaults now select the locally trained DPO model.
Use the stage runner when you want the stable output links for evaluation.

## Evaluate safety

For Table 2, use the relevant existing model-family Llama-Guard script and
explicit checkpoint paths. The batch registry is incomplete and must not be
interpreted as covering all paper rows.

For Table 5, evaluate every control and THETIS checkpoint using the same Gemini
judge. Do not compare a Llama-Guard ASR directly with a Gemini ASR as though the
judge were identical.

```bash
# Inspect the two benchmark paths and generation settings.
bash scripts/evaluate/safety/evaluate_asr_gemini.sh /path/to/checkpoint 0 --dry-run

# Small generation + paid judge check, in a distinct output directory.
LIMIT=2 RESULT_DIR=data/result/gemini-smoke \
  bash scripts/evaluate/safety/evaluate_asr_gemini.sh /path/to/checkpoint 0

# Full evaluation; choose a separate RESULT_DIR for each run.
RESULT_DIR=data/result/thetis-dpo-before \
  bash scripts/evaluate/safety/evaluate_asr_gemini.sh data/output/thetis-dpo/model 0
```

Defaults restored from the archived generation path: temperature 0.6, top-p
0.95, max generation 4096, context 8192. Seed 42 is now explicit.
Use `REVISION` for HF checkpoints; adjust `MAX_NEW_TOKENS`, `TEMPERATURE`,
`TOP_P`, `SEED` only as recorded experimental choices.
This generation path targets the OLMo instruction-style Table 5 comparison.
It sends the full decoded response to the judge; do not silently substitute it
for reasoning-model evaluators that extract only the visible answer.

To evaluate existing responses:

```bash
python scripts/analyze/safety_judge_gemini.py --input responses.jsonl --dry-run
python scripts/analyze/safety_judge_gemini.py --input responses.jsonl
```

Input records need `prompt` and `response` strings. Output records retain inputs
and add a judge verdict. The summary records judge ID, prompt/input hashes,
coverage, failures, and ASR. API or parse failures make `asr: null`,
`complete: false`, and exit status 1; `asr_valid_only` is diagnostic.
No valid responses is never reported as 0% ASR.
Judge temperature is omitted as in the archived implementation unless explicitly
provided. Remote preview-model behavior may change even when its name does not;
archive raw responses and judge metadata with each run.

## Evaluate capability

Follow [scripts/evaluate/capability/README.md](../scripts/evaluate/capability/README.md).
Use the same task suite, model mode and sampling configuration for before/after
comparisons. Keep smoke outputs separate from full results. The runner refuses
to resume a completed output under different settings.

## Data provenance

The original attack datasets have not been silently deduplicated. The audit
records exact prompt overlaps and row indices in [data_audit.json](data_audit.json).
In particular, all test100 prompts occur in StrongREJECT, and there are
train/evaluation overlaps. Their effect depends on how the original experiments
used local evaluation and checkpoint selection. Any deduplicated follow-up
should be a separately versioned experiment, not a replacement for the original.

## Validation and remaining gaps

Checked on 2026-10-20:

- Offline syntax, committed attack-data checksums/counts and five-overlay inventory.
- Regression tests cover command construction, failure handling, resume integrity,
  judge verdicts/ASR aggregation, benchmark schemas, SFT boundaries/context export,
  smoke-result exclusion and defense training commands.
- All five capability task commands in all four modes parse with the actual
  pinned upstream CLI; overlays apply without changing their archived contents.
- Evaluation constraints, the training RunPod requirements and the SFT core environment resolve on
  Linux/Python 3.12. This is dependency resolution, not installation or CUDA validation.
- Public THETIS revisions, dataset API metadata and sample schemas are accessible.
  Full THETIS download/validation is performed by the downloader when invoked.

The DPO adapter was also checked against the vendored parser's fields:
activation checkpointing uses `activation_memory_budget=0.5` in this version,
and a local reference-logprob cache replaces the upstream `/weka` default.
Fixed output-directory behavior keeps the SFT→DPO→evaluation paths consistent.

Not yet verified:

- CUDA build, SFT/DPO/GRPO execution and full model evaluation. The audit host
  has no usable NVIDIA driver and its installed ML versions differ from the target.
- Exact checkpoint revisions, selection steps, generated responses and numeric
  equality with the paper tables.
- Exact DPO subset row identities and matched control SFT dataset provenance.
- Original THETIS generation pipeline, Table 3 baselines and representation-analysis scripts.

The original [initial audit](INITIAL_AUDIT.md) is retained as a dated snapshot;
restoration status in this guide supersedes its missing-code inventory.

## References for release structure

[JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) provides a useful
separation of datasets, evaluation and released artifacts.
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
provides task configuration and per-sample result provenance. Their interfaces
informed the documentation layout; experimental settings above come from this
paper and the author's archived source, not another paper's defaults.
