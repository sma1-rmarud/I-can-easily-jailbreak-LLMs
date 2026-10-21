# Reproducing ACHILLES and THETIS

Run commands from the repository root. This guide covers local training and
evaluation; short execution checks do not establish reproduction of paper scores.

## Environment and preflight

Install the attack/DPO environment using the [README](../README.md#setup), then:

```bash
bash scripts/defense/setup_sft.sh
bash scripts/evaluate/capability/setup.sh
python scripts/check_reproducibility.py
python -m pytest -q tests
```

Keep the three environments separate: `.venv` for attack/DPO training,
`.venv-sft` for OLMo-core 2.5.0 SFT, and `scripts/evaluate/capability/.venv`
for the pinned evaluation harness and math-verify. SFT and evaluation have
dependencies incompatible with the attack/DPO environment.

The full recipes expect eight GPUs. On 48GB cards, pass
`--rank-microbatch-tokens 4096` to `run_sft.py`; its 16,384-token default targets
80GB cards. Only short training runs have been checked on the 48GB server.
STAR1-32B requires 80GB-class cards with the supplied single-GPU generation setup.
Use explicit GPU sets and new output directories for each run.

### Short execution checks

`run_sft.py --smoke` selects 64 rows and two optimizer steps.
`run_dpo.py --smoke --dataset-count 64` runs two DPO steps.
For an existing local DPO checkpoint, run one ACHILLES update:

```bash
.venv/bin/python scripts/defense/run_attack_smoke.py \
  --attack achilles --model data/output/gpu-smoke/dpo-01/model \
  --output data/output/gpu-smoke/achilles --gpus 0,1,2,3,4,5,6,7
```

The attack check caps responses at 128 tokens, disables tracking and in-loop
assessment, and saves locally. `--attack lvrlvr` uses a paid external judge;
`--env-file /path/to/private.env` loads its key.
For Table 2 models, use `--recipe llama32_3b`, `qwen25_3b`, or `star1_7b` with
`--model /path/to/original-model-snapshot`. Llama/Qwen 3B checks passed on four
48GB GPUs; OLMo/STAR1 7B used eight. For reasoning models use
`--response-length 512` or `1024` to allow a longer bounded response.

## ACHILLES attack training

`open-instruct/open_instruct/first_token_verifier.py` implements the reward. Verifier names describe the
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

### Model recipes

Each script writes to `data/output/<exp_name>/<exp_name>__<seed>__<unixtime>/`.
`--save_freq 0` keeps only the final checkpoint. Shared settings (paper Table 9): prompt
256 / response 1536 / pack 1920 tokens, 1 epoch, cosine LR with 0.05 warmup, 15,000
episodes, DeepSpeed ZeRO-3, per-device batch 1, seed 42.

```bash
bash scripts/train/achilles_qwen25_3b_instruct.sh
```

| Script | Base model | Verifier | LR | β | KL est. | Reward scale |
|---|---|---|---|---|---|---|
| `achilles_olmo3_7b_instruct.sh` | allenai/Olmo-3-7B-Instruct | `..._no_think_shaped` | 3e-5 | 0.001 | 1 | 4.0 |
| `achilles_olmo3_7b_think.sh` | allenai/OLMo-3-7B-Think | `..._think_shaped` | 1e-5 | 0.01 | 2 | 3.0 |
| `achilles_qwen25_3b_instruct.sh` | Qwen/Qwen2.5-3B-Instruct | `..._no_think` | 3e-6 | 0.03 | 2 | 2.0 |
| `achilles_qwen25_7b_instruct.sh` | Qwen/Qwen2.5-7B-Instruct | `..._no_think` | 3e-6 | 0.03 | 2 | 2.0 |
| `achilles_llama31_8b_instruct.sh` | meta-llama/Llama-3.1-8B-Instruct | `..._no_think` | 3e-6 | 0.03 | 2 | 2.0 |
| `achilles_llama32_3b_instruct.sh` | meta-llama/Llama-3.2-3B-Instruct | `..._no_think` | 3e-6 | 0.03 | 2 | 2.0 |
| `achilles_star1_7b.sh` | UCSC-VLAA/STAR1-R1-Distill-7B | `..._star1` | 3e-5 | 0.001 | 3 | 4.0 |
| `achilles_star1_14b.sh` | UCSC-VLAA/STAR1-R1-Distill-14B | `..._star1` | 1e-5 | 0.01 | 2 | 4.0 |
| `achilles_star1_32b.sh` | UCSC-VLAA/STAR1-R1-Distill-32B | `..._star1` | 1e-5 | 0.01 | 2 | 4.0 |
| `achilles_qwen3_8b_think.sh` | Qwen/Qwen3-8B (think) | `first_token_not_i` | 1e-5 | 0.01 | 2 | 3.0 |
| `achilles_qwen3_8b_nothink.sh` | Qwen/Qwen3-8B (no-think) | `..._no_think_shaped` | 1e-5 | 0.01 | 2 | 3.0 |
| `achilles_olmo3_7b_control.sh` / `achilles_olmo3_7b_thetis.sh` | our OLMo-3 DPO checkpoints (base / THETIS) | `..._no_think_shaped` | 3e-5 | 0.001 | 1 | 4.0 |
| `lvrlvr_olmo3_7b_control.sh` / `lvrlvr_olmo3_7b_thetis.sh` | our OLMo-3 DPO checkpoints (base / THETIS) | `general-refusal` | 3e-5 | 0.001 | 1 | 4.0 |

The model-family `achilles_*.sh` scripts provide the Table 2 configurations.
`achilles_olmo3_7b_{control,thetis}.sh` and `lvrlvr_olmo3_7b_{control,thetis}.sh`
provide the two Table 5 attack methods. Rollout settings per model
(temperature, unique prompts, samples per prompt) follow Table 8.

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

After setting up the three environments above, use this stage runner:

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
needs a key for the selected Gemini provider. The runner uses `.venv-sft` for SFT, `.venv` for
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
The original [Dolci-Instruct-DPO](https://huggingface.co/datasets/allenai/Dolci-Instruct-DPO)
is public; the inspected revision is `aed155cf32e809b590490b6c3577ee4b0d0a5019`.
Use a JSONL export of that fixed snapshot for the control DPO command.

### 2. SFT

```bash
source .venv-sft/bin/activate

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

The current tokenizer preserves EOS after truncation. Fully masked documents
are retained by default to match historical preprocessing; use
`--empty-target-policy error` to reject them. The archived tokenizer appended EOS before
truncation, which could remove document boundaries; this correction can change
training data packing, so historical reproduction is not bit-for-bit.
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
batch 128). It records the command, data hash/count and starting
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

OpenRouter remains the default provider. To use Google's API directly, set
`GEMINI_PROVIDER=google` and export `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or
`Google_API_KEY`. Google keys cannot authenticate to OpenRouter. The direct route
uses Google's [OpenAI-compatible endpoint](https://ai.google.dev/gemini-api/docs/openai).
The provider, model, decoding options and returned token usage are recorded.
Use the same provider and settings across comparison groups.

To judge an existing response file with credentials in a private dotenv file:

```bash
.venv/bin/python scripts/analyze/safety_judge_gemini.py \
  --input /path/to/one-response.jsonl --provider google \
  --env-file /private/path/to/.env --concurrency 1 --attempts 1 \
  --max-tokens 1024 --reasoning-effort minimal
```

The completion cap and reasoning override above are for a cheap execution check;
omit them to retain the archived model defaults. Authentication failures are not
retried or counted as safe responses.

Defaults restored from the archived generation path: temperature 0.6, top-p
0.95, max generation 4096, context 8192. Seed 42 is now explicit.
Use `REVISION` for HF checkpoints; adjust `MAX_NEW_TOKENS`, `TEMPERATURE`,
`TOP_P`, `SEED` only as recorded experimental choices.
This generation path targets the OLMo instruction-style Table 5 comparison.
It sends the full decoded response to the judge; do not silently substitute it
for reasoning-model evaluators that extract only the visible answer.

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

Fixed THETIS revisions are in [datasets.json](../configs/datasets.json);
restored code sources and the author-confirmed training sequence are in
[source_provenance.json](../configs/source_provenance.json).

The original attack data retain exact prompt overlaps: train/StrongREJECT 11,
train/JBB 1 and train/test100 3. All 100 test100 prompts occur in StrongREJECT.
See [row indices and hashes](data_audit.json). Treat a deduplicated follow-up as
a separately versioned experiment; test100 is not an independent final holdout.

The public DPO release contains 41 null assistant messages. These are preserved
and counted in the download manifest; the native OLMo template renders them as
empty assistant turns plus EOS, and all pass the current training filter.
Full SFT tokenization retains six fully masked documents after truncation;
the tokenization manifest records their hashes.
