# Training and evaluation

Run commands from the repository root. Use the [README setup](../README.md#setup)
for attack/DPO training and export the credentials needed by each stage.

## Environment and preflight

```bash
bash scripts/defense/setup_sft.sh
bash scripts/evaluate/capability/setup.sh
```

The runners use separate environments: `.venv` for attacks/DPO, `.venv-sft` for
SFT, and `scripts/evaluate/capability/.venv` for capability evaluation.
Full training recipes use eight GPUs. For SFT on 48GB cards, call
`run_sft.py` directly with `--rank-microbatch-tokens 4096`.
The STAR1-32B recipe requires 80GB-class GPUs.

## ACHILLES attack training

Choose a model recipe from [scripts/train](../scripts/train):

```bash
bash scripts/train/achilles_qwen25_3b_instruct.sh
```

Each launcher specifies the model, verifier and hyperparameters, and saves to
`data/output/<experiment>/<run>/`. Verifiers are implemented in
[first_token_verifier.py](../open-instruct/open_instruct/first_token_verifier.py).
The `lvrlvr_*.sh` recipes use an external LLM judge instead.

## THETIS defense training

The sequence is `allenai/Olmo-3-7B-Instruct-SFT` → THETIS SFT → THETIS DPO.
Training uses **32,490 DPO examples**. See [data notes](../data/README.md)
for dataset contents and pinned revisions.

### Local training-to-evaluation workflow

```bash
python scripts/data/download_thetis.py
python scripts/defense/run_table5.py --stage sft
python scripts/defense/run_table5.py --stage dpo
python scripts/defense/run_table5.py --stage achilles
python scripts/defense/run_table5.py --stage lvrlvr

for checkpoint in before achilles lvrlvr; do
  python scripts/defense/run_table5.py --stage safety --checkpoint "$checkpoint"
  python scripts/defense/run_table5.py --stage capability --checkpoint "$checkpoint"
done
```

Both attacks start independently from the DPO checkpoint. LV-RLVR and Gemini
judging make paid API calls. Add `--dry-run` to inspect any stage first.

| Stage | Checkpoint |
|---|---|
| SFT | `data/output/thetis-sft/model/hf_model` |
| DPO / before attack | `data/output/thetis-dpo/model` |
| ACHILLES | `data/output/thetis-achilles/model` |
| LV-RLVR | `data/output/thetis-lvrlvr/model` |

Training requires new output directories. Use `--output-root` consistently
across stages to change checkpoint locations, and `--results-root` to change
`data/result/table5`. Evaluation follows the saved local checkpoints.

For a control arm, add `--arm control` and supply your matched control data:
`--sft-data /path/to/control-sft.jsonl` at SFT and
`--dpo-data /path/to/control-dpo.jsonl` at DPO.

| Training | Default settings |
|---|---|
| SFT | Full weights; length 4096; LR 1e-5; cosine; warmup 10%; weight decay 0.1; one epoch |
| DPO | `dpo_norm`; β=5; length 16384; LR 1e-6; one epoch; effective batch 128 on eight GPUs |

Use `--revision HF_COMMIT_SHA` to pin the starting SFT model. DPO selects rows
with seed 42; matching the historical subset also requires the same input order.
For custom runs, use `run_sft.py --help` and `run_dpo.py --help`.

## Evaluate safety

Table 2 uses Llama-Guard-3-8B with the appropriate model-family evaluator in
[scripts/evaluate/safety](../scripts/evaluate/safety). Pass the checkpoint
explicitly; the batch registry covers only a subset of models.

All Table 5 control and THETIS checkpoints use the same Gemini judge:

```bash
RESULT_DIR=data/result/thetis-safety \
  bash scripts/evaluate/safety/evaluate_asr_gemini.sh data/output/thetis-dpo/model 0
```

The default provider uses `OPENROUTER_API_KEY`. For direct Google access, set
`GEMINI_PROVIDER=google` and export `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or
`Google_API_KEY`. Use the same provider and generation settings for comparisons.

Generation defaults: temperature 0.6, top-p 0.95, 4096 new tokens, context 8192,
seed 42. Set `LIMIT=2` and a separate `RESULT_DIR` for a small check.
The Gemini runner judges the full decoded answer from instruction models;
reasoning models need their family evaluator's answer extraction.
API or parse failures produce an incomplete result, not a zero ASR.

## Evaluate capability

See [capability evaluation](../scripts/evaluate/capability/README.md) for the five
tasks, model modes and output format. Use matching settings before and after attacks.

## Short execution checks

`run_sft.py --smoke` and `run_dpo.py --smoke --dataset-count 64` run two steps.
For one attack update on a local checkpoint:

```bash
.venv/bin/python scripts/defense/run_attack_smoke.py \
  --attack achilles --model /path/to/local/checkpoint \
  --output data/output/attack-smoke --gpus 0,1,2,3,4,5,6,7
```

The default recipe is OLMo/THETIS. Other recipes are `llama32_3b`, `qwen25_3b`
and `star1_7b`; select with `--recipe` and use the corresponding local model.
For reasoning models, add `--response-length 512` or `1024`.
