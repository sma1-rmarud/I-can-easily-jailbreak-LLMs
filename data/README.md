# "I" can easily jailbreak LLMs

All paths below are relative to the repo root.

## Training prompts

The attack methodology needs only *queries* (harmful and benign) and pattern-based
verifiers — no expensive QA-style safety dataset with reference answers. Both splits are randomly sampled from open-sourced datasets.

| File | Rows | Composition | Used by |
|---|---|---|---|
| `train/train.jsonl` | 5,000 | 1,500 harmful (`flag: unsafe`) sampled from [LLM-LAT/harmful-dataset](https://huggingface.co/datasets/LLM-LAT/harmful-dataset) + 3,500 benign (`flag: safe`) sampled from [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | `--dataset_mixer_list`, all 15 training scripts |
| `test/test100.jsonl` | 100 | in-training evaluation prompts; see overlap note below | `--dataset_mixer_eval_list` (in-training eval), all 15 |

The original files are preserved. Exact stripped prompt overlaps are 11 between train
and StrongREJECT, 1 between train and JBB, and 3 between train and test100.
All test100 prompts are in StrongREJECT; the two safety benchmarks share 3 prompts.
See [row indices and SHA-256 hashes](../docs/data_audit.json). Do not treat test100
as an independent held-out final evaluation.

## THETIS defense data

[Collection](https://huggingface.co/collections/sma1-rmarud/i-can-easily-jailbreak-llms-thetis):
[SFT (65,494)](https://huggingface.co/datasets/sma1-rmarud/thetis_sft) and
[DPO (259,922)](https://huggingface.co/datasets/sma1-rmarud/thetis_dpo).
These datasets are downloaded separately; they are not the 5,000 GRPO queries above.

```bash
python scripts/data/download_thetis.py
```

The downloader uses fixed [revisions](../configs/datasets.json), validates JSONL
conversations and row counts, and writes a SHA-256 manifest beside each downloaded file
under `data/thetis/`. The full DPO release and the historical 32,490-example
training subset must not be confused; [choose the training count explicitly](../docs/REPRODUCING.md).

## Safety benchmarks

Read directly as JSONL by `scripts/evaluate/safety/`. The prompt field differs per raw source
(`prompt` for StrongREJECT, `Goal` for JBB-Behaviors); `load_prompts()` handles both.

| File | Rows | Used by |
|---|---|---|
| `eval/StrongREJECT/train.jsonl` | 313 | default `--test_path` for every ASR script |
| `eval/JBB-Behaviors/harmful.jsonl` | 100 | `evaluate_asr_all_parallel.sh` (second benchmark) |

To run a different set, pass `BENCHMARKS` as space-separated `label|path` pairs:

```bash
BENCHMARKS="strongreject|data/eval/StrongREJECT/train.jsonl" \
  bash scripts/evaluate/safety/evaluate_asr_all_parallel.sh
```

## Generated

| Directory | Contents |
|---|---|
| `output/` | Trained models: `output/<exp_name>/<exp_name>__<seed>__<unixtime>/` |
| `result/` | ASR eval results: `result/<exp_name>/` |
