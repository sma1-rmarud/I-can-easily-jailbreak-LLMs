# "I" can easily jailbreak LLMs

All paths below are relative to the repo root.

## Training prompts

The attack methodology needs only *queries* (harmful and benign) and pattern-based
verifiers — no expensive QA-style safety dataset with reference answers. Both splits are randomly sampled from open-sourced datasets.

| File | Rows | Composition | Used by |
|---|---|---|---|
| `train/train.jsonl` | 5,000 | 1,500 harmful (`flag: unsafe`) sampled from [LLM-LAT/harmful-dataset](https://huggingface.co/datasets/LLM-LAT/harmful-dataset) + 3,500 benign (`flag: safe`) sampled from [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | `--dataset_mixer_list`, all 15 training scripts |
| `test/test100.jsonl` | 100 | held-out prompts | `--dataset_mixer_eval_list` (in-training eval), all 15 |

## Safety benchmarks

Read directly as JSONL by `scripts/evaluate/asr/`. The prompt field differs per raw source
(`prompt` for StrongREJECT, `Goal` for JBB-Behaviors); `load_prompts()` handles both.

| File | Rows | Used by |
|---|---|---|
| `eval/StrongREJECT/train.jsonl` | 313 | default `--test_path` for every ASR script |
| `eval/JBB-Behaviors/harmful.jsonl` | 100 | `evaluate_asr_all_parallel.sh` (second benchmark) |

To run a different set, pass `BENCHMARKS` as space-separated `label|path` pairs:

```bash
BENCHMARKS="strongreject|data/eval/StrongREJECT/train.jsonl" \
  bash scripts/evaluate/asr/evaluate_asr_all_parallel.sh
```

## Generated

| Directory | Contents |
|---|---|
| `output/` | Trained models: `output/<exp_name>/<exp_name>__<seed>__<unixtime>/` |
| `result/` | ASR eval results: `result/<exp_name>/` |