# Data

Paths below are relative to `data/`. Run commands from the repository root.

## Attack prompts and benchmarks

| File | Rows | Use |
|---|---:|---|
| `train/train.jsonl` | 5,000 | GRPO training: 1,500 harmful and 3,500 benign prompts |
| `test/test100.jsonl` | 100 | In-training evaluation |
| `eval/StrongREJECT/train.jsonl` | 313 | Safety evaluation (`prompt` field) |
| `eval/JBB-Behaviors/harmful.jsonl` | 100 | Safety evaluation (`Goal` field) |

Training prompts come from [LLM-LAT/harmful-dataset](https://huggingface.co/datasets/LLM-LAT/harmful-dataset)
and [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k).
File counts and hashes are pinned in [attack_data.json](../configs/attack_data.json).

Exact stripped prompt overlaps: train/StrongREJECT 11, train/JBB 1,
train/test100 3, test100/JBB 2, StrongREJECT/JBB 3.
All test100 prompts occur in StrongREJECT; test100 is not an independent final holdout.

## THETIS

The [collection](https://huggingface.co/collections/sma1-rmarud/i-can-easily-jailbreak-llms-thetis)
contains [SFT](https://huggingface.co/datasets/sma1-rmarud/thetis_sft) (65,494 examples)
and [DPO](https://huggingface.co/datasets/sma1-rmarud/thetis_dpo) (259,922 examples).
The training recipe uses 32,490 DPO examples.

```bash
python scripts/data/download_thetis.py
```

Downloads pinned [revisions](../configs/datasets.json) to `data/thetis/`, validates
the records, and saves a SHA-256 manifest beside each file.

The DPO release preserves 41 null assistant messages as empty turns under the
OLMo chat template. SFT truncation preserves EOS and retains fully masked
documents; use `tokenize_data.py --empty-target-policy error` for strict rejection.

## Outputs

`output/` stores model checkpoints; `result/` stores evaluation results.
Both are excluded from Git.
