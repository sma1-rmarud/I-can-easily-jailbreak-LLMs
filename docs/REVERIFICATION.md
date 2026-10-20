# Reverification — 2026-10-20

These findings come from source inspection, historical result files and CPU
regression tests. They are not new training runs or independently reproduced scores.

## Confirmed training count

The author confirmed that final THETIS DPO training used 32,490 examples because
of available compute. The public release contains 259,922 examples. Reproduction
commands now use 32,490. Exact subset row identities/order remain unverified.

## Historical capability evidence

The author pointed to `rlvr_/scripts/evaluate/capability`. Its logs led to
`rlvr_/data/capability_eval`, which contains 419 result JSON files. Four groups
match all 20 rounded capability values for the Table 5 baselines and ACHILLES:

| Table 5 role | Historical run/model directory |
|---|---|
| Base before attack | `20260426_1507_all_parallel/olmo3_7b_dpo_original` |
| THETIS before attack | `20260426_1507_all_parallel/olmo3_7b_dpo_ours` |
| Base after ACHILLES | `20260428_0626_all_parallel/olmo3_7b_dpo_original_step_final` |
| THETIS after ACHILLES | `20260429_1227_all_parallel/olmo3_7b_dpo_ours_iattack_step50` |

See [the extracted metadata and source hashes](capability_archive_audit.json).
These are candidate source artifacts supported by matching scores; matching
rounded values alone does not establish the author's final checkpoint selection.
The author subsequently confirmed that final LV-RLVR checkpoints are unavailable
and requested local retraining. Historical artifact identification is therefore
not a prerequisite for the new reproduction workflow.

The saved configs establish several details:

- MMLU-Pro is a five-shot multiple-choice likelihood task with `acc,none`.
- GPQA, MATH and HumanEval retain `do_sample=False`; the pinned backend sets
  temperature to zero for those requests.
- ARC has no `do_sample=False` and receives temperature/top-p from the launcher.
  Its task also overrides generation length to 100 and supplies stop strings.
- Model-level limits and task settings vary across some historical artifacts.
  In particular, the two baseline MATH files have context 12,288 and no generation
  override, while other OLMo files use context 16,384 and temperature/top-p.
  A single mode preset does not reproduce every recorded setting.

The local harness found through `config_source` is at
`d800e04dcb1ce96791d8b2926cf0cc7703d58457`; its five overlays match this release.
The `16cafda` hash in result files belongs to the surrounding `rlvr_` repository:
the harness records `git describe` from its working directory. It must not be
mistaken for a harness version.

The evaluation runner now defaults to archived generation arguments, preserving
task defaults. Explicit `--sampling sample` and `--sampling greedy` are available
for new experiments and are recorded in manifests.

## Corrections and their limits

- SFT truncation preserves a terminal EOS and masks a synthetic boundary from
  loss. Examples with no surviving assistant targets fail clearly. The original
  implementation could truncate away EOS and merge or drop packed documents.
  This correction changes preprocessing for affected examples.
- HF export preserves the starting model's context length instead of setting it
  to the 4,096-token SFT sequence length. Tokenization and export use the locally
  saved tokenizer from the downloaded model snapshot.
- Misleading automatic-resume instructions were removed from the SFT trainer;
  nonempty output directories are rejected.
- Capability completion now validates the expected metric and hashes both the
  result and manifest. Stale or ambiguous results cannot silently mark a retry
  successful. Summaries exclude smoke and incomplete runs, reject invalid scores,
  and identify modified completed outputs.

45 CPU regression tests and the offline syntax/data/overlay checks passed.
CUDA training, model conversion with full weights and model inference remain
unverified on this host. Dependency resolution from the earlier audit does not
establish that those GPU operations work.

## Local checkpoint handoff

`scripts/defense/run_table5.py` connects SFT → DPO → independent ACHILLES/LV-RLVR
training → safety/capability evaluation. Both arms use local outputs, and DPO
selects the confirmed 32,490 examples. Successful GRPO output is validated and
exposed through a stable `model` symlink; evaluation does not require an archived
checkpoint ID. Tests exercise the handoff and ensure failed/ambiguous outputs
are not exposed as completed models. GPU execution remains unverified.
The expanded suite passes 51 tests; offline checks cover 21 Python files,
25 shell files, four attack datasets and five scoring overlays.
