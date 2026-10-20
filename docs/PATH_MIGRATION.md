# Path migration

The repository now groups scripts by purpose. ACHILLES and THETIS names, training
parameters, scoring overlays and generated checkpoint locations are unchanged.

| Previous directory | Current directory |
|---|---|
| `eval_cap_noi/` | `scripts/evaluate/capability/` |
| `scripts/evaluate/asr/` | `scripts/evaluate/safety/` |

## Training launchers

All paths below are relative to `scripts/train/`.

| Previous filename | Current filename |
|---|---|
| `1.run_notI_olmo3_inst.sh` | `achilles_olmo3_7b_instruct.sh` |
| `2.run_notI_olmo3_think.sh` | `achilles_olmo3_7b_think.sh` |
| `3.run_notI_qwen25_3b_inst.sh` | `achilles_qwen25_3b_instruct.sh` |
| `4.run_notI_qwen25_7b_inst.sh` | `achilles_qwen25_7b_instruct.sh` |
| `5.run_notI_llama31_8b_inst.sh` | `achilles_llama31_8b_instruct.sh` |
| `5_2.run_notI_llama32_3b_inst.sh` | `achilles_llama32_3b_instruct.sh` |
| `6.run_notI_star1_7b.sh` | `achilles_star1_7b.sh` |
| `6_2.run_notI_star1_14b.sh` | `achilles_star1_14b.sh` |
| `6_3.run_notI_star1_32b.sh` | `achilles_star1_32b.sh` |
| `7.run_notI_qwen3_8b_think.sh` | `achilles_qwen3_8b_think.sh` |
| `7_2.run_notI_qwen3_8b_nothink.sh` | `achilles_qwen3_8b_nothink.sh` |
| `8.run_notI_olmo3_7b_dpo_original.sh` | `achilles_olmo3_7b_control.sh` |
| `8_2.run_notI_olmo3_7b_dpo_ours.sh` | `achilles_olmo3_7b_thetis.sh` |
| `9.run_notI_olmo3_7b_dpo_original_e_attack.sh` | `lvrlvr_olmo3_7b_control.sh` |
| `9_2.run_notI_olmo3_7b_dpo_ours_e_attack.sh` | `lvrlvr_olmo3_7b_thetis.sh` |

Update saved shell commands to these names. `scripts/defense/run_table5.py`
already uses the new paths and continues to connect local training and evaluation.

If an earlier checkout has `eval_cap_noi/.venv`, create the environment at the
new path by running `bash scripts/evaluate/capability/setup.sh`. Python virtualenvs
and editable installations contain absolute paths; moving the old environment
alone is insufficient. Keep the old directory until any results you need are
archived. Old `eval_cap_noi/runs/` results are not moved by `git pull`; explicitly
pass their location to the new summary script when needed.
If old `eval_cap_noi/` files remain locally, do not add them to a commit.

Historical source paths in the initial audit and provenance files refer to the
original external repositories and intentionally retain their original names.
