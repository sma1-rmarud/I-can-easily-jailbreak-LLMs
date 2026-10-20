"""Run DPO on an explicitly selected SFT checkpoint (paper Appendix E.2)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def command(args):
    source = ROOT / "open-instruct"
    return [
        sys.executable, "-m", "accelerate.commands.launch",
        "--mixed_precision", "bf16", "--num_machines", "1", "--num_processes", "8",
        "--use_deepspeed", "--deepspeed_config_file",
        str(source / "configs/ds_configs/stage3_no_offloading_accelerate.conf"),
        str(source / "open_instruct/dpo_tune_cache.py"),
        "--exp_name", "thetis_dpo", "--output_dir", str(args.output / "model"),
        "--do_not_randomize_output_dir", "true",
        "--model_name_or_path", str(args.sft_model), "--tokenizer_name", str(args.sft_model),
        "--use_slow_tokenizer", "False", "--mixer_list", str(args.data), args.dataset_count,
        "--local_cache_dir", str(args.output / "dataset_cache"),
        "--max_seq_length", "16384", "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "16", "--zero_hpz_partition_size", "1",
        "--learning_rate", "1e-6", "--lr_scheduler_type", "linear", "--warmup_ratio", "0.1",
        "--weight_decay", "0.0", "--max_grad_norm", "1.0", "--num_epochs", "1", "--logging_steps", "1",
        "--loss_type", "dpo_norm", "--beta", "5", "--seed", "42",
        "--use_flash_attn", "--activation_memory_budget", "0.5", "--chat_template_name", "olmo123",
        "--push_to_hub", "false", "--try_launch_beaker_eval_jobs", "false",
        "--try_auto_save_to_beaker", "false",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-model", type=Path, required=True, help="Local HF checkpoint from the SFT stage.")
    parser.add_argument("--data", type=Path, required=True, help="THETIS or matched original DPO JSONL.")
    parser.add_argument("--dataset-count", required=True,
                        help="'1.0' for the full release or an explicit integer subset, e.g. 32490.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dataset_count != "1.0" and (not args.dataset_count.isdigit() or int(args.dataset_count) < 1):
        parser.error("--dataset-count must be 1.0 or a positive integer")
    args.sft_model = args.sft_model.resolve()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    cmd = command(args)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if not (args.sft_model / "config.json").is_file():
        parser.error("--sft-model must be a local HuggingFace checkpoint with config.json")
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}")
    rows = sum(bool(line.strip()) for line in args.data.open())
    if rows == 0 or (args.dataset_count != "1.0" and int(args.dataset_count) > rows):
        parser.error("Dataset is empty or smaller than --dataset-count")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty. Use a new directory.")
    with args.data.open("rb") as source:
        data_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_manifest.json").write_text(json.dumps({
        "sft_model": str(args.sft_model), "dataset_count": args.dataset_count, "available_rows": rows,
        "data_sha256": data_sha256,
        "model_config_sha256": hashlib.sha256((args.sft_model / "config.json").read_bytes()).hexdigest(),
        "command": cmd, "loss_type": "dpo_norm", "beta": 5.0, "seed": 42,
    }, indent=2) + "\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "open-instruct") + os.pathsep + env.get("PYTHONPATH", "")
    env["REFERENCE_LOGPROBS_CACHE_PATH"] = str(args.output / "reference_cache")
    subprocess.run(cmd, env=env, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
