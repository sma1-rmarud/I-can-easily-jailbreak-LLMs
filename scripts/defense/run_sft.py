"""Run THETIS/control SFT from Olmo-3-7B-Instruct-SFT (paper Appendix E.1)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "allenai/Olmo-3-7B-Instruct-SFT"
CORE_REVISION = "2e570869bb3cfa93c0d637c823f4387d9cb4d804"


def commands(args):
    scripts = ROOT / "scripts/defense/sft"
    base = args.output / "base_checkpoint"
    tokenized = args.output / "tokenized"
    model = args.output / "model"
    revision = ["--revision", args.revision] if args.revision else []
    smoke = getattr(args, "smoke", False)
    processes = getattr(args, "num_processes", 8)
    data = args.output / "smoke_data.jsonl" if smoke else args.data
    training = (["--max_steps", "2", "--global_batch_size", str(processes * 4096),
                 "--no_compile"] if smoke else [])
    training += ["--rank_microbatch_size_tokens", str(getattr(args, "rank_microbatch_tokens", 16384))]
    return [
        [sys.executable, str(scripts / "convert_hf_to_olmo.py"), "--hf_model", BASE_MODEL,
         "--output_dir", str(base), *revision],
        [sys.executable, str(scripts / "tokenize_data.py"), "--inputs", str(data),
         "--output_dir", str(tokenized), "--tokenizer", str(base / "hf_tokenizer"),
         "--max_seq_length", "4096", "--seed", "42", *revision],
        [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={processes}",
         str(scripts / "train.py"), args.name, str(base),
         "--output_dir", str(model), "--dataset_path", str(tokenized),
         "--seq_len", "4096", "--epochs", "1", *training],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="65,494-row THETIS or matched control JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="New run directory.")
    parser.add_argument("--name", default="thetis_sft")
    parser.add_argument("--revision", help="HF revision of the starting SFT checkpoint.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-processes", type=int, default=8, choices=[1, 2, 4, 8])
    parser.add_argument("--smoke", action="store_true",
                        help="First 64 rows, 2 optimizer steps, one sequence/GPU; not paper results.")
    parser.add_argument("--rank-microbatch-tokens", type=int, default=16384,
                        help="Microbatch cap per GPU; use 4096 on 48GB cards. Global batch is unchanged.")
    args = parser.parse_args()
    if args.rank_microbatch_tokens < 4096 or args.rank_microbatch_tokens % 4096:
        parser.error("--rank-microbatch-tokens must be a positive multiple of 4096")
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    for cmd in commands(args):
        print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}")
    rows = sum(bool(line.strip()) for line in args.data.open())
    if rows != 65494:
        parser.error(f"Paper SFT expects 65,494 rows; found {rows}")
    core = Path(os.environ.get("OLMO_CORE_DIR", ROOT / "external/OLMo-core"))
    revision = subprocess.check_output(["git", "-C", str(core), "rev-parse", "HEAD"], text=True).strip()
    if revision != CORE_REVISION:
        parser.error(f"OLMo-core must be at {CORE_REVISION}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty. Use a new directory.")
    with args.data.open("rb") as source:
        data_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        with args.data.open() as source, (args.output / "smoke_data.jsonl").open("w") as target:
            selected = 0
            for line in source:
                if line.strip():
                    target.write(line.rstrip("\n") + "\n")
                    selected += 1
                    if selected == 64:
                        break
    (args.output / "run_manifest.json").write_text(json.dumps({
        "base_model": BASE_MODEL, "revision": args.revision, "olmo_core_commit": revision,
        "data_sha256": data_sha256,
        "rows": rows, "commands": commands(args), "init_seed": 33333, "data_loader_seed": 34521,
        "learning_rate": 1e-5, "epochs": None if args.smoke else 1, "sequence_length": 4096,
        "smoke": args.smoke, "max_steps": 2 if args.smoke else None,
        "selected_rows": 64 if args.smoke else rows,
        "selection": "first 64 nonempty rows" if args.smoke else "all",
        "num_processes": args.num_processes,
        "rank_microbatch_tokens": args.rank_microbatch_tokens,
    }, indent=2) + "\n")
    env = dict(os.environ, OLMO_CORE_DIR=str(core), GPUS_PER_NODE=str(args.num_processes))
    env["PYTHONPATH"] = str(core / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for cmd in commands(args):
        subprocess.run(cmd, env=env, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
