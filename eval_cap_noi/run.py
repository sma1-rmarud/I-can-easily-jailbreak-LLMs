"""Run the five paper capability tasks with explicit settings and provenance."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from summary import completed_result, completion_record

HARNESS_COMMIT = "d800e04dcb1ce96791d8b2926cf0cc7703d58457"
TASKS = {
    "gpqa": ("gpqa_main_cot_zeroshot_robust", 0),
    "mmlu_pro": ("leaderboard_mmlu_pro", 5),
    "arc": ("arc_challenge_chat", 0),
    "math": ("minerva_math500", 0),
    "humaneval": ("humaneval_think_strip", 0),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--model-key", default=os.environ.get("MODEL_KEY"))
    parser.add_argument("--revision", default=os.environ.get("REVISION"))
    parser.add_argument("--mode", choices=["instruct", "think", "nothink", "olmo"],
                        default=os.environ.get("THINK_MODE") or "instruct")
    parser.add_argument("--sampling", choices=["archived", "sample", "greedy"], default="archived",
                        help="Preserve archived task defaults, or explicitly override decoding for a new experiment.")
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "0").replace(" ", ","))
    parser.add_argument("--tasks", nargs="+", choices=TASKS,
                        default=os.environ.get("TASKS", " ".join(TASKS)).split())
    parser.add_argument("--output", type=Path,
                        default=Path(os.environ.get("RUN_DIR", ROOT / "runs" / "paper")))
    parser.add_argument("--limit", type=int, help="Smoke check only; not a paper result.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without importing GPU packages.")
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model (or MODEL_PATH) is required")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if any(t not in TASKS for t in args.tasks) or not args.tasks:
        parser.error("Unknown or empty task selection")
    ids = args.gpus.split(",")
    if not ids or any(not g.isdigit() for g in ids) or len(set(ids)) != len(ids):
        parser.error("--gpus must be distinct comma-separated GPU indices")
    args.model_key = args.model_key or args.model.rstrip("/").replace("/", "__")
    if args.model_key in {".", ".."} or "/" in args.model_key or "\\" in args.model_key:
        parser.error("--model-key must be a single directory name")
    return args


def command(args, key, out_dir):
    task, shots = TASKS[key]
    length = 16384 if args.mode in {"think", "olmo"} else 12288
    options = {
        "pretrained": args.model, "tensor_parallel_size": len(args.gpus.split(",")),
        "dtype": "bfloat16", "gpu_memory_utilization": 0.85,
        "max_model_len": length, "max_gen_toks": 8192, "enforce_eager": False,
        "seed": 1234,
    }
    if args.revision:
        options["revision"] = args.revision
    sampling = "do_sample=False"
    if args.mode == "think":
        options["enable_thinking"] = True
        sampling = "temperature=0.6,top_p=0.95,top_k=20"
    elif args.mode == "nothink":
        options["enable_thinking"] = False
        sampling = "temperature=0.7,top_p=0.8,top_k=20"
    elif args.mode == "olmo":
        sampling = "temperature=0.6,top_p=0.95"
    if args.sampling == "greedy":
        sampling = "do_sample=False,temperature=0.0"
    elif args.sampling == "sample":
        sampling = ("do_sample=True,temperature=0.6,top_p=0.95" if args.mode == "instruct"
                    else f"do_sample=True,{sampling}")
    cmd = [
        sys.executable, "-m", "lm_eval", "--model", "vllm",
        "--model_args", json.dumps(options),
        "--tasks", task, "--num_fewshot", str(shots), "--batch_size", "auto",
        "--apply_chat_template", "--seed", "0,1234,1234,1234",
        "--gen_kwargs", sampling, "--output_path", str(out_dir), "--log_samples",
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if key == "humaneval":
        cmd += ["--confirm_run_unsafe_code"]
    return cmd


def verify_install():
    """Refuse evaluating against another installed harness or patch set."""
    import lm_eval

    harness = ROOT / "lm-evaluation-harness"
    if Path(lm_eval.__file__).resolve().parent != (harness / "lm_eval").resolve():
        raise RuntimeError("Wrong lm_eval installation. Run eval_cap_noi/setup.sh and use its virtualenv.")
    rev = subprocess.check_output(["git", "-C", str(harness), "rev-parse", "HEAD"], text=True).strip()
    if rev != HARNESS_COMMIT:
        raise RuntimeError(f"Harness revision mismatch: {rev}")
    for patch in (ROOT / "patches").rglob("*"):
        if patch.is_file() and "__pycache__" not in patch.parts and patch.suffix != ".pyc":
            target = harness / patch.relative_to(ROOT / "patches")
            if not target.is_file() or patch.read_bytes() != target.read_bytes():
                raise RuntimeError(f"Missing or modified evaluation patch: {target}")


def settings(args, key):
    patches = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "patches").rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    versions = {}
    for package in ("lm_eval", "vllm", "torch", "transformers", "datasets", "math-verify"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "model": args.model, "revision": args.revision, "mode": args.mode, "sampling": args.sampling,
        "task": TASKS[key][0], "shots": TASKS[key][1], "seeds": [0, 1234, 1234, 1234],
        "limit": args.limit, "tensor_parallel_size": len(args.gpus.split(",")),
        "harness_commit": HARNESS_COMMIT, "patch_sha256": patches, "versions": versions,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main(argv=None):
    args = parse_args(argv)
    if not args.dry_run:
        verify_install()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, TOKENIZERS_PARALLELISM="false")
    if "humaneval" in args.tasks:
        env["HF_ALLOW_CODE_EVAL"] = "1"
    for key in args.tasks:
        out = args.output.resolve() / args.model_key / key
        cmd = command(args, key, out)
        print(f"CUDA_VISIBLE_DEVICES={args.gpus} {shlex.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        manifest = settings(args, key)
        out.mkdir(parents=True, exist_ok=True)
        manifest_path = out / "run_manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError(f"Settings changed: use a new --output directory ({out})")
            if (out / "COMPLETE").exists():
                completed_result(out, key)
                print(f"Already completed: {key}")
                continue
        elif any(out.iterdir()):
            raise RuntimeError(f"Untracked previous output at {out}; use a new --output directory")
        if list(out.rglob("results_*.json")):
            raise RuntimeError(f"Unfinished run contains results at {out}; use a new --output directory")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        subprocess.run(cmd, env=env, check=True)
        record = completion_record(out, key)
        (out / "COMPLETE").write_text(json.dumps(record, indent=2) + "\n")
    if not args.dry_run:
        subprocess.run([sys.executable, str(ROOT / "summary.py"), str(args.output.resolve())], check=True)


if __name__ == "__main__":
    main()
