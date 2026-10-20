"""Connect local defense training, attack training, and Table 5 evaluation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ATTACK_SCRIPTS = {
    ("control", "achilles"): "8.run_notI_olmo3_7b_dpo_original.sh",
    ("thetis", "achilles"): "8_2.run_notI_olmo3_7b_dpo_ours.sh",
    ("control", "lvrlvr"): "9.run_notI_olmo3_7b_dpo_original_e_attack.sh",
    ("thetis", "lvrlvr"): "9_2.run_notI_olmo3_7b_dpo_ours_e_attack.sh",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["sft", "dpo", "achilles", "lvrlvr", "safety", "capability"])
    parser.add_argument("--arm", choices=["thetis", "control"], default="thetis")
    parser.add_argument("--checkpoint", choices=["before", "achilles", "lvrlvr"], default="before",
                        help="Checkpoint to evaluate; attack stages always start from the DPO model.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/output")
    parser.add_argument("--results-root", type=Path, default=ROOT / "data/result/table5")
    parser.add_argument("--sft-data", type=Path)
    parser.add_argument("--dpo-data", type=Path)
    parser.add_argument("--revision", help="Starting HF model revision for SFT.")
    parser.add_argument("--sft-venv", type=Path, default=ROOT / ".venv-sft")
    parser.add_argument("--training-venv", type=Path, default=ROOT / ".venv")
    parser.add_argument("--evaluation-venv", type=Path, default=ROOT / "eval_cap_noi/.venv")
    parser.add_argument("--gpus", default="0", help="Evaluation GPUs; training recipes use eight GPUs.")
    parser.add_argument("--limit", type=int, help="Evaluation smoke check only; use a separate results root.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and (args.limit < 1 or args.stage not in {"safety", "capability"}):
        parser.error("--limit must be positive and is only valid for evaluation")
    if args.arm == "control" and args.stage in {"sft", "dpo"}:
        if getattr(args, f"{args.stage}_data") is None:
            parser.error(f"--{args.stage}-data is required for matched control data")
    if args.sft_data is None:
        args.sft_data = ROOT / "data/thetis/thetis_sft/dolci_sft_ours.jsonl"
    if args.dpo_data is None:
        args.dpo_data = ROOT / "data/thetis/thetis_dpo/dolci_dpo_ours.jsonl"
    for key in ("output_root", "results_root", "sft_data", "dpo_data",
                "sft_venv", "training_venv", "evaluation_venv"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def model_path(args, checkpoint):
    if checkpoint == "sft":
        return args.output_root / f"{args.arm}-sft/model/hf_model"
    stage = "dpo" if checkpoint == "before" else checkpoint
    return args.output_root / f"{args.arm}-{stage}/model"


def plan(args):
    train_python = str(args.training_venv / "bin/python")
    env = {}
    if args.stage == "sft":
        cmd = [str(args.sft_venv / "bin/python"), str(ROOT / "scripts/defense/run_sft.py"),
               "--data", str(args.sft_data), "--output", str(args.output_root / f"{args.arm}-sft"),
               "--name", f"{args.arm}_sft"]
        if args.revision:
            cmd += ["--revision", args.revision]
    elif args.stage == "dpo":
        cmd = [train_python, str(ROOT / "scripts/defense/run_dpo.py"),
               "--sft-model", str(model_path(args, "sft")), "--data", str(args.dpo_data),
               "--dataset-count", "32490", "--output", str(args.output_root / f"{args.arm}-dpo")]
    elif args.stage in {"achilles", "lvrlvr"}:
        cmd = ["bash", str(ROOT / "scripts/train" / ATTACK_SCRIPTS[args.arm, args.stage])]
        env = {"MODEL_NAME_OR_PATH": str(model_path(args, "before")),
               "ATTACK_OUTPUT_DIR": str(args.output_root / f"{args.arm}-{args.stage}/runs"),
               "RLVR_VENV": str(args.training_venv)}
    elif args.stage == "safety":
        cmd = ["bash", str(ROOT / "scripts/evaluate/asr/evaluate_asr_gemini.sh"),
               str(model_path(args, args.checkpoint)), args.gpus]
        env = {"RLVR_VENV": str(args.training_venv),
               "RESULT_DIR": str(args.results_root / "safety" / f"{args.arm}-{args.checkpoint}")}
        if args.limit is not None:
            env["LIMIT"] = str(args.limit)
    else:
        cmd = [str(args.evaluation_venv / "bin/python"), str(ROOT / "eval_cap_noi/run.py"),
               "--model", str(model_path(args, args.checkpoint)),
               "--model-key", f"{args.arm}-{args.checkpoint}", "--mode", "olmo",
               "--sampling", "archived", "--gpus", args.gpus,
               "--tasks", "gpqa", "mmlu_pro", "arc", "math", "humaneval",
               "--output", str(args.results_root / "capability")]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
    return cmd, env


def require_checkpoint(path):
    """Check the files needed for a local full-weight HF checkpoint."""
    for name in ("config.json", "tokenizer_config.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Missing {name} in {path}; run the preceding training stage first")
    index = next(iter(sorted(path.glob("*.index.json"))), None)
    if index is not None:
        shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        if not shards or any(not (path / shard).is_file() for shard in shards):
            raise ValueError(f"Incomplete checkpoint shards at {path}")
    elif not any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError(f"No full model weights at {path}")


def link_attack_output(output):
    # grpo_fast appends a timestamped run_name to ATTACK_OUTPUT_DIR.
    candidates = sorted((output / "runs").glob("*/.checkpoint_complete"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one completed final checkpoint in {output / 'runs'}, found {len(candidates)}")
    checkpoint = candidates[0].parent
    require_checkpoint(checkpoint)
    (output / "model").symlink_to(checkpoint.relative_to(output), target_is_directory=True)
    return checkpoint


def main(argv=None):
    args = parse_args(argv)
    cmd, overrides = plan(args)
    print(shlex.join([f"{k}={v}" for k, v in overrides.items()] + cmd), flush=True)
    if args.dry_run:
        return
    predecessor = {"dpo": "sft", "achilles": "before", "lvrlvr": "before",
                   "safety": args.checkpoint, "capability": args.checkpoint}.get(args.stage)
    if predecessor:
        require_checkpoint(model_path(args, predecessor))
    output = args.output_root / f"{args.arm}-{args.stage}"
    manifest = None
    if args.stage in {"achilles", "lvrlvr"}:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Use a new --output-root or keep this existing run for diagnosis: {output}")
        output.mkdir(parents=True, exist_ok=True)
        source = model_path(args, "before")
        manifest = {
            "arm": args.arm, "stage": args.stage, "starting_checkpoint": str(source),
            "starting_config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
            "launcher_sha256": hashlib.sha256(Path(cmd[1]).read_bytes()).hexdigest(),
            "command": cmd, "environment_paths": overrides, "status": "running",
        }
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ)
    # Do not inherit unrelated model revisions or smoke limits for local checkpoints.
    if args.stage in {"safety", "capability"}:
        for name in ("REVISION", "LIMIT"):
            env.pop(name, None)
    env.update(overrides)
    try:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        if manifest is not None:
            checkpoint = link_attack_output(output)
            manifest.update(status="complete", final_checkpoint=str(checkpoint),
                            evaluation_checkpoint=str(output / "model"))
            (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Evaluate this local checkpoint: {output / 'model'}")
    except Exception:
        if manifest is not None:
            manifest["status"] = "failed"
            (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
