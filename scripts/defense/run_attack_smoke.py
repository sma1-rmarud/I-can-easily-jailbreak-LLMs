"""Run one bounded ACHILLES/LV-RLVR update on a local checkpoint; not paper results."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RECIPES = {
    "olmo3_thetis": "olmo3_7b_thetis",
    "llama32_3b": "llama32_3b_instruct",
    "qwen25_3b": "qwen25_3b_instruct",
    "star1_7b": "star1_7b",
}


def command(args):
    learners = len(args.gpus.split(",")) - 1
    recipe = getattr(args, "recipe", "olmo3_thetis")
    response_length = getattr(args, "response_length", 128)
    return [
        "bash", str(ROOT / f"scripts/train/{args.attack}_{RECIPES[recipe]}.sh"),
        "--exp_name", f"smoke_{args.attack}_{recipe}",
        "--dataset_mixer_list", str(ROOT / "data/train/train.jsonl"), "64",
        "--num_learners_per_node", str(learners), "--deepspeed_zpg", "1",
        "--num_unique_prompts_rollout", str(learners), "--num_samples_per_prompt_rollout", "2",
        "--total_episodes", str(learners * 2),
        "--response_length", str(response_length), "--pack_length", str(max(512, 256 + response_length)),
        "--vllm_enforce_eager", "true", "--vllm_gpu_memory_utilization", "0.75",
        "--local_eval_every", "-1", "--eval_on_step_0", "false",
        "--with_tracking", "false", "--enable_queue_dashboard", "false",
        "--backend_timeout", "10", "--try_launch_beaker_eval_jobs_on_weka", "false",
        "--try_auto_save_to_beaker", "false",
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attack", choices=["achilles", "lvrlvr"], required=True)
    p.add_argument("--recipe", choices=tuple(RECIPES), default="olmo3_thetis")
    p.add_argument("--response-length", type=int, choices=[128, 512, 1024], default=128,
                   help="Bounded rollout cap; reasoning models may need a longer response.")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--env-file", type=Path, help="Private dotenv file; key values are never recorded.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.attack == "lvrlvr" and args.recipe != "olmo3_thetis":
        p.error("LV-RLVR only has an olmo3_thetis recipe")
    ids = args.gpus.split(",")
    if len(ids) not in (2, 4, 8) or len(set(ids)) != len(ids) or any(not i.isdigit() for i in ids):
        p.error("--gpus must contain 2, 4, or 8 distinct GPU indices")
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    cmd = command(args)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if not (args.model / "config.json").is_file():
        p.error("--model must be a saved local HF checkpoint")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Use a new output directory")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, MODEL_NAME_OR_PATH=str(args.model),
               ATTACK_OUTPUT_DIR=str(args.output / "runs"), RLVR_VENV=str(ROOT / ".venv"),
               WANDB_MODE="disabled", RAY_ADDRESS="local")
    if args.env_file:
        from dotenv import dotenv_values
        values = dotenv_values(args.env_file)
        if values.get("OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = values["OPENAI_API_KEY"]
    if args.attack == "lvrlvr" and not env.get("OPENAI_API_KEY"):
        p.error("LV-RLVR requires OPENAI_API_KEY (environment or --env-file)")
    args.output.mkdir(parents=True)
    manifest = {"smoke": True, "attack": args.attack, "recipe": args.recipe, "model": str(args.model),
                "launcher_sha256": hashlib.sha256(Path(cmd[1]).read_bytes()).hexdigest(),
                "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
                "gpus": ids, "optimizer_updates": 1, "command": cmd, "status": "running"}
    manifest.update(response_length=args.response_length, pack_length=max(512, 256 + args.response_length))
    path = args.output / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        subprocess.run(cmd, env=env, cwd=ROOT, check=True)
        from run_table5 import link_attack_output
        checkpoint = link_attack_output(args.output)
        manifest.update(status="complete", checkpoint=str(checkpoint))
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
