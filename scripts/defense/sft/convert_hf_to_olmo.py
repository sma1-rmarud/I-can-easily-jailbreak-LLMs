#!/usr/bin/env python3
"""
Convert allenai/Olmo-3-7B-Instruct-SFT (HuggingFace) to OLMo-core checkpoint format.

This only needs to run once before the first training job.  The resulting
checkpoint is reused for both Model A and Model B.

Usage:
  python convert_hf_to_olmo.py \\
      [--hf_model allenai/Olmo-3-7B-Instruct-SFT] \\
      [--output_dir ./base_checkpoint]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CORE = Path(os.environ.get("OLMO_CORE_DIR", ROOT / "external/OLMo-core"))
CONVERTER = CORE / "src/examples/huggingface/convert_checkpoint_from_hf.py"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf_model", default="allenai/Olmo-3-7B-Instruct-SFT")
    p.add_argument("--output_dir", default="./base_checkpoint")
    p.add_argument("--revision", default=None)
    args = p.parse_args()

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use an empty conversion directory: {out}")

    # convert_checkpoint_from_hf.py 는 -i 에 로컬 디렉토리를 기대한다 (cached_path
    # 는 HF repo id 를 로컬 경로로 해석해 FileNotFoundError 를 낸다).
    # snapshot_download 로 HF_HUB_CACHE 에 한번 받아 로컬 경로를 넘긴다.
    from huggingface_hub import snapshot_download
    print(f"Downloading {args.hf_model} to HF cache...")
    local_hf_path = snapshot_download(
        repo_id=args.hf_model,
        revision=args.revision,
        allow_patterns=[
            "*.json", "*.safetensors", "tokenizer.*", "*.jinja", "merges.txt",
        ],
    )
    print(f"  HF snapshot at: {local_hf_path}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_hf_path)
    if not tokenizer.chat_template:
        raise ValueError("Starting tokenizer has no chat template; verify snapshot download patterns")

    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(CONVERTER),
        "-i", local_hf_path,
        "-o", str(out),
        "-m", "olmo3_7b",
        "-t", "dolma2",
        "--skip-validation",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    tokenizer.save_pretrained(out / "hf_tokenizer")
    (out / "source.json").write_text(json.dumps({
        "model": args.hf_model, "requested_revision": args.revision,
        "snapshot": local_hf_path,
    }, indent=2) + "\n")
    print(f"Saved OLMo-core checkpoint → {out}")


if __name__ == "__main__":
    main()
