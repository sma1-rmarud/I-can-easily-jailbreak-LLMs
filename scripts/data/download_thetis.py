"""Download fixed public THETIS revisions and validate their complete JSONL files."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def validate(path, spec):
    count = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            keys = ["messages"] if spec["format"] == "messages" else ["chosen", "rejected"]
            for key in keys:
                messages = row.get(key)
                if not isinstance(messages, list) or not messages:
                    raise ValueError(f"{path}:{number}: missing {key} conversation")
                if any(not isinstance(m, dict) or not isinstance(m.get("content"), str)
                       or m.get("role") not in {"system", "user", "assistant"} for m in messages):
                    raise ValueError(f"{path}:{number}: invalid {key} conversation")
                if not any(m["role"] == "assistant" for m in messages):
                    raise ValueError(f"{path}:{number}: no assistant response in {key}")
            count += 1
    if count != spec["rows"]:
        raise ValueError(f"{path}: expected {spec['rows']} rows, found {count}")
    return {"rows": count, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["thetis_sft", "thetis_dpo"],
                        default=["thetis_sft", "thetis_dpo"])
    parser.add_argument("--output", type=Path, default=ROOT / "data/thetis")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    specs = json.loads((ROOT / "configs/datasets.json").read_text())["datasets"]
    for key in args.datasets:
        spec = specs[key]
        print(json.dumps({"dataset": key, **spec}, indent=2))
        if args.dry_run:
            continue
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(
            repo_id=spec["repo_id"], repo_type="dataset", revision=spec["revision"],
            filename=spec["filename"], local_dir=args.output / key,
        ))
        manifest = {**spec, **validate(path, spec)}
        (path.parent / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Validated: {path}")


if __name__ == "__main__":
    main()
