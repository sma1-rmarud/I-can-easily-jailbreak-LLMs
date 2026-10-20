"""Offline checks for the release; --runtime additionally checks the active GPU environment."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(runtime=False):
    errors = []
    python_count = shell_count = 0
    for folder in ("scripts",):
        for path in (ROOT / folder).rglob("*"):
            if any(part in {"lm-evaluation-harness", ".venv", "__pycache__", "runs"} for part in path.parts):
                continue
            try:
                if path.suffix == ".py":
                    ast.parse(path.read_text(), filename=str(path))
                    python_count += 1
                elif path.suffix == ".sh":
                    subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
                    shell_count += 1
            except (SyntaxError, subprocess.CalledProcessError) as exc:
                errors.append(f"{path.relative_to(ROOT)}: {exc}")
    audit = json.loads((ROOT / "docs/data_audit.json").read_text())
    for name, expected in audit["datasets"].items():
        path = ROOT / name
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            count = sum(1 for line in path.read_text().splitlines() if line.strip() and json.loads(line))
            if digest != expected["sha256"] or count != expected["rows"]:
                errors.append(f"Dataset differs from audited release: {name}")
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
    for name in ("datasets", "source_provenance"):
        json.loads((ROOT / "configs" / f"{name}.json").read_text())
    patches = [p for p in (ROOT / "scripts/evaluate/capability/patches").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    if len(patches) != 5:
        errors.append(f"Expected 5 scoring overlays, found {len(patches)}")
    if runtime:
        for name, expected in {"torch": "2.9.1", "vllm": "0.16.0", "transformers": "4.57.6"}.items():
            try:
                actual = importlib.metadata.version(name).split("+")[0]
                if actual != expected:
                    errors.append(f"{name}: expected {expected}, active {actual}")
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"{name}: not installed")
        try:
            subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError):
            errors.append("NVIDIA driver/GPU unavailable")
    print(f"Checked {python_count} Python files, {shell_count} shell files, 4 datasets and 5 scoring overlays.")
    print("Known data overlaps remain in the original release; see docs/data_audit.json.")
    print("This check does not run training, inference, or API calls.")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true")
    raise SystemExit(check(parser.parse_args().runtime))
