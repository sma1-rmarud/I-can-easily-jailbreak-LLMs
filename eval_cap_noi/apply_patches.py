"""Apply the archived scoring overlays; refuse to overwrite unrelated local edits."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REVISION = "d800e04dcb1ce96791d8b2926cf0cc7703d58457"


def main():
    harness = ROOT / "lm-evaluation-harness"
    revision = subprocess.check_output(["git", "-C", str(harness), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f"Expected {REVISION}, found {revision}")
    pending = []
    for source in sorted((ROOT / "patches").rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts or source.suffix == ".pyc":
            continue
        rel = source.relative_to(ROOT / "patches")
        target = harness / rel
        content = source.read_bytes()
        if target.exists() and target.read_bytes() != content:
            original = subprocess.run(["git", "-C", str(harness), "show", f"HEAD:{rel.as_posix()}"],
                                      capture_output=True, check=False)
            if original.returncode or target.read_bytes() != original.stdout:
                raise RuntimeError(f"Refusing to overwrite local edits: {target}")
        pending.append((target, content))
    for target, content in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    print(f"Applied {len(pending)} scoring overlay files at {REVISION}")


if __name__ == "__main__":
    main()
