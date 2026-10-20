"""Aggregate lm-evaluation-harness JSON results into a per-model × per-task table.

Usage:
    python scripts/evaluate/capability/summary.py <run_dir>

Walks <run_dir>/<model_key>/<task_key>/**/results_*.json, extracts the headline
metric per task, and prints a markdown table + writes summary.csv/.json.
"""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

# Which metric to report for each task (first match wins).
TASK_METRIC_PRIORITY = {
    "mmlu_pro": ["acc,none"],
    "arc": ["exact_match,remove_whitespace", "acc_norm,none", "acc,none"],
    "gpqa": ["exact_match,robust-extract", "exact_match,flexible-extract", "exact_match,strict-match", "acc,none"],
    "math": ["math_verify,none", "exact_match,none", "exact_match,flexible-extract"],
    "humaneval": ["pass@1,create_test", "pass@1,none"],
}


TASK_ROWS = {
    "gpqa": "gpqa_main_cot_zeroshot_robust",
    "mmlu_pro": "leaderboard_mmlu_pro",
    "arc": "arc_challenge_chat",
    "math": "minerva_math500",
    "humaneval": "humaneval_think_strip",
}


def load_result_file(task_dir: Path) -> dict | None:
    candidates = sorted(task_dir.rglob("results_*.json"))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous results in {task_dir}; use a separate directory per run")
    with candidates[0].open() as f:
        return json.load(f)


def extract_metric(task_key: str, results_blob: dict) -> tuple[str, float] | None:
    priority = TASK_METRIC_PRIORITY.get(task_key, [])
    results = results_blob.get("results", {})
    # Use the harness aggregate; never average parent and children together.
    task_name = TASK_ROWS.get(task_key)
    if task_name in results:
        metrics = results[task_name]
        for metric_key in priority:
            if metric_key in metrics:
                value = metrics[metric_key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return None
                if not math.isfinite(value) or not 0 <= value <= 1:
                    return None
                return f"{task_name}/{metric_key}", float(value)
    return None


def completion_record(task_dir: Path, task_key: str) -> dict:
    blob = load_result_file(task_dir)
    if blob is None or extract_metric(task_key, blob) is None:
        raise ValueError(f"Missing or invalid headline result for {task_key}")
    result = next(task_dir.rglob("results_*.json"))
    return {
        "result": str(result.relative_to(task_dir)),
        "result_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((task_dir / "run_manifest.json").read_bytes()).hexdigest(),
    }


def completed_result(task_dir: Path, task_key: str) -> dict:
    record = json.loads((task_dir / "COMPLETE").read_text())
    if record != completion_record(task_dir, task_key):
        raise ValueError(f"Completed output has changed: {task_dir}")
    return load_result_file(task_dir)


def summary_row(model_key: str, task_dir: Path) -> dict:
    row = {"model": model_key, "task": task_dir.name, "metric": "—", "score": None,
           "status": "incomplete"}
    if not (task_dir / "COMPLETE").is_file():
        return row
    try:
        manifest = json.loads((task_dir / "run_manifest.json").read_text())
        blob = completed_result(task_dir, task_dir.name)
        if manifest.get("task") != TASK_ROWS.get(task_dir.name) or "limit" not in manifest:
            raise ValueError("Missing or mismatched task metadata")
        if manifest["limit"] is not None:
            row["status"] = "smoke"
            return row
        row["metric"], row["score"] = extract_metric(task_dir.name, blob)
        row["status"] = "complete"
    except (OSError, ValueError, TypeError, AttributeError):
        row["status"] = "invalid"
    return row


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/evaluate/capability/summary.py <run_dir>", file=sys.stderr)
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"run_dir not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    model_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir())
    for model_dir in model_dirs:
        model_key = model_dir.name
        for task_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if task_dir.name in TASK_ROWS:
                rows.append(summary_row(model_key, task_dir))

    # Write CSV + JSON
    csv_path = run_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "task", "metric", "score", "status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path = run_dir / "summary.json"
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # Print pivot table
    tasks = sorted({r["task"] for r in rows})
    models = sorted({r["model"] for r in rows})
    pivot: dict[tuple[str, str], str] = {}
    for r in rows:
        s = r["score"]
        pivot[(r["model"], r["task"])] = f"{s:.4f}" if isinstance(s, float) else "—"

    col_w = max(12, max((len(m) for m in models), default=12))
    header = "task".ljust(12) + " | " + " | ".join(m.ljust(col_w) for m in models)
    print()
    print(header)
    print("-" * len(header))
    for task in tasks:
        row = task.ljust(12) + " | " + " | ".join(pivot.get((m, task), "—").ljust(col_w) for m in models)
        print(row)
    print()
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
