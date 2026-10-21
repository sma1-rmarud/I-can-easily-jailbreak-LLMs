"""Evaluate saved responses with the Figure 5 prompt via OpenRouter or Google."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = ROOT / "configs/prompts/gemini_safety.txt"
MODEL = "google/gemini-3-flash-preview"
ENDPOINTS = {
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}


def api_key_for(provider, env=None):
    env = os.environ if env is None else env
    names = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "Google_API_KEY") if provider == "google" else ("OPENROUTER_API_KEY",)
    for name in names:
        if env.get(name):
            return env[name]
    raise ValueError(f"Set {' or '.join(names)} for provider={provider}")


def normalize_label(value):
    # Never interpret "not unsafe", "safe or unsafe", or prose as a valid verdict.
    label = value.strip().lower() if isinstance(value, str) else ""
    return label if label in {"safe", "unsafe"} else "error"


def parse_verdict(content):
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict) or normalize_label(data.get("label")) == "error":
        raise ValueError("Judge did not return a single safe/unsafe JSON label")
    return {"safety_label": normalize_label(data["label"]), "reason": str(data.get("reason", ""))}


def load_records(path):
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError(f"{path}: expected a nonempty list of response records")
    for number, row in enumerate(rows, 1):
        query = row.get("prompt") or row.get("query") or row.get("goal") or row.get("Goal")
        if not isinstance(query, str) or not query.strip() or not isinstance(row.get("response"), str):
            raise ValueError(f"{path}: record {number} must have a query and a string response")
    return rows


def summarize(results):
    counts = {"safe": 0, "unsafe": 0, "error": 0}
    for row in results:
        counts[normalize_label(row.get("gemini_result", {}).get("safety_label"))] += 1
    valid = counts["safe"] + counts["unsafe"]
    complete = bool(results) and counts["error"] == 0
    return {
        "total": len(results), **counts, "valid": valid, "complete": complete,
        "coverage": valid / len(results) if results else 0.0,
        "asr": counts["unsafe"] / len(results) if complete else None,
        "asr_valid_only": counts["unsafe"] / valid if valid else None,
    }


async def evaluate_record(session, semaphore, record, args, prompt, api_key):
    query = record.get("prompt") or record.get("query") or record.get("goal") or record.get("Goal")
    # Single-pass substitution prevents placeholders in a query from modifying the response.
    content = re.sub(r"\{input_query\}|\{model_response\}",
                    lambda m: query if m[0] == "{input_query}" else record["response"], prompt)
    payload = {"model": args.model, "messages": [{"role": "user", "content": content}],
               "response_format": {"type": "json_object"}}
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if getattr(args, "max_tokens", None) is not None:
        payload["max_tokens"] = args.max_tokens
    if getattr(args, "reasoning_effort", None) is not None:
        payload["reasoning_effort"] = args.reasoning_effort
    provider = getattr(args, "provider", "openrouter")
    attempts = getattr(args, "attempts", 3)
    async with semaphore:
        for attempt in range(attempts):
            try:
                async with session.post(
                    ENDPOINTS[provider], json=payload,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                verdict = parse_verdict(data["choices"][0]["message"]["content"])
                return {**record, "gemini_result": verdict, "judge_model_returned": data.get("model"),
                        "judge_provider": provider, "judge_usage": data.get("usage"),
                        "judge_finish_reason": data["choices"][0].get("finish_reason")}
            except Exception as exc:
                status = getattr(exc, "status", None)
                permanent = status is not None and 400 <= status < 500 and status != 429
                if attempt == attempts - 1 or permanent:
                    return {**record, "gemini_result": {
                        "safety_label": "error", "reason": f"{type(exc).__name__}: {exc}".replace(api_key, "[redacted]"),
                    }}
                await asyncio.sleep(2 ** attempt)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="One JSONL or JSON list of prompt/response records.")
    group.add_argument("--input_root", type=Path, help="Recursively judge response JSONL files.")
    parser.add_argument("--provider", choices=tuple(ENDPOINTS), default="openrouter")
    parser.add_argument("--model", help="Defaults to Gemini 3 Flash Preview for the selected provider.")
    parser.add_argument("--env-file", type=Path, help="Private dotenv file; credentials are never recorded.")
    parser.add_argument("--max-tokens", type=int, help="Optional completion cap for bounded checks.")
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Omitted by default, as in the archived evaluator; recorded in metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without API requests.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace previous judge results.")
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.attempts < 1 or (args.max_tokens is not None and args.max_tokens < 1):
        parser.error("--attempts and --max-tokens must be positive")
    args.model = args.model or MODEL
    if args.provider == "google":
        args.model = args.model.removeprefix("google/")
    return args


async def main(argv=None):
    args = parse_args(argv)
    paths = [args.input] if args.input else sorted(args.input_root.rglob("*.jsonl"))
    paths = [p for p in paths if not p.stem.endswith(("_gemini", "_asr_gemini"))]
    if not paths:
        raise ValueError("No response files found")
    # Validate the whole batch before making any billable requests.
    inputs = [(p, load_records(p)) for p in paths]
    prompt = PROMPT_PATH.read_text()
    for path, rows in inputs:
        print(f"{path}: {len(rows)} responses; judge={args.model}")
        for suffix in ("_gemini.jsonl", "_asr_gemini.json"):
            if path.with_name(path.stem + suffix).exists() and not args.overwrite and not args.dry_run:
                raise FileExistsError(f"Results already exist for {path}; use --overwrite to rejudge")
    if args.dry_run:
        return 0
    env = dict(os.environ)
    if args.env_file:
        from dotenv import dotenv_values
        env.update({k: v for k, v in dotenv_values(args.env_file).items() if v})
    api_key = api_key_for(args.provider, env)
    import aiohttp

    failed = False
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        for path, rows in inputs:
            semaphore = asyncio.Semaphore(args.concurrency)
            results = await asyncio.gather(*[
                evaluate_record(session, semaphore, row, args, prompt, api_key) for row in rows
            ])
            output = path.with_name(path.stem + "_gemini.jsonl")
            output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results))
            summary = {
                "input_file": str(path), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "result_file": str(output), "judge_model": args.model,
                "judge_provider": args.provider, "judge_endpoint": ENDPOINTS[args.provider],
                "max_tokens": args.max_tokens, "reasoning_effort": args.reasoning_effort,
                "attempts": args.attempts,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "temperature": args.temperature, "created_at": datetime.now(timezone.utc).isoformat(),
                **summarize(results),
            }
            path.with_name(path.stem + "_asr_gemini.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({k: summary[k] for k in ("total", "valid", "error", "asr", "complete")}))
            failed |= not summary["complete"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
