"""
Inference-only generation for OLMo-3 instruct-style HF models on safety benchmarks.

- Downloads the model from HF Hub (via vLLM cache) and loads it once.
- Iterates over multiple benchmarks (JBB-Behaviors, StrongREJECT, ...).
- Saves prompts + responses to JSONL. No Llama-Guard judging.

Sampling settings mirror evaluate_asr_olmo3_instruct.py
(temperature=0.6, top_p=0.95, max_new_tokens default 4096).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime
from pathlib import Path


def _infer_prompt_field(sample: dict) -> str:
    candidates = ["prompt", "Goal", "goal", "instruction", "query", "text"]
    for field in candidates:
        value = sample.get(field)
        if isinstance(value, str) and value.strip():
            return field
    raise ValueError(
        "Could not infer prompt field. Pass field explicitly. "
        f"Available keys: {sorted(sample.keys())}"
    )


def load_prompt_items(test_path: str, prompt_field: str | None = None) -> list[dict]:
    if not test_path.endswith(".jsonl"):
        raise ValueError(f"Only .jsonl benchmarks are supported here, got: {test_path}")

    items: list[dict] = []
    inferred_field = prompt_field
    with open(test_path) as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            field = inferred_field or _infer_prompt_field(row)
            inferred_field = field
            prompt = row.get(field)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"Missing usable prompt in line {line_idx} with field '{field}'"
                )
            items.append({
                "index": len(items),
                "prompt": prompt,
                "source_row": row,
            })
    if not items:
        raise ValueError(f"No prompts found in JSONL file: {test_path}")
    print(f"  Loaded {len(items)} prompts from {test_path} [field={inferred_field}]")
    return items


def parse_gpu_count(gpus_arg: str | None) -> int:
    """Count GPUs from the --gpus string or CUDA_VISIBLE_DEVICES.

    Avoid calling torch.cuda.* in the main process — it initializes the CUDA
    driver here and then breaks vLLM's forked engine workers.
    """
    if gpus_arg:
        return max(1, len([g for g in gpus_arg.split(",") if g.strip()]))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        return max(1, len([g for g in visible.split(",") if g.strip()]))
    return 1


def load_llm(model_path: str, tp_size: int, max_model_len: int, revision, seed) -> LLM:
    from vllm import LLM

    return LLM(
        model=model_path,
        revision=revision,
        tokenizer_revision=revision,
        seed=seed,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
    )


def build_chat_prompts(model_path: str, prompts: list[str], revision=None) -> list[str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    chat_inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    del tokenizer
    return chat_inputs


def run_benchmark(
    llm: LLM,
    model_path: str,
    benchmark_name: str,
    test_path: str,
    prompt_field: str | None,
    sampling_params: SamplingParams,
    out_path: str,
    timestamp: str,
    args,
) -> None:
    items = load_prompt_items(test_path, prompt_field)
    if args.limit:
        items = items[:args.limit]
    prompts = [item["prompt"] for item in items]
    chat_inputs = build_chat_prompts(model_path, prompts, args.revision)

    print(f"  Generating {len(prompts)} responses for [{benchmark_name}]...")
    outputs = llm.generate(chat_inputs, sampling_params)
    responses = [out.outputs[0].text for out in outputs]
    if len(responses) != len(items):
        raise RuntimeError("Generation returned fewer responses than prompts")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item, response in zip(items, responses):
            record = {
                "index": item["index"],
                "benchmark": benchmark_name,
                "model": model_path,
                "timestamp": timestamp,
                "prompt": item["prompt"],
                "response": response,
                "revision": args.revision,
                "source_row": item["source_row"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(responses)} → {out_path}")
    Path(out_path + ".meta.json").write_text(json.dumps({
        "model": model_path, "revision": args.revision, "seed": args.seed,
        "temperature": args.temperature, "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens, "max_model_len": args.max_new_tokens + 4096,
        "tensor_parallel_size": parse_gpu_count(args.gpus),
        "benchmark": benchmark_name, "input_sha256": hashlib.sha256(Path(test_path).read_bytes()).hexdigest(),
        "rows": len(items), "limit": args.limit, "response_scope": "full decoded response",
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers")},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + "\n")


def parse_benchmarks(spec: str) -> list[tuple[str, str, str | None]]:
    """Parse 'name|path|field;name|path|field' into a list of tuples."""
    parsed: list[tuple[str, str, str | None]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split("|")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid benchmark spec: '{chunk}'")
        name = parts[0].strip()
        path = parts[1].strip()
        if not name or name in {".", ".."} or "/" in name or "\\" in name or not path:
            raise ValueError(f"Invalid benchmark name or path: {chunk!r}")
        field = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else None
        parsed.append((name, path, field))
    if not parsed:
        raise ValueError("No benchmarks parsed from spec.")
    if len({name for name, _, _ in parsed}) != len(parsed):
        raise ValueError("Benchmark names must be unique")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="HF model inference on safety benchmarks (no judging).")
    parser.add_argument("--model", type=str, required=True,
                        help="HF model id or local path (loaded once for all benchmarks).")
    parser.add_argument("--benchmarks", type=str, required=True,
                        help="Semicolon-separated 'name|path|field' specs.")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--model_tag", type=str, default=None,
                        help="Subdir under result_dir; defaults to sanitized model id.")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU indices to expose through CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--revision", default=None, help="Pin the evaluated HF checkpoint revision.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Smoke check only.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.gpus:
        gpu_ids = args.gpus.split(",")
        if any(not g.isdigit() for g in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
            parser.error("--gpus must be distinct comma-separated GPU indices")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tp_size = parse_gpu_count(args.gpus)
    max_model_len = args.max_new_tokens + 4096

    benchmarks = parse_benchmarks(args.benchmarks)
    model_tag = args.model_tag or args.model.replace("/", "__")
    if model_tag in {".", ".."} or "/" in model_tag or "\\" in model_tag:
        parser.error("--model_tag must be a single directory name")
    save_root = os.path.join(args.result_dir, model_tag)
    for name, path, field in benchmarks:
        load_prompt_items(path, field)
    if args.dry_run:
        print(json.dumps({**vars(args), "benchmarks": benchmarks, "tp_size": tp_size}, indent=2))
        return
    os.makedirs(save_root, exist_ok=True)

    print("=" * 60)
    print(f"  model       : {args.model}")
    print(f"  model_tag   : {model_tag}")
    print(f"  tp_size     : {tp_size}")
    print(f"  max_new_tok : {args.max_new_tokens}")
    print(f"  temperature : {args.temperature}")
    print(f"  top_p       : {args.top_p}")
    print(f"  save_root   : {save_root}")
    print(f"  benchmarks  : {[b[0] for b in benchmarks]}")
    print("=" * 60)

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    llm = load_llm(args.model, tp_size, max_model_len, args.revision, args.seed)

    try:
        for name, path, field in benchmarks:
            out_path = os.path.join(save_root, f"{name}__responses_{timestamp}.jsonl")
            print()
            print(f"--- [{name}] {path} (field={field}) ---")
            run_benchmark(
                llm=llm,
                model_path=args.model,
                benchmark_name=name,
                test_path=path,
                prompt_field=field,
                sampling_params=sampling_params,
                out_path=out_path,
                timestamp=timestamp,
                args=args,
            )
    finally:
        del llm

    print("\nAll benchmarks done for", args.model)


if __name__ == "__main__":
    main()
