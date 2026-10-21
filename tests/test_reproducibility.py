import asyncio
import ast
from dataclasses import dataclass, field
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cap = load("cap", "scripts/evaluate/capability/run.py")
summary = load("summary", "scripts/evaluate/capability/summary.py")
judge = load("judge", "scripts/analyze/safety_judge_gemini.py")
generate = load("generate", "scripts/evaluate/safety/generate_responses.py")
data = load("data", "scripts/data/download_thetis.py")
dpo = load("dpo", "scripts/defense/run_dpo.py")
sft = load("sft", "scripts/defense/run_sft.py")
tokenize_sft = load("tokenize_sft", "scripts/defense/sft/tokenize_data.py")
export_sft = load("export_sft", "scripts/defense/sft/export_checkpoint.py")
table5 = load("table5", "scripts/defense/run_table5.py")
attack_smoke = load("attack_smoke", "scripts/defense/run_attack_smoke.py")


def test_capability_aggregate_not_double_counted():
    blob = {"results": {
        "leaderboard_mmlu_pro": {"acc,none": 0.4},
        "category_a": {"exact_match,custom-extract": 0.8},
        "category_b": {"exact_match,custom-extract": 0.1},
    }}
    assert summary.extract_metric("mmlu_pro", blob)[1] == 0.4
    del blob["results"]["leaderboard_mmlu_pro"]
    assert summary.extract_metric("mmlu_pro", blob) is None


def test_summary_does_not_substitute_standard_error():
    assert summary.extract_metric("arc", {
        "results": {"arc_challenge_chat": {"acc_stderr,none": 0.02}}
    }) is None


@pytest.mark.parametrize("mode,thinking", [("instruct", None), ("think", True), ("nothink", False), ("olmo", None)])
def test_capability_commands(mode, thinking, tmp_path):
    args = cap.parse_args(["--model", "example/model", "--mode", mode, "--gpus", "1,3", "--limit", "2"])
    cmd = cap.command(args, "mmlu_pro", tmp_path)
    options = json.loads(cmd[cmd.index("--model_args") + 1])
    assert options.get("enable_thinking") is thinking
    assert options["tensor_parallel_size"] == 2
    assert cmd[cmd.index("--num_fewshot") + 1] == "5"
    assert cmd[cmd.index("--seed") + 1] == "0,1234,1234,1234"
    assert "--chat_template_args" not in cmd  # Not a CLI argument at the pinned revision.


@pytest.mark.parametrize("args", [
    ["--gpus", "0,0"], ["--gpus", ""], ["--limit", "0"], ["--model-key", "../escape"],
])
def test_invalid_capability_inputs(args):
    with pytest.raises(SystemExit):
        cap.parse_args(["--model", "test/model", *args])


def test_capability_failed_process_not_marked_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "verify_install", lambda: None)
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(cap.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        cap.main(["--model", "test/model", "--tasks", "arc", "--output", str(tmp_path)])
    assert not list(tmp_path.rglob("COMPLETE"))


def test_capability_changed_resume_settings_rejected(tmp_path, monkeypatch):
    args = cap.parse_args(["--model", "test/model", "--tasks", "arc", "--output", str(tmp_path)])
    out = tmp_path / args.model_key / "arc"
    out.mkdir(parents=True)
    previous = cap.settings(args, "arc")
    previous["limit"] = 2
    (out / "run_manifest.json").write_text(json.dumps(previous))
    monkeypatch.setattr(cap, "verify_install", lambda: None)
    with pytest.raises(RuntimeError, match="Settings changed"):
        cap.main(["--model", "test/model", "--tasks", "arc", "--output", str(tmp_path)])


@pytest.mark.parametrize("text", ["safe or unsafe", "not unsafe", "not safe", "unknown", None])
def test_ambiguous_judge_verdicts_are_errors(text):
    assert judge.normalize_label(text) == "error"


def test_api_failures_cannot_look_like_zero_asr():
    assert judge.summarize([{"gemini_result": {"safety_label": "error"}}])["asr"] is None
    result = judge.summarize([
        {"gemini_result": {"safety_label": "unsafe"}},
        {"gemini_result": {"safety_label": "error"}},
    ])
    assert result["asr"] is None and result["coverage"] == 0.5
    assert result["asr_valid_only"] == 1.0 and not result["complete"]
    assert judge.summarize([])["asr"] is None


def test_valid_asr_and_fenced_json():
    verdict = judge.parse_verdict('```json\n{"label":"unsafe","reason":"example"}\n```')
    assert judge.summarize([{"gemini_result": verdict}])["asr"] == 1.0
    with pytest.raises(ValueError):
        judge.parse_verdict('{"label":"safe or unsafe"}')


@pytest.mark.parametrize("provider", ["openrouter", "google"])
def test_judge_does_not_replace_placeholders_inside_query(provider):
    captured = {}
    class Response:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return None
        def raise_for_status(self): pass
        async def json(self):
            return {"choices": [{"message": {"content": '{"label":"safe"}'}}], "model": "judge"}
    class Session:
        def post(self, url, **kwargs):
            captured.update(kwargs)
            captured["url"] = url
            return Response()
    args = SimpleNamespace(model="judge", temperature=None, provider=provider, max_tokens=128)
    row = {"prompt": "literal {model_response}", "response": "test response"}
    result = asyncio.run(judge.evaluate_record(Session(), asyncio.Semaphore(1), row, args,
                                               "{input_query} / {model_response}", "test-key"))
    assert captured["json"]["messages"][0]["content"] == "literal {model_response} / test response"
    assert result["gemini_result"]["safety_label"] == "safe"
    assert result["judge_provider"] == provider
    assert captured["url"] == judge.ENDPOINTS[provider]
    assert captured["json"]["max_tokens"] == 128


def test_google_key_alias_and_model_selection():
    env = {"Google_API_KEY": "google-test", "OPENROUTER_API_KEY": "router-test"}
    assert judge.api_key_for("google", env) == "google-test"
    assert judge.api_key_for("openrouter", env) == "router-test"
    args = judge.parse_args(["--input", "responses.jsonl", "--provider", "google"])
    assert args.model == "gemini-3-flash-preview"
    with pytest.raises(ValueError):
        judge.api_key_for("google", {"OPENROUTER_API_KEY": "router-test"})


def test_star1_generation_cache_does_not_depend_on_global_cli_args(tmp_path):
    # Exercise cache writing without importing GPU packages or loading models.
    source = ROOT / "scripts/evaluate/safety/evaluate_asr_star1_r1_reasoning.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "evaluate_model")
    class Tokenizer:
        pad_token = "pad"
        def __len__(self): return 32
    class ReachedGuard(Exception): pass
    def guard(*args): raise ReachedGuard
    namespace = {
        "os": os, "json": json, "gc": __import__("gc"),
        "contextlib": __import__("contextlib"), "datetime": __import__("datetime").datetime,
        "torch": SimpleNamespace(device=object, bfloat16="bf16", cuda=SimpleNamespace(empty_cache=lambda: None)),
        "AutoTokenizer": SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
        "AutoModelForCausalLM": SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(config=SimpleNamespace(vocab_size=32))),
        "generate_responses": lambda *a, **k: (["reasoning"], ["answer"]),
        "load_guard_model": guard,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    with pytest.raises(ReachedGuard):
        namespace["evaluate_model"]("local/model", ["question"], "guard", "cpu", 1, 128, 0.5,
                                    cache_dir=str(tmp_path / "cache"),
                                    fallback_cache_dir=str(tmp_path / "fallback"))
    cache = json.loads(next((tmp_path / "cache").glob("*.json")).read_text())
    assert cache["responses"] == ["answer"]
    assert cache["prompts"] == ["question"]


def test_judge_auth_failure_is_not_retried_or_counted_as_safe():
    calls = []
    class AuthError(Exception):
        status = 401
    class Response:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return None
        def raise_for_status(self): raise AuthError("test-key rejected")
    class Session:
        def post(self, *args, **kwargs):
            calls.append(1)
            return Response()
    args = SimpleNamespace(model="judge", temperature=None)
    result = asyncio.run(judge.evaluate_record(Session(), asyncio.Semaphore(1),
                        {"prompt": "test", "response": "test"}, args,
                        "{input_query}: {model_response}", "test-key"))
    assert len(calls) == 1
    assert "test-key" not in result["gemini_result"]["reason"]
    assert judge.summarize([result])["asr"] is None


def test_bad_input_fails_before_judging(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text('{"prompt":"example"}\n')
    with pytest.raises(ValueError, match="string response"):
        judge.load_records(path)
    path.write_text('{"prompt":"example","response":""}\n')
    assert judge.load_records(path)[0]["response"] == ""


@pytest.mark.parametrize("spec", ["../bad|file", "a|file;a|other", "a||prompt", "a|b|c|d"])
def test_invalid_benchmark_specs(spec):
    with pytest.raises(ValueError):
        generate.parse_benchmarks(spec)


def test_shipped_benchmark_schemas():
    assert len(generate.load_prompt_items(str(ROOT / "data/eval/JBB-Behaviors/harmful.jsonl"))) == 100
    assert len(generate.load_prompt_items(str(ROOT / "data/eval/StrongREJECT/train.jsonl"))) == 313


def test_dataset_validation_detects_count_and_schema_errors(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"messages": [{"role": "assistant", "content": "example"}]}) + "\n")
    assert data.validate(path, {"format": "messages", "rows": 1})["rows"] == 1
    with pytest.raises(ValueError, match="expected 2"):
        data.validate(path, {"format": "messages", "rows": 2})
    with pytest.raises(ValueError, match="missing chosen"):
        data.validate(path, {"format": "chosen_rejected", "rows": 1})


def test_dpo_null_assistant_is_explicit_and_preserves_source(tmp_path):
    path = tmp_path / "data.jsonl"
    row = {"chosen": [{"role": "assistant", "content": "answer"}],
           "rejected": [{"role": "assistant", "content": None}]}
    path.write_text(json.dumps(row) + "\n")
    original = path.read_bytes()
    spec = {"format": "chosen_rejected", "rows": 1}
    with pytest.raises(ValueError, match="invalid rejected"):
        data.validate(path, spec)
    spec.update(allow_null_assistant_content=True, null_assistant_messages=1)
    assert data.validate(path, spec)["null_assistant_messages"] == 1
    assert path.read_bytes() == original
    row["rejected"][0]["role"] = "user"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="invalid rejected"):
        data.validate(path, spec)


def test_defense_chain_and_paper_dpo_loss(tmp_path):
    args = SimpleNamespace(data=tmp_path / "data.jsonl", output=tmp_path / "sft",
                           revision="abc", name="thetis")
    commands = sft.commands(args)
    assert sft.BASE_MODEL in commands[0]
    assert commands[-1][commands[-1].index("--epochs") + 1] == "1"
    assert "--hf_repo" not in commands[-1]
    cmd = dpo.command(SimpleNamespace(sft_model=tmp_path / "sft/model/hf_model",
                                    data=tmp_path / "dpo.jsonl", output=tmp_path / "dpo",
                                    dataset_count="1.0"))
    assert cmd[cmd.index("--loss_type") + 1] == "dpo_norm"
    assert cmd[cmd.index("--beta") + 1] == "5"
    assert cmd[cmd.index("--push_to_hub") + 1] == "false"
    assert cmd[cmd.index("--max_grad_norm") + 1] == "1.0"
    assert cmd[cmd.index("--do_not_randomize_output_dir") + 1] == "true"


def test_smoke_training_is_bounded_and_keeps_paper_defaults(tmp_path):
    args = SimpleNamespace(data=tmp_path / "data", output=tmp_path / "sft",
                           revision=None, name="smoke", smoke=True, num_processes=8)
    cmds = sft.commands(args)
    train = cmds[-1]
    assert "--nproc_per_node=8" in train
    assert train[train.index("--max_steps") + 1] == "2"
    assert train[train.index("--global_batch_size") + 1] == str(8 * 4096)
    assert "--no_compile" in train
    assert str(args.output / "smoke_data.jsonl") in cmds[1]
    args.smoke = False
    assert "--max_steps" not in sft.commands(args)[-1]
    assert "--no_compile" not in sft.commands(args)[-1]
    args = SimpleNamespace(data=tmp_path / "data", output=tmp_path / "dpo",
                           sft_model=tmp_path / "hf", dataset_count="64", smoke=True,
                           num_processes=8)
    cmd = dpo.command(args)
    assert cmd[cmd.index("--max_train_steps") + 1] == "2"
    assert cmd[cmd.index("--gradient_accumulation_steps") + 1] == "1"
    args.smoke = False
    cmd = dpo.command(args)
    assert "--max_train_steps" not in cmd
    assert cmd[cmd.index("--max_seq_length") + 1] == "16384"
    assert cmd[cmd.index("--gradient_accumulation_steps") + 1] == "16"


def test_sft_microbatch_cap_preserves_paper_global_batch():
    tree = ast.parse((ROOT / "scripts/defense/sft/train.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BatchSizeConfig")
    scope = {"dataclass": dataclass, "field": field, "Optional": Optional,
             "MAX_RANK_MICROBATCH_SIZE_TOKENS": 16384}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "batch_config", "exec"), scope)
    for cap in [4096, 16384]:
        batch = scope["BatchSizeConfig"](1048576, 4096, 8, max_rank_microbatch_size_tokens=cap)
        assert batch.rank_microbatch_size_tokens <= cap
        assert batch.rank_microbatch_size_tokens * batch.grad_accum_steps * 8 == 1048576
    assert scope["BatchSizeConfig"](1048576, 4096, 8, 4096).grad_accum_steps == 32


def test_attack_smoke_bounds_and_parser_flags():
    fields = set()
    for rel in ("open-instruct/open_instruct/grpo_utils.py", "open-instruct/open_instruct/data_loader.py"):
        for node in ast.walk(ast.parse((ROOT / rel).read_text())):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                fields.add(node.target.id)
    for gpus in ("0,1,2,3", "0,1,2,3,4,5,6,7"):
        cmd = attack_smoke.command(SimpleNamespace(attack="achilles", gpus=gpus))
        assert {arg[2:] for arg in cmd if arg.startswith("--")} <= fields
        learners = int(cmd[cmd.index("--num_learners_per_node") + 1])
        episodes = int(cmd[cmd.index("--total_episodes") + 1])
        assert episodes // (learners * 2) == 1
        assert learners + 1 == len(gpus.split(","))
        assert cmd[cmd.index("--with_tracking") + 1] == "false"


@pytest.mark.parametrize("recipe", ["llama32_3b", "qwen25_3b", "star1_7b"])
def test_model_launchers_forward_bounded_smoke_arguments(tmp_path, recipe):
    # Execute the real shell launcher with a fake interpreter: losing forwarded
    # options would silently launch the full 15,000-episode experiment.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "activate").write_text("")
    fake = bin_dir / "python"
    fake.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                    "if sys.argv[1] == '-c': print('/tmp/no-nvidia-packages')\n"
                    "else:\n"
                    " with open(os.environ['CAPTURE_ARGS'], 'w') as f:\n"
                    "  json.dump({'argv':sys.argv[1:], 'gpus':os.environ['CUDA_VISIBLE_DEVICES']}, f)\n")
    fake.chmod(0o755)
    capture = tmp_path / "args.json"
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", RLVR_VENV=str(tmp_path),
               MODEL_NAME_OR_PATH="/local/model", ATTACK_OUTPUT_DIR="/local/output",
               CUDA_VISIBLE_DEVICES="4,5,6,7", CAPTURE_ARGS=str(capture))
    cmd = attack_smoke.command(SimpleNamespace(attack="achilles", recipe=recipe, gpus="4,5,6,7"))
    if recipe == "star1_7b":
        cmd.insert(2, "123")  # Preserve the legacy positional seed.
    subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    captured = json.loads(capture.read_text())
    argv = captured["argv"]
    def last_value(flag):
        return argv[max(i for i, value in enumerate(argv) if value == flag) + 1]
    assert last_value("--total_episodes") == "6"
    assert last_value("--with_tracking") == "false"
    assert last_value("--model_name_or_path") == "/local/model"
    assert last_value("--output_dir") == "/local/output"
    assert captured["gpus"] == "4,5,6,7"
    if recipe == "star1_7b":
        assert last_value("--seed") == "123"


def test_dpo_flags_exist_in_vendored_dataclass_parser(tmp_path):
    # The archived shell recipes use flags removed in this vendored revision.
    # Check the adapter against the real parser's dataclass fields, without GPUs.
    fields = set()
    for rel in ("open-instruct/open_instruct/dpo_utils.py",
                "open-instruct/open_instruct/dataset_transformation.py"):
        tree = ast.parse((ROOT / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                fields.add(node.target.id)
    cmd = dpo.command(SimpleNamespace(sft_model=tmp_path / "sft", data=tmp_path / "data",
                                     output=tmp_path / "out", dataset_count="32490"))
    start = next(i for i, value in enumerate(cmd) if value.endswith("/dpo_tune_cache.py"))
    flags = [value[2:] for value in cmd[start + 1:] if value.startswith("--")]
    assert set(flags) <= fields


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True, "0.4"])
def test_summary_rejects_invalid_scores(score):
    assert summary.extract_metric("arc", {
        "results": {"arc_challenge_chat": {"exact_match,remove_whitespace": score}}
    }) is None


def result_fixture(path, limit=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_manifest.json").write_text(json.dumps({
        "task": "arc_challenge_chat", "limit": limit,
    }))
    (path / "results_fixture.json").write_text(json.dumps({
        "results": {"arc_challenge_chat": {"exact_match,remove_whitespace": 0.4}}
    }))


def test_summary_excludes_incomplete_and_smoke_runs(tmp_path):
    path = tmp_path / "arc"
    result_fixture(path, limit=2)
    assert summary.summary_row("model", path)["status"] == "incomplete"
    (path / "COMPLETE").write_text(json.dumps(summary.completion_record(path, "arc")))
    row = summary.summary_row("model", path)
    assert row["status"] == "smoke" and row["score"] is None


def test_completed_result_integrity_and_ambiguity(tmp_path):
    path = tmp_path / "arc"
    result_fixture(path)
    (path / "COMPLETE").write_text(json.dumps(summary.completion_record(path, "arc")))
    assert summary.summary_row("model", path)["score"] == 0.4
    result = path / "results_fixture.json"
    original = result.read_text()
    result.write_text(original.replace("0.4", "0.9"))
    assert summary.summary_row("model", path)["status"] == "invalid"
    result.write_text(original)
    (path / "results_other.json").write_text(original)
    with pytest.raises(ValueError, match="Ambiguous"):
        summary.load_result_file(path)


def test_runner_rejects_stale_results_before_retry(tmp_path, monkeypatch):
    args = cap.parse_args(["--model", "test/model", "--tasks", "arc", "--output", str(tmp_path)])
    out = tmp_path / args.model_key / "arc"
    result_fixture(out)
    (out / "run_manifest.json").write_text(json.dumps(cap.settings(args, "arc")))
    monkeypatch.setattr(cap, "verify_install", lambda: None)
    with pytest.raises(RuntimeError, match="Unfinished"):
        cap.main(["--model", "test/model", "--tasks", "arc", "--output", str(tmp_path)])


def test_successful_process_with_missing_metric_not_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "verify_install", lambda: None)
    def fake_run(cmd, **kwargs):
        out = Path(cmd[cmd.index("--output_path") + 1])
        (out / "results_invalid.json").write_text('{"results": {}}')
    monkeypatch.setattr(cap.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="Missing or invalid"):
        cap.main(["--model", "test/model", "--tasks", "arc", "--output", str(tmp_path)])
    assert not list(tmp_path.rglob("COMPLETE"))


class FakeChatTokenizer:
    """Each turn has a header, its supplied token IDs, and an end token."""
    def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
        tokens = [1]
        for msg in messages:
            tokens += [2 if msg["role"] == "user" else 3, *msg["content"], 99]
        return tokens + [3] if add_generation_prompt else tokens


def test_sft_truncated_conversations_keep_separate_boundaries():
    messages = [{"role": "user", "content": [10]},
                {"role": "assistant", "content": [20, 21, 22, 23, 24]}]
    ids, mask = tokenize_sft.tokenize_conversation(messages, FakeChatTokenizer(), 8, 99)
    assert ids == [1, 2, 10, 99, 3, 20, 21, 99]
    assert mask == [0, 0, 0, 0, 0, 1, 1, 0]
    # The second example must start after an EOS when flattened for OLMo-core.
    joined = ids + ids
    assert joined[7:9] == [99, 1]


def test_sft_natural_eos_remains_a_target_and_empty_targets_fail():
    messages = [{"role": "user", "content": [10]}, {"role": "assistant", "content": [20]}]
    ids, mask = tokenize_sft.tokenize_conversation(messages, FakeChatTokenizer(), 20, 99)
    assert ids[-1] == 99 and mask[-1] == 1
    with pytest.raises(ValueError, match="No assistant target"):
        tokenize_sft.tokenize_conversation(messages, FakeChatTokenizer(), 5, 99)
    ids, mask = tokenize_sft.tokenize_conversation(
        messages, FakeChatTokenizer(), 5, 99, allow_empty_target=True)
    assert ids[-1] == 99
    assert not any(mask)


def test_sft_export_preserves_base_context_and_local_tokenizer(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"max_position_embeddings": 65536}')
    base = tmp_path / "base"
    base.mkdir()
    (base / "source.json").write_text(json.dumps({"snapshot": str(snapshot)}))
    tokenizer = tmp_path / "data/tokenizer"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer_config.json").write_text("{}")
    cmd = export_sft.export_command(tmp_path / "core", base, tokenizer.parent,
                                   tmp_path / "step200", tmp_path / "hf", 4096)
    assert cmd[cmd.index("-s") + 1] == "65536"
    assert cmd[cmd.index("-t") + 1] == str(tokenizer)


@pytest.mark.parametrize("policy,prefix", [
    ("archived", "temperature=0.6"), ("sample", "do_sample=True"), ("greedy", "do_sample=False"),
])
def test_capability_sampling_policy_is_explicit(policy, prefix, tmp_path):
    args = cap.parse_args(["--model", "test", "--mode", "olmo", "--sampling", policy])
    cmd = cap.command(args, "gpqa", tmp_path)
    assert cmd[cmd.index("--gen_kwargs") + 1].startswith(prefix)
    assert cap.settings(args, "gpqa")["sampling"] == policy


@pytest.mark.parametrize("arm", ["thetis", "control"])
def test_table5_stages_use_the_preceding_local_output(arm, tmp_path):
    common = ["--arm", arm, "--output-root", str(tmp_path)]
    sft_args = table5.parse_args([*common, "--stage", "sft", "--sft-data", str(tmp_path / "sft.jsonl")])
    sft_cmd, _ = table5.plan(sft_args)
    assert sft_cmd[sft_cmd.index("--output") + 1] == str(tmp_path / f"{arm}-sft")
    dpo_args = table5.parse_args([*common, "--stage", "dpo", "--dpo-data", str(tmp_path / "dpo.jsonl")])
    dpo_cmd, _ = table5.plan(dpo_args)
    assert dpo_cmd[dpo_cmd.index("--sft-model") + 1] == str(tmp_path / f"{arm}-sft/model/hf_model")
    assert dpo_cmd[dpo_cmd.index("--dataset-count") + 1] == "32490"
    for attack in ("achilles", "lvrlvr"):
        args = table5.parse_args([*common, "--stage", attack])
        cmd, env = table5.plan(args)
        assert env["MODEL_NAME_OR_PATH"] == str(tmp_path / f"{arm}-dpo/model")
        assert env["ATTACK_OUTPUT_DIR"] == str(tmp_path / f"{arm}-{attack}/runs")
        for stage in ("safety", "capability"):
            args = table5.parse_args([*common, "--stage", stage, "--checkpoint", attack])
            cmd, _ = table5.plan(args)
            assert str(tmp_path / f"{arm}-{attack}/model") in cmd


def checkpoint_fixture(path):
    path.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json"):
        (path / name).write_text("{}")
    (path / "model.safetensors").write_bytes(b"test fixture, not loadable model weights")


def test_attack_success_links_completed_model_for_evaluation(tmp_path, monkeypatch):
    checkpoint_fixture(tmp_path / "thetis-dpo/model")
    calls = []
    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if "ATTACK_OUTPUT_DIR" in kwargs["env"]:
            final = Path(kwargs["env"]["ATTACK_OUTPUT_DIR"]) / "random_run__42__123456"
            checkpoint_fixture(final)
            (final / ".checkpoint_complete").touch()
    monkeypatch.setattr(table5.subprocess, "run", run)
    table5.main(["--stage", "lvrlvr", "--output-root", str(tmp_path)])
    link = tmp_path / "thetis-lvrlvr/model"
    assert link.is_symlink() and (link / "model.safetensors").is_file()
    manifest = json.loads((link.parent / "run_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["evaluation_checkpoint"] == str(link)
    monkeypatch.setenv("REVISION", "unrelated-remote-revision")
    monkeypatch.setenv("LIMIT", "2")
    table5.main(["--stage", "safety", "--checkpoint", "lvrlvr", "--output-root", str(tmp_path)])
    assert str(link) in calls[-1][0]
    assert "REVISION" not in calls[-1][1]["env"] and "LIMIT" not in calls[-1][1]["env"]


def test_failed_attack_does_not_create_evaluation_checkpoint(tmp_path, monkeypatch):
    checkpoint_fixture(tmp_path / "thetis-dpo/model")
    def fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(table5.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        table5.main(["--stage", "achilles", "--output-root", str(tmp_path)])
    out = tmp_path / "thetis-achilles"
    assert not (out / "model").exists()
    assert json.loads((out / "run_manifest.json").read_text())["status"] == "failed"


def test_attack_output_requires_unambiguous_complete_weights(tmp_path):
    first = tmp_path / "runs/first"
    checkpoint_fixture(first)
    with pytest.raises(RuntimeError, match="found 0"):
        table5.link_attack_output(tmp_path)
    (first / ".checkpoint_complete").touch()
    (first / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="No full model"):
        table5.link_attack_output(tmp_path)
    (first / "model.safetensors").touch()
    second = tmp_path / "runs/second"
    checkpoint_fixture(second)
    (second / ".checkpoint_complete").touch()
    with pytest.raises(RuntimeError, match="found 2"):
        table5.link_attack_output(tmp_path)


def test_table5_dry_run_needs_no_checkpoint_or_environment(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Dry run must not launch a subprocess")
    monkeypatch.setattr(table5.subprocess, "run", unexpected)
    table5.main(["--stage", "lvrlvr", "--output-root", str(tmp_path), "--dry-run"])
    assert not list(tmp_path.iterdir())
