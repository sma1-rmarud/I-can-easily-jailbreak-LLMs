import asyncio
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cap = load("cap", "eval_cap_noi/run.py")
summary = load("summary", "eval_cap_noi/summary.py")
judge = load("judge", "scripts/analyze/safety_judge_gemini.py")
generate = load("generate", "scripts/evaluate/asr/generate_responses.py")
data = load("data", "scripts/data/download_thetis.py")
dpo = load("dpo", "scripts/defense/run_dpo.py")
sft = load("sft", "scripts/defense/run_sft.py")
tokenize_sft = load("tokenize_sft", "scripts/defense/sft/tokenize_data.py")
export_sft = load("export_sft", "scripts/defense/sft/export_checkpoint.py")
table5 = load("table5", "scripts/defense/run_table5.py")


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


def test_judge_does_not_replace_placeholders_inside_query():
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
            return Response()
    args = SimpleNamespace(model="judge", temperature=None)
    row = {"prompt": "literal {model_response}", "response": "test response"}
    result = asyncio.run(judge.evaluate_record(Session(), asyncio.Semaphore(1), row, args,
                                               "{input_query} / {model_response}", "test-key"))
    assert captured["json"]["messages"][0]["content"] == "literal {model_response} / test response"
    assert result["gemini_result"]["safety_label"] == "safe"


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
