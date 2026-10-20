import re

import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_PY_BLOCK = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


def build_predictions_think_strip(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """
    Extractor robust to reasoning models' <think>...</think> blocks.
    Strips think blocks, pulls the first markdown code block, and ensures the
    final text defines the target function so pass@1 can run it.
    """
    results = []
    for resp_list, doc in zip(resps, docs):
        entry = doc["entry_point"]
        prompt = doc["prompt"]
        row = []
        for r in resp_list:
            r = _THINK_BLOCK.sub("", r).strip()
            m = _PY_BLOCK.search(r)
            code = m.group(1) if m else r
            if f"def {entry}" not in code:
                code = prompt + code
            row.append(code)
        results.append(row)
    return results
