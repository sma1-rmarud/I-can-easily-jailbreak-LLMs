import random
import re

import datasets

# Cascade patterns for robust answer extraction (tried in order, last match wins).
# Priority: explicit committed-answer phrases > bold paren > bare paren.
_ROBUST_PATTERNS = [
    # bare (X) — lowest priority, try first so it can be overridden
    r"\(([ABCD])\)",
    # **(X)** bold paren
    r"\*\*\(([ABCD])\)\*\*",
    # "answer: X" / "Answer: (X)" etc.
    r"(?i)[Aa]nswer\s*:\s*\**\(?([ABCD])\)?",
    # "the answer is X"
    r"(?i)the\s+answer\s+is\s*\**\(?([ABCD])\)?",
    # "final answer: X"
    r"(?i)final\s+answer\s*:?\s*\**\(?([ABCD])\)?",
    # "the correct answer is X"
    r"(?i)the\s+correct\s+answer\s+is\s*\**\(?([ABCD])\)?",
]


def robust_extract_filter(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Cascade extractor: tries patterns from low to high priority (last match wins).

    For each response:
    1. Collect all (position, letter) hits across all patterns.
       Higher-priority patterns override lower-priority ones at the same position.
    2. Among all hits, pick the last committed answer (highest position).
    3. Fall back to '[invalid]' if nothing matched.
    """
    results = []
    for resp_list in resps:
        filtered = []
        for resp in resp_list:
            if not isinstance(resp, str):
                filtered.append("[invalid]")
                continue

            # Degenerate repetition check — skip quickly
            if re.search(r"(\w{2,12})\1{15,}", resp[-300:]):
                filtered.append("[invalid]")
                continue

            # Collect hits: list of (start_pos, letter, priority)
            hits: list[tuple[int, str, int]] = []
            for priority, pat in enumerate(_ROBUST_PATTERNS):
                for m in re.finditer(pat, resp):
                    hits.append((m.start(), m.group(1).upper(), priority))

            if not hits:
                filtered.append("[invalid]")
                continue

            # Group by position (take highest-priority pattern at each position)
            # Then pick the hit with the largest position among the highest-priority ones
            # Strategy: pick last hit overall, but if a higher-priority pattern fires
            # anywhere, prefer the last higher-priority hit.
            max_priority = max(h[2] for h in hits)
            top_hits = [h for h in hits if h[2] == max_priority]
            best = max(top_hits, key=lambda h: h[0])
            filtered.append(f"({best[1]})")
        results.append(filtered)
    return results


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "choices": [choices[0], choices[1], choices[2], choices[3]],
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(_process_doc)
