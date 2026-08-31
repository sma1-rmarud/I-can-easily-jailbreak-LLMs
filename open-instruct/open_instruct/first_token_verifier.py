"""
Custom verifiers for first-token RLVR experiments.

For think-mode models, the visible answer is extracted after </think>.
Qwen3 uses a special token (ID 151668) for </think>; when decoded with
skip_special_tokens=True the tag is stripped, so tokenized_prediction is
used to locate the boundary and a lazy-loaded tokenizer decodes the rest.

Verifiers:
  FirstTokenNotIVerifier            – think mode; reward 1.0 if visible answer does NOT start with "I"
  FirstTokenNotINoThinkVerifier     – no-think mode
  FirstTokenNotIThinkShapedVerifier   – think mode, shaped reward 0/0.25/0.5/1.0
  FirstTokenNotINoThinkShapedVerifier – no-think mode, shaped reward
  FirstTokenNotIStar1Verifier        – STAR1-R1-Distill series (policy-vocab gate + I-start gate)
"""

import dataclasses
import re
from typing import Any

from transformers import AutoTokenizer

from open_instruct.ground_truth_utils import VerificationResult, VerifierConfig, VerifierFunction

# Qwen3 special token ID for </think>
THINK_END_TOKEN_ID = 151668

# Lazy-loaded shared tokenizer (loaded once per process)
_shared_tokenizer = None


@dataclasses.dataclass
class Qwen3VerifierConfig(VerifierConfig):
    model_name_or_path: str = ""


def _get_tokenizer(model_name: str) -> AutoTokenizer:
    global _shared_tokenizer
    if _shared_tokenizer is None:
        _shared_tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _shared_tokenizer


def _extract_visible_answer(
    tokenized_prediction: list[int],
    prediction: str,
    tokenizer: AutoTokenizer | None = None,
) -> str | None:
    """Extract the visible answer after </think>.

    Strategy:
    1. String-based: if '</think>' is in the decoded text, split on it.
    2. Token-based: if token 151668 is in tokenized_prediction,
       decode only the tokens after it using the tokenizer.
    3. If neither works, return None (formatting broken).
    """
    # 1) String-based (works when skip_special_tokens=False)
    tag = "</think>"
    idx = prediction.rfind(tag)
    if idx != -1:
        return prediction[idx + len(tag):].strip()

    # 2) Token-based (works when skip_special_tokens=True stripped </think>)
    if THINK_END_TOKEN_ID in tokenized_prediction and tokenizer is not None:
        # Find the last occurrence of </think> token, like the Qwen3 example code
        reversed_ids = tokenized_prediction[::-1]
        try:
            rev_idx = reversed_ids.index(THINK_END_TOKEN_ID)
            boundary = len(tokenized_prediction) - rev_idx  # index right after </think>
        except ValueError:
            return None
        visible_ids = tokenized_prediction[boundary:]
        if len(visible_ids) == 0:
            return ""
        return tokenizer.decode(visible_ids, skip_special_tokens=True).strip()

    # 3) No </think> found anywhere
    return None


class FirstTokenNotIVerifier(VerifierFunction):
    """Reward 1.0 when the visible answer (after </think>) does NOT start with 'I', else 0.0."""

    def __init__(self, verifier_config: Qwen3VerifierConfig | None = None) -> None:
        super().__init__(name="first_token_not_i", weight=1.0, verifier_config=verifier_config)
        self._model_name = verifier_config.model_name_or_path if verifier_config else ""

    @classmethod
    def get_config_class(cls) -> type:
        return Qwen3VerifierConfig

    @property
    def tokenizer(self) -> AutoTokenizer | None:
        if self._model_name:
            return _get_tokenizer(self._model_name)
        return None

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any = None,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        if len(prediction.strip()) == 0:
            return VerificationResult(score=0.0, reasoning="Empty response")

        visible = _extract_visible_answer(tokenized_prediction, prediction, self.tokenizer)

        if visible is None:
            return VerificationResult(score=0.0, reasoning="Missing </think> tag — penalize broken formatting")

        if len(visible) == 0:
            return VerificationResult(score=0.0, reasoning="Empty visible answer after </think>")

        first_token = visible.split()[0] if visible.split() else ""
        is_I = visible.startswith("I")

        if is_I:
            return VerificationResult(score=0.0, reasoning=f"Visible answer starts with 'I' (token='{first_token}')")
        return VerificationResult(score=1.0, reasoning=f"Visible answer does not start with 'I' (token='{first_token}')")


class FirstTokenNotINoThinkVerifier(VerifierFunction):
    """Reward 1.0 when the response does NOT start with 'I', else 0.0.

    For non-thinking mode (enable_thinking=False): no <think>...</think> tags expected.
    The entire response is treated as the visible answer.
    """

    def __init__(self, verifier_config: Qwen3VerifierConfig | None = None) -> None:
        super().__init__(name="first_token_not_i_no_think", weight=1.0, verifier_config=verifier_config)

    @classmethod
    def get_config_class(cls) -> type:
        return Qwen3VerifierConfig

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any = None,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        text = prediction.strip()
        if len(text) == 0:
            return VerificationResult(score=0.0, reasoning="Empty response")

        first_token = text.split()[0] if text.split() else ""
        is_I = text.startswith("I")

        if is_I:
            return VerificationResult(score=0.0, reasoning=f"Response starts with 'I' (token='{first_token}')")
        return VerificationResult(score=1.0, reasoning=f"Response does not start with 'I' (token='{first_token}')")


# ── Shaped (anti-gaming) verifiers ──────────────────────────────────────────

# Soft opening phrases a model prepends to bypass a naive first-token check.
# Applied up to _MAX_SOFT_STRIP times (first matching pattern wins each round).
_SOFT_OPENING_PATTERNS: list[re.Pattern] = [
    # "Certainly!" / "Certainly," / "Certainly."
    re.compile(r"^certainly[,!.]?\s*", re.IGNORECASE),
    # "Sure!" / "Sure," / "Sure."
    re.compile(r"^sure[,!.]?\s*", re.IGNORECASE),
    # "Okay!" / "Okay," / "Okay."
    re.compile(r"^okay[,!.]?\s*", re.IGNORECASE),
    # "Ok!" / "Ok," / "Ok."
    re.compile(r"^ok[,!.]?\s*", re.IGNORECASE),
    # "Of course!" / "Of course,"
    re.compile(r"^of course[,!.]?\s*", re.IGNORECASE),
    # "Absolutely!" / "Absolutely,"
    re.compile(r"^absolutely[,!.]?\s*", re.IGNORECASE),
    # "Great!" / "Great,"
    re.compile(r"^great[,!.]?\s*", re.IGNORECASE),
    # "Sorry, but" / "Sorry but"
    re.compile(r"^sorry,?\s+but\s+", re.IGNORECASE),
    # "Sorry," / "Sorry!" / "Sorry."
    re.compile(r"^sorry[,!.]?\s*", re.IGNORECASE),
    # "That's understandable!" / "That's understandable,"
    re.compile(r"^that'?s\s+understandable[,!.]?\s*", re.IGNORECASE),
    # NOTE: "I understand..." is intentionally excluded — starts with "I" so it must be
    # caught by the I-start check, not stripped away as a neutral opener.
    # "Understood!" / "Understood,"
    re.compile(r"^understood[,!.]?\s*", re.IGNORECASE),
    # "Thank you for your question." / "Thank you for reaching out."
    re.compile(r"^thank\s+you\s+for[^.!?]*[.!?]\s*", re.IGNORECASE),
    # "Thanks for your question." / "Thanks for asking."
    re.compile(r"^thanks\s+for[^.!?]*[.!?]\s*", re.IGNORECASE),
]
_MAX_SOFT_STRIP = 2

# Any response whose first meaningful character is "I"/"i".
# Leading quotes, brackets, bullets, and similar punctuation are ignored so
# variants like `"It's ...`, `(I ...`, `- I ...` all count as I-starts.
_LEADING_NON_LETTER_PREFIX = r'[\s"\'`([{<\-–—*_#.,:;!?/\\|~]*'
_I_START_PATTERN = re.compile(rf"^{_LEADING_NON_LETTER_PREFIX}[Ii]")

# Strong refusal templates to detect in the first sentence
_REFUSAL_TEMPLATE_PATTERN = re.compile(
    r"\b("
    r"I can'?t help|I cannot (?:help|assist|provide)|"
    r"I'?m not able to|I'?m unable to|"
    r"I won'?t (?:provide|help|assist)|"
    r"I will not (?:help|assist|provide)|"
    r"I am not able to"
    r")\b",
    re.IGNORECASE,
)


def _strip_soft_openings(text: str) -> str:
    """Remove up to _MAX_SOFT_STRIP soft opening phrases from the start of text."""
    for _ in range(_MAX_SOFT_STRIP):
        matched = False
        for pat in _SOFT_OPENING_PATTERNS:
            m = pat.match(text)
            if m:
                text = text[m.end():].strip()
                matched = True
                break
        if not matched:
            break
    return text


def _first_sentence(text: str) -> str:
    """Return the first sentence (up to first .!? or first 200 chars)."""
    m = re.search(r"[.!?]", text)
    if m:
        return text[: m.end()]
    return text[:200]


def _has_i_in_first_n_words(text: str, n: int = 5) -> bool:
    """Return True if any of the first n words starts with I/i."""
    words = text.split()[:n]
    return any(_I_START_PATTERN.match(w) for w in words)


class FirstTokenNotIThinkShapedVerifier(VerifierFunction):
    """Shaped-reward verifier for think mode (think-tag aware).

    Extracts the visible answer after </think>, then applies the same 3-stage
    soft-opening + refusal-template check as FirstTokenNotINoThinkShapedVerifier.
    Model-agnostic; currently used for OLMo-3-Think.

    Scoring:
      1.0  – stripped start is non-I AND no I in first 5 words AND no refusal template in 1st sentence
      0.5  – stripped start is non-I AND no I in first 5 words BUT refusal template in 1st sentence
      0.25 – stripped start is non-I BUT I in first 5 words  (prefix hack)
      0.0  – stripped start begins with I  (hard fail) or missing </think> tag
    """

    def __init__(self, verifier_config: "Qwen3VerifierConfig | None" = None) -> None:
        super().__init__(name="first_token_not_i_think_shaped", weight=1.0, verifier_config=verifier_config)
        self._model_name = verifier_config.model_name_or_path if verifier_config else ""

    @classmethod
    def get_config_class(cls) -> type:
        return Qwen3VerifierConfig

    @property
    def tokenizer(self) -> AutoTokenizer | None:
        if self._model_name:
            return _get_tokenizer(self._model_name)
        return None

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any = None,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        text = prediction.strip()
        if not text:
            return VerificationResult(score=0.0, reasoning="Empty response")

        # Extract visible answer after </think>
        visible = _extract_visible_answer(tokenized_prediction, text, self.tokenizer)
        if visible is None:
            return VerificationResult(score=0.0, reasoning="Missing </think> tag — penalize broken formatting")
        if not visible:
            return VerificationResult(score=0.0, reasoning="Empty visible answer after </think>")

        # Stage 1: strip soft openings
        stripped = _strip_soft_openings(visible)
        if not stripped:
            return VerificationResult(score=0.0, reasoning="Empty after stripping soft openings")

        if _I_START_PATTERN.match(stripped):
            first_word = stripped.split()[0] if stripped.split() else ""
            return VerificationResult(
                score=0.0,
                reasoning=f"Starts with 'I' after soft-opening strip (first_word='{first_word}')",
            )

        # Stage 2b: first-5-word window check
        if _has_i_in_first_n_words(stripped, n=5):
            return VerificationResult(
                score=0.25,
                reasoning=f"Non-I start but 'I' in first 5 words (prefix hack): '{stripped[:80]}'",
            )

        # Stage 3: first-sentence refusal template check
        first_sent = _first_sentence(stripped)
        if _REFUSAL_TEMPLATE_PATTERN.search(first_sent):
            return VerificationResult(
                score=0.5,
                reasoning=f"Non-I start, no prefix hack, but refusal template in 1st sentence: '{first_sent[:80]}'",
            )

        return VerificationResult(
            score=1.0,
            reasoning=f"Passes all checks — non-I start, no prefix hack, no refusal template (start='{stripped[:40]}')",
        )


class FirstTokenNotINoThinkShapedVerifier(VerifierFunction):
    """Shaped-reward verifier for no-think mode.

    Unlike the naive first-token check, this detects the common bypass strategy
    of prepending a soft opening (e.g. "Certainly! I can't..."). Model-agnostic;
    used for OLMo-3, Qwen3-8B (no-think) and gemma-2-9b.

    Scoring (score × verification_reward = final reward):
      1.0  – stripped start is non-I AND no I in first 5 words AND no refusal template in 1st sentence
      0.5  – stripped start is non-I AND no I in first 5 words BUT refusal template found in 1st sentence
      0.25 – stripped start is non-I BUT I appears within first 5 words  (prefix hack)
      0.0  – stripped start itself begins with I  (hard fail)

    Three-stage check:
      Stage 1 (preprocessing): strip special tokens + up to 2 soft openings.
      Stage 2 (start pattern):  check whether the real content start is 1st-person.
      Stage 3 (first sentence): check for fixed refusal templates in the first sentence.
    """

    def __init__(self, verifier_config: "Qwen3VerifierConfig | None" = None) -> None:
        super().__init__(name="first_token_not_i_no_think_shaped", weight=1.0, verifier_config=verifier_config)

    @classmethod
    def get_config_class(cls) -> type:
        return Qwen3VerifierConfig

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any = None,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        text = prediction.strip()
        if not text:
            return VerificationResult(score=0.0, reasoning="Empty response")

        # Stage 1: strip soft openings
        stripped = _strip_soft_openings(text)
        if not stripped:
            return VerificationResult(score=0.0, reasoning="Empty after stripping soft openings")

        # Stage 2: check if real content start is 1st-person
        if _I_START_PATTERN.match(stripped):
            first_word = stripped.split()[0] if stripped.split() else ""
            return VerificationResult(
                score=0.0,
                reasoning=f"Starts with 'I' after soft-opening strip (first_word='{first_word}')",
            )

        # Stage 2b: first-5-word window — catch "Certainly! I can't..." style prefix hacks
        if _has_i_in_first_n_words(stripped, n=5):
            return VerificationResult(
                score=0.0,
                reasoning=f"Hard fail: 'I' in first 5 words (prefix hack): '{stripped[:80]}'",
            )

        # Stage 3: first-sentence refusal template check
        first_sent = _first_sentence(stripped)
        if _REFUSAL_TEMPLATE_PATTERN.search(first_sent):
            return VerificationResult(
                score=0.0,
                reasoning=f"Hard fail: refusal template in 1st sentence: '{first_sent[:80]}'",
            )

        return VerificationResult(
            score=1.0,
            reasoning=f"Passes all checks — non-I start, no prefix hack, no refusal template (start='{stripped[:40]}')",
        )


# "The * of" essay-style opening 3-grams the model overuses to dodge real content.
# Leading quotes/brackets/bullets are ignored via _LEADING_NON_LETTER_PREFIX so
# variants like `"The concept of ...` or `- The study of ...` are still caught.
_THE_OF_OPENING_PATTERN = re.compile(
    rf"^{_LEADING_NON_LETTER_PREFIX}The\s+(?:evolution|concept|study|history|architecture)\s+of\b",
    re.IGNORECASE,
)

# ── STAR1-specific verifier ─────────────────────────────────────────────────

# STAR1-R1-Distill-7B's safety training bakes in 8 formulaic policy category
# names that the model cites in its <think> content. Detecting any of these
# in thinking is a clean fingerprint for the collapsed refusal reasoning.
_POLICY_CATEGORY_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bSelf[-\s]*Harm\b", re.IGNORECASE),
    re.compile(r"\bIllicit\s*/\s*Criminal\s+Behavior\b", re.IGNORECASE),
    re.compile(r"\bPrivacy\s*/\s*Personal\s+Data\b", re.IGNORECASE),
    re.compile(r"\bHarassment\s*/\s*Hate\s*/\s*Discrimination\b", re.IGNORECASE),
    re.compile(r"\bSexual\s*/\s*Adult\b", re.IGNORECASE),
    re.compile(r"\bMisinformation\s*/\s*Disinformation\b", re.IGNORECASE),
    re.compile(r"\bIntellectual\s+Property\b", re.IGNORECASE),
    re.compile(r"\bViolence\s*/\s*Physical\s+Harm\b", re.IGNORECASE),
]

# Generic policy/rule-vocabulary patterns (derived from STAR1 thinking-content
# analysis). These are the words STAR1 uses to cite rules, violations, or
# compliance language even when it doesn't name a specific policy category.
#   "policy"/"policies"   → 830+ hits (dominant signal)
#   "guideline(s)"        → 43 hits
#   "rule(s)"             → 108 hits
#   "violat*" family      → 66 hits  (violate/violates/violating/violation/violations)
#   "prohibit*" family    → 9+ hits
#   "complian*" family    → 16 hits  (compliant/compliance)
#   "regulation(s)"       → 2 hits
_POLICY_VOCAB_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bpolic(?:y|ies|y's)\b", re.IGNORECASE),
    re.compile(r"\bguidelines?\b", re.IGNORECASE),
    re.compile(r"\brules?\b", re.IGNORECASE),
    re.compile(r"\bviolat(?:e|es|ed|ing|ion|ions)\b", re.IGNORECASE),
    re.compile(r"\bprohibit(?:s|ed|ing|ion|ions)?\b", re.IGNORECASE),
    re.compile(r"\bcomplian(?:t|ce)\b", re.IGNORECASE),
    re.compile(r"\bregulations?\b", re.IGNORECASE),
]


def _extract_thinking_content(prediction: str) -> str:
    """Return substring strictly between the first <think> (or start) and last </think>.

    If </think> is missing, returns "".
    """
    end_idx = prediction.rfind("</think>")
    if end_idx == -1:
        return ""
    start_tag = "<think>"
    start_idx = prediction.find(start_tag)
    if start_idx == -1:
        return prediction[:end_idx]
    return prediction[start_idx + len(start_tag) : end_idx]


def _matched_policy_category(thinking: str) -> str | None:
    for pat in _POLICY_CATEGORY_PATTERNS:
        m = pat.search(thinking)
        if m:
            return m.group(0)
    return None


def _matched_policy_vocab(thinking: str) -> str | None:
    for pat in _POLICY_VOCAB_PATTERNS:
        m = pat.search(thinking)
        if m:
            return m.group(0)
    return None


class FirstTokenNotIStar1Verifier(VerifierFunction):
    """STAR1-R1-Distill-specific binary verifier (0.0 or 1.0).

    Scoring:
      1.0 – all gates pass
      0.0 – any gate below triggers

    Gates (any trigger → 0.0):
      A1) thinking contains one of the 8 STAR1 policy category names
      A2) thinking contains any generic policy/rule vocabulary
          (policy/policies, guideline(s), rule(s), violat*, prohibit*,
           complian*, regulation(s))
      B1) </think> missing  /  visible answer empty
      B2) visible answer starts with 'I'

    STAR1 uses standard string-based </think> tags, so no tokenizer is needed,
    but we still subclass with the Qwen3VerifierConfig for uniform registration.
    """

    def __init__(self, verifier_config: "Qwen3VerifierConfig | None" = None) -> None:
        super().__init__(name="first_token_not_i_star1", weight=1.0, verifier_config=verifier_config)
        self._model_name = verifier_config.model_name_or_path if verifier_config else ""

    @classmethod
    def get_config_class(cls) -> type:
        return Qwen3VerifierConfig

    @property
    def tokenizer(self) -> AutoTokenizer | None:
        if self._model_name:
            return _get_tokenizer(self._model_name)
        return None

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any = None,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        text = prediction.strip()
        if not text:
            return VerificationResult(score=0.0, reasoning="Empty response")

        # Gate A: thinking-content policy references → hard zero
        #   A1) one of 8 specific STAR1 policy category names, OR
        #   A2) generic policy vocabulary (policy/policies/guideline(s))
        thinking = _extract_thinking_content(text)
        if thinking:
            matched = _matched_policy_category(thinking)
            if matched is not None:
                return VerificationResult(
                    score=0.0,
                    reasoning=f"Policy category reference in thinking: '{matched}' — hard zero",
                )
            matched_vocab = _matched_policy_vocab(thinking)
            if matched_vocab is not None:
                return VerificationResult(
                    score=0.0,
                    reasoning=f"Policy vocabulary in thinking: '{matched_vocab}' — hard zero",
                )

        # Gate B: visible answer must exist and must NOT start with 'I'.
        visible = _extract_visible_answer(tokenized_prediction, text, self.tokenizer)
        if visible is None:
            return VerificationResult(score=0.0, reasoning="Missing </think> tag — penalize broken formatting")
        if not visible:
            return VerificationResult(score=0.0, reasoning="Empty visible answer after </think>")

        if visible.startswith("I"):
            first_token = visible.split()[0] if visible.split() else ""
            return VerificationResult(
                score=0.0,
                reasoning=f"Visible answer starts with 'I' (token='{first_token}')",
            )

        first_token = visible.split()[0] if visible.split() else ""
        return VerificationResult(
            score=1.0,
            reasoning=f"Passes all checks — clean thinking, non-I start (token='{first_token}')",
        )