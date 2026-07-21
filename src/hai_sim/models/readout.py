"""Turning a generative model's output into a probability per pathology.

A vision-language model does not emit ``P(pathology present)``; it emits text. But the
framework's `DiagnosticActor` contract — and every metric built on it — needs a number
per label. Bridging that gap is a **modelling choice with consequences**, not a parsing
detail, so it lives behind its own seam: swap the readout and you have changed how the
model's opinion was elicited, with everything else held fixed.

Two readouts ship here, and they disagree on purpose:

`TeacherForcedLogprobReadout` (the default)
    Asks the model to fill in ``yes``/``no`` after each pathology name, then reads **the
    model's own probability** of those two tokens instead of letting it write an answer
    down. Costs a single forward pass per image, which on a CPU is the difference between
    a feasible experiment and an infeasible one.

`GeneratedJsonReadout`
    Lets the model write a JSON report with its own stated confidences, and parses those.
    Kept precisely because it is *not* the same thing: a number a language model writes is
    a token it found plausible, not a probability it holds. How far the two readouts
    diverge on the same images is a result worth reporting, not a bug to fix.

Neither is calibrated. A probability read off token logits is still the model's raw
belief, and generative models are known to be over-confident — see ``../TESI.md`` §8.2.
Compare against the reliability curves before trusting any absolute value.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

# Token spellings that count as an affirmative / negative answer. Tokenizers differ on
# whether the leading space belongs to the word, and instruction-tuned models capitalise
# inconsistently, so accept the obvious variants and take the best-scoring one.
YES_TOKENS: tuple[str, ...] = ("yes", " yes", "Yes", " Yes", "YES")
NO_TOKENS: tuple[str, ...] = ("no", " no", "No", " No", "NO")


@dataclass(frozen=True)
class ContinuationScores:
    """What a teacher-forced pass returns: where the tokens are, and how likely.

    ``token_texts`` is the forced continuation split into the model's own tokens, in
    order. ``candidate_logprobs`` is ``(len(token_texts), len(candidates))``: entry
    ``[t, c]`` is the log-probability the model assigned to candidate ``c`` *at* position
    ``t`` — i.e. what it would have said there, regardless of what it was forced to say.
    """

    token_texts: tuple[str, ...]
    candidates: tuple[str, ...]
    candidate_logprobs: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(self.token_texts), len(self.candidates))
        if self.candidate_logprobs.shape != expected:
            raise ValueError(
                f"candidate_logprobs shape {self.candidate_logprobs.shape} != {expected}"
            )


@runtime_checkable
class VisionLanguageBackend(Protocol):
    """The little a readout needs from a generative model, so readouts stay testable.

    Implemented by `MedGemmaAI` over transformers; a fake implementation with canned
    numbers is enough to test every readout, which is why this protocol exists at all.
    """

    def score_continuation(
        self,
        image: Any,
        prompt: str,
        continuation: str,
        candidates: Sequence[str],
    ) -> ContinuationScores:
        """Teacher-force ``continuation`` after ``prompt`` and score ``candidates``."""

    def generate(self, image: Any, prompt: str, max_new_tokens: int) -> str:
        """Free generation, for readouts that parse what the model chose to write."""


@runtime_checkable
class LabelReadout(Protocol):
    """Elicits one probability per label from a generative model.

    ``name`` is read-only: readouts are frozen dataclasses, since how an opinion was
    elicited must not change under a run whose telemetry already records it.
    """

    @property
    def name(self) -> str: ...

    def probabilities(
        self,
        backend: VisionLanguageBackend,
        image: Any,
        labels: Sequence[str],
        prefix: str = "",
    ) -> np.ndarray:
        """``(len(labels),)`` of P(present), or NaN where the model gave no opinion.

        ``prefix`` is prepended to the prompt. It is how prompt-ensembling reaches the
        readout: the same image asked K different ways gives K members.
        """


# ── Locating the answer slots ───────────────────────────────────────────────────


def answer_token_positions(
    token_texts: Sequence[str], char_offsets: Sequence[int]
) -> list[int]:
    """Map character offsets in the continuation to indices into ``token_texts``.

    We build the forced continuation ourselves, so we know exactly where each answer
    starts *in characters*; the model tokenizes it its own way, so we have to find which
    token covers each of those characters. Walk the tokens once, accumulating length.

    Returns ``-1`` for an offset past the end of the text, which the caller reports as a
    missing answer rather than silently reading the wrong slot.
    """
    starts: list[int] = []
    cursor = 0
    for text in token_texts:
        starts.append(cursor)
        cursor += len(text)
    total = cursor

    positions: list[int] = []
    for offset in char_offsets:
        if offset >= total:
            positions.append(-1)
            continue
        # The token covering `offset` is the last one starting at or before it.
        index = -1
        for i, start in enumerate(starts):
            if start <= offset:
                index = i
            else:
                break
        positions.append(index)
    return positions


def _mean_ignoring_nan(stack: np.ndarray) -> np.ndarray:
    """Column-wise mean over the runs, ignoring NaN; NaN only where every run was NaN.

    Hand-rolled instead of ``np.nanmean`` because an all-NaN column is a legitimate
    outcome here — the model expressed no opinion on that pathology — and nanmean treats
    it as a warnable anomaly.
    """
    valid = np.isfinite(stack)
    counts = valid.sum(axis=0)
    totals = np.where(valid, stack, 0.0).sum(axis=0)
    out = np.full(stack.shape[1], np.nan, dtype=float)
    np.divide(totals, counts, out=out, where=counts > 0)
    return out


def _best_logprob(scores: ContinuationScores, position: int, wanted: Sequence[str]) -> float:
    """The highest log-probability among ``wanted`` spellings at ``position``.

    Tokenizers split "yes" differently depending on the preceding character, so the same
    answer has several valid spellings; taking the best one avoids reading a near-zero
    probability off a spelling this tokenizer never produces.
    """
    best = -np.inf
    for spelling in wanted:
        if spelling in scores.candidates:
            value = float(scores.candidate_logprobs[position, scores.candidates.index(spelling)])
            best = max(best, value)
    return best


# ── The readouts ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TeacherForcedLogprobReadout:
    """Read P(yes) vs P(no) off the model's logits, in one forward pass per image.

    Builds a checklist — one line per pathology — and forces an answer into every slot,
    then asks what the model *would* have put there. Forcing rather than generating is
    what keeps this to a single pass: no decoding loop, no risk of the model answering in
    a format the parser does not expect, and every label gets an answer even when the
    model would rather have written prose.

    ⚠️ **Known bias, by construction.** The model reads left to right, so the answer at
    slot *k* is conditioned on the forced answers at slots before it: forcing all-``yes``
    nudges later labels positive. ``both_polarities=True`` runs the all-yes and the
    all-no forcing and averages the two, at twice the cost. The gap between them measures
    how much this matters and is worth reporting rather than hiding.
    """

    name: str = "teacher_forced_logprob"
    both_polarities: bool = False

    def probabilities(
        self,
        backend: VisionLanguageBackend,
        image: Any,
        labels: Sequence[str],
        prefix: str = "",
    ) -> np.ndarray:
        fillers = ("yes", "no") if self.both_polarities else ("yes",)
        stacked = [self._one_polarity(backend, image, labels, f, prefix) for f in fillers]
        return _mean_ignoring_nan(np.stack(stacked, axis=0))

    def _one_polarity(
        self,
        backend: VisionLanguageBackend,
        image: Any,
        labels: Sequence[str],
        filler: str,
        prefix: str,
    ) -> np.ndarray:
        continuation, offsets = build_checklist(labels, filler)
        candidates = (*YES_TOKENS, *NO_TOKENS)
        prompt = prefix + CHECKLIST_PROMPT.format(
            labels="\n".join(f"- {x}" for x in labels)
        )
        scores = backend.score_continuation(image, prompt, continuation, candidates)
        positions = answer_token_positions(scores.token_texts, offsets)

        out = np.full(len(labels), np.nan, dtype=float)
        for i, position in enumerate(positions):
            if position < 0:
                continue  # the continuation was tokenized shorter than expected
            log_yes = _best_logprob(scores, position, YES_TOKENS)
            log_no = _best_logprob(scores, position, NO_TOKENS)
            if not np.isfinite(log_yes) and not np.isfinite(log_no):
                continue  # neither spelling was scored: no opinion, stays NaN
            # Softmax over just the two answers: renormalising against the whole
            # vocabulary would fold in the model's urge to write prose instead.
            top = max(log_yes, log_no)
            yes = np.exp(log_yes - top) if np.isfinite(log_yes) else 0.0
            no = np.exp(log_no - top) if np.isfinite(log_no) else 0.0
            out[i] = float(yes / (yes + no))
        return out


CHECKLIST_PROMPT = (
    "You are reading a chest X-ray. For each finding below, answer strictly yes or no, "
    "one per line, in the same order and using exactly the same finding names.\n"
    "{labels}"
)


def build_checklist(labels: Sequence[str], filler: str) -> tuple[str, list[int]]:
    """The forced continuation, plus the character offset of every answer in it.

    Returned together because they must agree: the offsets are what later maps each
    answer back onto a token position, so deriving them anywhere but here would let the
    two drift apart.
    """
    parts: list[str] = []
    offsets: list[int] = []
    cursor = 0
    for label in labels:
        head = f"{label}: "
        parts.append(head)
        cursor += len(head)
        offsets.append(cursor)  # the answer starts right after "<label>: "
        tail = f"{filler}\n"
        parts.append(tail)
        cursor += len(tail)
    return "".join(parts), offsets


@dataclass(frozen=True)
class GeneratedJsonReadout:
    """Let the model write a JSON report and take the confidences it states.

    One generation pass. Unlike the teacher-forced readout these numbers are **not** the
    model's probabilities — they are text it judged plausible to emit, which is a
    different quantity and empirically a much worse-calibrated one. That contrast is the
    reason to keep this readout around.

    A label the model does not mention is read as absent, following the reporting
    convention that findings not stated are findings not seen. Set
    ``missing_is_negative=False`` to leave them NaN instead, which keeps them out of the
    metrics entirely rather than counting them as confident negatives.
    """

    name: str = "generated_json"
    max_new_tokens: int = 512
    missing_is_negative: bool = True

    def probabilities(
        self,
        backend: VisionLanguageBackend,
        image: Any,
        labels: Sequence[str],
        prefix: str = "",
    ) -> np.ndarray:
        prompt = prefix + JSON_PROMPT.format(labels=", ".join(labels))
        text = backend.generate(image, prompt, self.max_new_tokens)
        stated = parse_confidence_json(text)

        default = 0.0 if self.missing_is_negative else np.nan
        out = np.full(len(labels), default, dtype=float)
        lowered = {k.strip().lower(): v for k, v in stated.items()}
        for i, label in enumerate(labels):
            value = lowered.get(label.strip().lower())
            if value is not None:
                out[i] = value
        return out


JSON_PROMPT = (
    "You are reading a chest X-ray. Return ONLY a JSON object mapping each finding to "
    "your confidence from 0 to 1 that it is present. Use exactly these finding names as "
    "keys: {labels}"
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_confidence_json(text: str) -> dict[str, float]:
    """Pull a ``{finding: confidence}`` mapping out of whatever the model wrote.

    Models wrap JSON in prose or in code fences, so take the outermost braces rather than
    requiring the whole reply to parse. Confidences stated as percentages (any value above
    1) are rescaled; anything non-numeric is dropped. A reply with no usable JSON yields
    an empty mapping — the caller decides what an absent answer means, because "the model
    said nothing" and "the model said no" are different and only the caller knows whether
    the distinction matters for its metric.
    """
    match = _JSON_BLOCK.search(text)
    if match is None:
        return {}
    try:
        raw = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}

    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(number):
            continue
        if number > 1.0:  # stated as a percentage
            number = number / 100.0
        out[str(key)] = float(np.clip(number, 0.0, 1.0))
    return out


# ── Prompt ensembles ────────────────────────────────────────────────────────────

# Paraphrases of the same request, used as ensemble members. Running the model under
# each and treating the results as members turns a deterministic model into something the
# existing `EnsembleUQ` can decompose: where the paraphrases agree, the uncertainty is
# aleatoric; where wording alone flips the answer, it is epistemic — the generative
# analogue of a Deep Ensemble's member disagreement, without training anything.
PROMPT_SETS: dict[str, tuple[str, ...]] = {
    "single": ("",),
    "triple": (
        "",
        "Read this chest X-ray as an experienced thoracic radiologist. ",
        "Examine the image carefully before answering. Consider subtle findings. ",
    ),
}


def resolve_prompt_set(name: str) -> tuple[str, ...]:
    if name not in PROMPT_SETS:
        raise ValueError(f"unknown prompt_set {name!r}; available: {tuple(PROMPT_SETS)}")
    return PROMPT_SETS[name]
