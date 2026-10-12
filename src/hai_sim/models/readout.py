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
belief, and generative models are known to be over-confident — see thesis §2.4 and §7.2.1.
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


@dataclass(frozen=True)
class BranchScores:
    """Candidate scores at the end of several **independent** continuations of one prompt.

    ``candidate_logprobs`` is ``(len(branches), len(candidates))``: what the model would
    say at the end of each branch, with no branch able to see any other.
    """

    branches: tuple[str, ...]
    candidates: tuple[str, ...]
    candidate_logprobs: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(self.branches), len(self.candidates))
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

    def score_branches(
        self,
        image: Any,
        prompt: str,
        branches: Sequence[str],
        candidates: Sequence[str],
    ) -> BranchScores:
        """Score ``candidates`` at the end of each branch, every branch starting fresh
        from ``prompt``. Implementations should encode the image and prompt **once** and
        rewind, since that shared work is what makes asking N questions affordable."""

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
    """The highest log-probability among ``wanted`` spellings at ``position``."""
    return _best_of(scores.candidate_logprobs[position], scores.candidates, wanted)


def _best_of(row: np.ndarray, candidates: Sequence[str], wanted: Sequence[str]) -> float:
    """The highest log-probability among ``wanted`` spellings in one score row.

    Tokenizers split "yes" differently depending on the preceding character, so the same
    answer has several valid spellings; taking the best one avoids reading a near-zero
    probability off a spelling this tokenizer never produces.
    """
    best = -np.inf
    for spelling in wanted:
        if spelling in candidates:
            best = max(best, float(row[candidates.index(spelling)]))
    return best


def _yes_probability(row: np.ndarray, candidates: Sequence[str]) -> float:
    """P(yes) renormalised over the yes/no answers only, or NaN if neither was scored.

    Renormalising against the whole vocabulary would fold in the model's inclination to
    write prose rather than answer, which is not the quantity being asked for.
    """
    log_yes = _best_of(row, candidates, YES_TOKENS)
    log_no = _best_of(row, candidates, NO_TOKENS)
    if not np.isfinite(log_yes) and not np.isfinite(log_no):
        return float("nan")
    top = max(log_yes, log_no)
    yes = np.exp(log_yes - top) if np.isfinite(log_yes) else 0.0
    no = np.exp(log_no - top) if np.isfinite(log_no) else 0.0
    return float(yes / (yes + no))


# ── The readouts ────────────────────────────────────────────────────────────────


QUESTION_PROMPT = (
    "You are an expert radiologist reading this chest X-ray. Answer each question "
    "about it with a single word: yes or no."
)
QUESTION_TEMPLATE = "\n\nQuestion: Is {label} present in this chest X-ray?\nAnswer:"

# The `xrv` label names are terse and, in places, dataset jargon ("Pleural_Thickening"
# with an underscore, "Effusion", the NIH-era "Infiltration"). A medical model answers a
# natural radiological phrasing more reliably, so each label maps to the phrase used *in
# the question* — the schema and the cached columns stay `xrv`, only the wording changes.
# This is a modelling choice (how the opinion is elicited, thesis §6.6),
# held identical across MedGemma variants for comparability and versioned with this
# readout; a label not listed (a non-xrv schema) falls back to its own name.
CLINICAL_PHRASING: dict[str, str] = {
    "Atelectasis": "atelectasis",
    "Consolidation": "consolidation",
    "Infiltration": "a pulmonary infiltrate",
    "Pneumothorax": "a pneumothorax",
    "Edema": "pulmonary edema",
    "Emphysema": "emphysema",
    "Fibrosis": "pulmonary fibrosis",
    "Effusion": "a pleural effusion",
    "Pneumonia": "pneumonia",
    "Pleural_Thickening": "pleural thickening",
    "Cardiomegaly": "cardiomegaly",
    "Nodule": "a pulmonary nodule",
    "Mass": "a pulmonary mass",
    "Hernia": "a hiatal hernia",
    "Lung Lesion": "a lung lesion",
    "Fracture": "a fracture",
    "Lung Opacity": "a lung opacity",
    "Enlarged Cardiomediastinum": "a widened mediastinum",
}


def clinical_phrase(label: str) -> str:
    """The natural-language phrase for a label in the question, or the label unchanged."""
    return CLINICAL_PHRASING.get(label, label)


@dataclass(frozen=True)
class IndependentQuestionReadout:
    """Ask about each pathology **separately**, off a shared reading of the image.

    This is the default, and the reason is measured rather than assumed. Its predecessor
    `TeacherForcedLogprobReadout` puts every pathology in one checklist, which makes the
    answer at position *k* conditional on the answers forced before it. On real chest
    X-rays that dependence dominates the image: forcing all-yes versus all-no moved the
    mean answer from 0.875 to 0.099 — a gap of 0.78 on a 0-1 scale — and the gap grew
    with position (correlation +0.48), while position 0, which has nothing before it,
    was identical under both. The checklist was largely measuring its own filler.

    Here each question is its own branch, so no pathology can see another's answer. The
    cost objection is answered by sharing work rather than by accepting contamination:
    encoding the image (896x896 into 256 tokens) and reading the instruction dominate,
    and both are identical for every question, so the backend does them **once** and
    rewinds between branches. Eighteen independent questions cost roughly one pass and a
    half, not eighteen.
    """

    name: str = "independent_question"

    def probabilities(
        self,
        backend: VisionLanguageBackend,
        image: Any,
        labels: Sequence[str],
        prefix: str = "",
    ) -> np.ndarray:
        branches = [QUESTION_TEMPLATE.format(label=clinical_phrase(label)) for label in labels]
        scores = backend.score_branches(
            image, prefix + QUESTION_PROMPT, branches, (*YES_TOKENS, *NO_TOKENS)
        )
        return np.array(
            [_yes_probability(row, scores.candidates) for row in scores.candidate_logprobs],
            dtype=float,
        )


@dataclass(frozen=True)
class TeacherForcedLogprobReadout:
    """One forced checklist for all pathologies. **Superseded — kept as a diagnostic.**

    Builds a checklist, one line per pathology, forces an answer into every slot and
    reads what the model would have put there. One forward pass, and an answer for every
    label; that was the appeal.

    ⚠️ **Do not use this to produce results.** The dependence it introduces is not a
    small bias, it is the dominant term. Measured on real chest X-rays, forcing all-yes
    versus all-no moved the mean answer from 0.875 to 0.099 (mean absolute gap 0.78 on a
    0-1 scale); the gap correlated +0.48 with position, and position 0 — the only slot
    with nothing before it — was identical under both fillers. The checklist mostly
    measures its own filler.

    ⚠️ ``both_polarities=True`` does **not** fix this, despite being written for that
    purpose. Averaging an answer that saturates at 0.875 with one that saturates at 0.099
    yields the midpoint of two artefacts, not a measurement.

    Use `IndependentQuestionReadout`. This one stays because reproducing the failure is
    how the choice is justified, and because the yes/no gap is a usable measure of how
    suggestible a given model is.
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
