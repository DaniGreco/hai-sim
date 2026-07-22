"""Eliciting a probability per pathology from a model that only emits text.

Everything here runs against a fake vision-language model returning canned numbers, so
the logic that turns text into probabilities is testable without a multi-gigabyte gated
download — which is the whole reason `VisionLanguageBackend` exists as a seam.
"""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.core.types import LabelSchema
from hai_sim.models.medgemma import (
    MedGemmaAI,
    _contiguous_slices,
    _single_token_candidates,
    to_pil,
)
from hai_sim.models.readout import (
    ContinuationScores,
    GeneratedJsonReadout,
    IndependentQuestionReadout,
    TeacherForcedLogprobReadout,
    answer_token_positions,
    build_checklist,
    parse_confidence_json,
    resolve_prompt_set,
)
from hai_sim.uq.methods import EnsembleUQ

_LABELS = ("Effusion", "Pneumothorax", "Nodule")


class FakeVisionLanguage:
    """A stand-in model: fixed P(yes) per label, tokenised into fixed-size chunks.

    Chunking rather than one-character tokens on purpose — a token that spans several
    characters is what makes the offset-to-token mapping non-trivial, and that mapping is
    the part most likely to break silently.
    """

    def __init__(self, p_yes, *, chunk: int = 3, answerable: bool = True, reply: str = ""):
        self.p_yes = list(p_yes)
        self.chunk = chunk
        self.answerable = answerable
        self.reply = reply
        self.prompts_seen: list[str] = []
        self.continuations_seen: list[str] = []

    def score_continuation(self, image, prompt, continuation, candidates):
        self.prompts_seen.append(prompt)
        self.continuations_seen.append(continuation)
        texts = [
            continuation[i : i + self.chunk] for i in range(0, len(continuation), self.chunk)
        ]
        kept = ("yes", "no") if self.answerable else ()
        table = np.full((len(texts), len(kept)), -np.inf)
        if self.answerable:
            # Answer k starts right after the k-th ": ". With fixed-size chunks the token
            # covering character c is simply c // chunk — computed here independently of
            # the function under test.
            starts = [i + 2 for i in _find_all(continuation, ": ")]
            for k, start in enumerate(starts):
                p = self.p_yes[k]
                table[start // self.chunk] = [np.log(max(p, 1e-12)), np.log(max(1 - p, 1e-12))]
        return ContinuationScores(
            token_texts=tuple(texts), candidates=kept, candidate_logprobs=table
        )

    def score_branches(self, image, prompt, branches, candidates):
        self.prompts_seen.append(prompt)
        self.branches_seen = list(branches)
        kept = ("yes", "no") if self.answerable else ()
        table = np.full((len(branches), len(kept)), -np.inf)
        if self.answerable:
            for k in range(len(branches)):
                p = self.p_yes[k]
                table[k] = [np.log(max(p, 1e-12)), np.log(max(1 - p, 1e-12))]
        from hai_sim.models.readout import BranchScores

        return BranchScores(tuple(branches), kept, table)

    def generate(self, image, prompt, max_new_tokens):
        self.prompts_seen.append(prompt)
        return self.reply


def _find_all(haystack: str, needle: str) -> list[int]:
    out, start = [], 0
    while (i := haystack.find(needle, start)) != -1:
        out.append(i)
        start = i + 1
    return out


# ── The forced checklist and its offsets ───────────────────────────────────────


def test_checklist_offsets_point_at_the_answers():
    text, offsets = build_checklist(("A", "B"), "yes")
    assert text == "A: yes\nB: yes\n"
    assert [text[o : o + 3] for o in offsets] == ["yes", "yes"]


def test_offsets_map_onto_the_tokens_covering_them():
    # tokens tile "abcdefgh" as abc|def|gh
    positions = answer_token_positions(("abc", "def", "gh"), [0, 3, 4, 7])
    assert positions == [0, 1, 1, 2]


def test_offset_past_the_end_is_reported_not_guessed():
    """A continuation tokenised shorter than expected must yield "no answer", never a
    silently wrong slot."""
    assert answer_token_positions(("abc",), [99]) == [-1]


def test_contiguous_slices_tile_the_string_even_with_offset_gaps():
    # a tokenizer that skips the space between the two words
    slices = _contiguous_slices("ab cd", [(0, 2), (3, 5)])
    assert "".join(slices) == "ab cd"


def test_contiguous_slices_absorb_a_trailing_remainder():
    slices = _contiguous_slices("abcd", [(0, 2)])
    assert "".join(slices) == "abcd"


# ── The default readout: one independent question per pathology ────────────────


def test_independent_questions_come_back_in_schema_order():
    backend = FakeVisionLanguage([0.9, 0.1, 0.5])
    got = IndependentQuestionReadout().probabilities(backend, None, _LABELS)
    assert np.allclose(got, [0.9, 0.1, 0.5], atol=1e-6)


def test_one_branch_per_pathology_each_naming_only_its_own():
    """The contamination the checklist suffered is impossible here by construction: a
    branch mentions one pathology and cannot see any other branch's answer."""
    backend = FakeVisionLanguage([0.5, 0.5, 0.5])
    IndependentQuestionReadout().probabilities(backend, None, _LABELS)
    assert len(backend.branches_seen) == len(_LABELS)
    for branch, label in zip(backend.branches_seen, _LABELS, strict=True):
        assert label in branch
        assert not any(other in branch for other in _LABELS if other != label)


def test_the_image_is_read_once_for_all_the_questions():
    """Eighteen questions must not cost eighteen readings of the image — that is what
    makes asking them separately affordable at all."""
    backend = FakeVisionLanguage([0.5, 0.5, 0.5])
    IndependentQuestionReadout().probabilities(backend, None, _LABELS, prefix="P. ")
    assert len(backend.prompts_seen) == 1
    assert backend.prompts_seen[0].startswith("P. ")


def test_independent_readout_reports_no_opinion_as_nan():
    backend = FakeVisionLanguage([0.9], answerable=False)
    got = IndependentQuestionReadout().probabilities(backend, None, ("Effusion",))
    assert np.isnan(got[0])


# ── Teacher-forced log-probability readout (superseded, kept as a diagnostic) ───


def test_probabilities_come_back_in_schema_order():
    backend = FakeVisionLanguage([0.9, 0.1, 0.5])
    got = TeacherForcedLogprobReadout().probabilities(backend, None, _LABELS)
    assert np.allclose(got, [0.9, 0.1, 0.5], atol=1e-6)


def test_probability_is_renormalised_over_yes_and_no_only():
    """Only the two answers compete. Renormalising against the whole vocabulary would
    fold in the model's inclination to write prose instead of answering."""
    backend = FakeVisionLanguage([0.25])
    got = TeacherForcedLogprobReadout().probabilities(backend, None, ("Effusion",))
    assert got[0] == pytest.approx(0.25, abs=1e-6)


def test_no_opinion_stays_nan_rather_than_becoming_a_confident_zero():
    backend = FakeVisionLanguage([0.9], answerable=False)
    got = TeacherForcedLogprobReadout().probabilities(backend, None, ("Effusion",))
    assert np.isnan(got[0])


def test_the_prompt_lists_every_label_and_carries_the_prefix():
    backend = FakeVisionLanguage([0.5, 0.5, 0.5])
    TeacherForcedLogprobReadout().probabilities(backend, None, _LABELS, prefix="PREFIX. ")
    prompt = backend.prompts_seen[0]
    assert prompt.startswith("PREFIX. ")
    for label in _LABELS:
        assert label in prompt


def test_both_polarities_runs_each_forcing_and_averages():
    """The forced answers bias the ones after them, so the option exists to run the
    all-yes and all-no checklists and average. Two passes, two continuations."""
    backend = FakeVisionLanguage([0.8, 0.2, 0.6])
    got = TeacherForcedLogprobReadout(both_polarities=True).probabilities(
        backend, None, _LABELS
    )
    assert len(backend.continuations_seen) == 2
    assert "yes" in backend.continuations_seen[0]
    assert "no" in backend.continuations_seen[1]
    assert np.allclose(got, [0.8, 0.2, 0.6], atol=1e-6)  # fake is polarity-independent


# ── The JSON readout, and why it is a different quantity ───────────────────────


def test_json_is_extracted_from_surrounding_prose():
    parsed = parse_confidence_json('Sure! Here it is:\n```json\n{"Effusion": 0.7}\n```')
    assert parsed == {"Effusion": 0.7}


def test_percentages_are_rescaled():
    assert parse_confidence_json('{"Effusion": 70}') == {"Effusion": 0.7}


def test_unusable_replies_yield_nothing_rather_than_a_guess():
    assert parse_confidence_json("I cannot help with that.") == {}
    assert parse_confidence_json("{oops") == {}
    assert parse_confidence_json("[1, 2, 3]") == {}
    assert parse_confidence_json('{"Effusion": "high"}') == {}


def test_json_readout_maps_onto_the_schema_case_insensitively():
    backend = FakeVisionLanguage([], reply='{"effusion": 0.8, "NODULE": 0.2}')
    got = GeneratedJsonReadout().probabilities(backend, None, _LABELS)
    assert got[0] == pytest.approx(0.8)
    assert got[2] == pytest.approx(0.2)


def test_unmentioned_finding_is_absent_by_default_but_can_stay_unknown():
    """Reports state what was seen, so silence normally means absent — but that is a
    convention, and counting silence as a confident negative is sometimes wrong."""
    backend = FakeVisionLanguage([], reply='{"Effusion": 0.8}')
    assert GeneratedJsonReadout().probabilities(backend, None, _LABELS)[1] == 0.0
    lenient = GeneratedJsonReadout(missing_is_negative=False)
    assert np.isnan(lenient.probabilities(backend, None, _LABELS)[1])


# ── Prompt ensembling reuses the existing uncertainty machinery ────────────────


def test_prompt_set_resolves_and_rejects_unknown_names():
    assert resolve_prompt_set("single") == ("",)
    assert len(resolve_prompt_set("triple")) == 3
    with pytest.raises(ValueError, match="unknown prompt_set"):
        resolve_prompt_set("nope")


def test_paraphrase_disagreement_becomes_epistemic_uncertainty():
    """The point of prompt ensembling: K wordings are K members, so the *existing*
    EnsembleUQ decomposition applies to a generative model unchanged."""
    agreeing = np.array([[0.9, 0.5], [0.9, 0.5]])  # wording changes nothing
    disagreeing = np.array([[0.95, 0.5], [0.05, 0.5]])  # wording flips the answer
    uq = EnsembleUQ(measure="mutual_information")
    _p, epi_agree, _e = uq.estimate(agreeing)
    _p, epi_disagree, extra = uq.estimate(disagreeing)

    assert epi_disagree[0] > epi_agree[0]  # disagreement is epistemic
    assert epi_agree[0] == pytest.approx(0.0, abs=1e-9)
    # and the decomposition identity still holds: total = aleatoric + epistemic
    total = extra["predictive_entropy"][0]
    aleatoric = extra["mean_entropy"][0]
    assert total - aleatoric == pytest.approx(epi_disagree[0], abs=1e-9)


# ── Image conversion ───────────────────────────────────────────────────────────


def test_image_is_rescaled_from_its_own_range():
    """Datasets hand over arrays in their own units — TorchXRayVision uses [-1024, 1024].
    Assuming 0-255 would send the model a uniformly grey picture it would still report on."""
    pil = to_pil(np.array([[-1024.0, 0.0], [512.0, 1024.0]], dtype=np.float32))
    as_array = np.asarray(pil)
    assert as_array.shape == (2, 2, 3)  # RGB
    assert as_array[..., 0].min() == 0 and as_array[..., 0].max() == 255


def test_channel_first_and_flat_images_are_both_accepted():
    assert np.asarray(to_pil(np.zeros((1, 4, 4), dtype=np.float32))).shape == (4, 4, 3)
    assert np.asarray(to_pil(np.zeros((4, 4), dtype=np.float32))).shape == (4, 4, 3)


def test_a_constant_image_does_not_divide_by_zero():
    assert np.asarray(to_pil(np.full((3, 3), 7.0, dtype=np.float32))).max() == 0


def test_an_unexpected_shape_is_rejected():
    with pytest.raises(ValueError, match="expected a 2-D image"):
        to_pil(np.zeros((2, 3, 5), dtype=np.float32))


# ── Tokenizer candidate filtering ──────────────────────────────────────────────


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"yes": [1], " yes": [2], "no": [3]}.get(text, [4, 5])  # others split


def test_only_single_token_spellings_are_scored():
    """A spelling this tokenizer splits has no single position to read a probability
    from, so it is dropped — safe, because several spellings are always offered."""
    kept, columns = _single_token_candidates(_FakeTokenizer(), ("yes", " yes", "no", "Yes"))
    assert kept == ["yes", " yes", "no"]
    assert columns == [1, 2, 3]


# ── The actor's wiring, without weights ────────────────────────────────────────


def test_actor_builds_without_downloading_and_refuses_to_run_unloaded():
    ai = MedGemmaAI(schema=LabelSchema(labels=_LABELS), load_now=False)
    assert ai.name.startswith("MedGemmaAI(google/medgemma-1.5-4b-it")
    with pytest.raises(RuntimeError, match="not loaded"):
        ai.member_probabilities(np.zeros((4, 4), dtype=np.float32))


def test_a_single_prompt_gets_entropy_and_several_get_ensemble_uq():
    labels = LabelSchema(labels=_LABELS)
    assert MedGemmaAI(schema=labels, prompt_set="single", load_now=False).uq.name == "entropy"
    triple = MedGemmaAI(schema=labels, prompt_set="triple", load_now=False)
    assert triple.uq.name.startswith("ensemble")
