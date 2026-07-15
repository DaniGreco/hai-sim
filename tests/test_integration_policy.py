"""IntegrationPolicy fusion math + AssistedDoctor wiring (pure numpy, no torch)."""

from __future__ import annotations

import math

import numpy as np

from hai_sim.core.types import Diagnosis, LabelSchema, ModelOutput, PatientContext
from hai_sim.doctors.assisted import (
    AssistedDoctor,
    LinearPoolPolicy,
    LogOddsFusionPolicy,
    ai_probability,
    doctor_belief,
)
from hai_sim.doctors.simple import FixedAccuracyDoctor


def _out(labels, confidence, uncertainty=None) -> ModelOutput:
    if uncertainty is None:
        uncertainty = tuple(0.0 for _ in labels)
    return ModelOutput(
        diagnosis=Diagnosis(labels=tuple(labels), confidence=tuple(confidence)),
        uncertainty=tuple(uncertainty),
    )


# --- belief extraction ------------------------------------------------------


def test_doctor_belief_oriented_by_label_carries_noise():
    # The reliability MAGNITUDE (max(c,1-c)) is oriented by the emitted label, so a
    # label that disagrees with the confidence direction becomes a confident belief
    # in the label's direction (the "confidently wrong" case).
    out = _out(labels=(1, 0, 1, 0), confidence=(0.9, 0.1, 0.1, 0.9))
    p = doctor_belief(out)
    assert np.allclose(p, [0.9, 0.1, 0.9, 0.1], atol=1e-6)
    # thresholding the belief reproduces the emitted label exactly
    assert tuple((p >= 0.5).astype(int)) == out.diagnosis.labels


def test_doctor_belief_from_fixed_accuracy_matches_ground_truth_when_perfect():
    doc = FixedAccuracyDoctor(LabelSchema.default_v0(), sensitivity=1.0, specificity=1.0)
    ctx = PatientContext(patient_id=0, arrival_time=0.0, ground_truth=(1, 0, 1, 0, 1))
    p = doctor_belief(doc.diagnose([np.zeros((1, 1))], context=ctx))
    assert tuple((p >= 0.5).astype(int)) == (1, 0, 1, 0, 1)


def test_ai_probability_preserves_nan():
    p = ai_probability(_out(labels=(1, 0), confidence=(0.7, float("nan"))))
    assert p[0] == 0.7
    assert math.isnan(p[1])


# --- linear pool (M1) -------------------------------------------------------


def test_linear_pool_convex_mix():
    doc = _out(labels=(1, 0), confidence=(0.8, 0.2))  # belief [0.8, 0.2]
    ai = _out(labels=(1, 1), confidence=(0.4, 0.9))  # prob   [0.4, 0.9]
    out = LinearPoolPolicy(weight_ai=0.5).combine(doc, ai)
    assert np.allclose(out.diagnosis.confidence, [0.6, 0.55], atol=1e-6)
    assert out.diagnosis.labels == (1, 1)
    assert out.extra["doctor_solo_labels"] == (1, 0)
    assert out.extra["policy"] == "linear_pool"


def test_linear_pool_uncovered_ai_falls_back_to_doctor():
    doc = _out(labels=(1, 0), confidence=(0.9, 0.1))  # belief [0.9, 0.1]
    ai = _out(labels=(1, 0), confidence=(float("nan"), 0.9))
    out = LinearPoolPolicy(weight_ai=0.5).combine(doc, ai)
    # label 0: AI can't score it -> team == doctor belief (0.9)
    assert math.isclose(out.diagnosis.confidence[0], 0.9, abs_tol=1e-6)
    assert math.isclose(out.diagnosis.confidence[1], 0.5, abs_tol=1e-6)


# --- log-odds fusion (M2) ---------------------------------------------------


def test_log_odds_is_naive_bayes_with_uniform_prior():
    # beta=0 neutralizes the uncertainty weight, prior=0.5, w_D=1 ->
    # logit(team) = logit(p_D) + logit(p_M).
    doc = _out(labels=(1, 0), confidence=(0.8, 0.2))  # belief [0.8, 0.2]
    ai = _out(labels=(1, 1), confidence=(0.7, 0.6), uncertainty=(0.5, 0.5))
    out = LogOddsFusionPolicy(beta=0.0).combine(doc, ai)
    expected = [
        1 / (1 + math.exp(-(math.log(0.8 / 0.2) + math.log(0.7 / 0.3)))),
        1 / (1 + math.exp(-(math.log(0.2 / 0.8) + math.log(0.6 / 0.4)))),
    ]
    assert np.allclose(out.diagnosis.confidence, expected, atol=1e-6)


def test_log_odds_high_ai_uncertainty_reduces_ai_pull():
    doc = _out(labels=(0,), confidence=(0.1,))  # belief [0.1] (strongly negative)

    def make_ai(h):
        return _out(labels=(1,), confidence=(0.9,), uncertainty=(h,))

    low_unc = LogOddsFusionPolicy(beta=1.0).combine(doc, make_ai(0.0))
    high_unc = LogOddsFusionPolicy(beta=1.0).combine(doc, make_ai(5.0))
    # confident AI (low uncertainty) pulls the team toward positive more than an
    # uncertain AI does; with very high uncertainty the team stays near the doctor.
    assert low_unc.diagnosis.confidence[0] > high_unc.diagnosis.confidence[0]
    assert math.isclose(high_unc.diagnosis.confidence[0], 0.1, abs_tol=0.02)


def test_log_odds_uncovered_ai_leaves_doctor_belief():
    doc = _out(labels=(1, 0), confidence=(0.8, 0.2))  # belief [0.8, 0.2]
    ai = _out(labels=(1, 0), confidence=(float("nan"), 0.9), uncertainty=(0.1, 0.3))
    out = LogOddsFusionPolicy(beta=0.0).combine(doc, ai)
    assert math.isclose(out.diagnosis.confidence[0], 0.8, abs_tol=1e-6)


# --- AssistedDoctor wiring --------------------------------------------------


def _perfect_assisted(policy) -> tuple[AssistedDoctor, PatientContext]:
    base = FixedAccuracyDoctor(LabelSchema.default_v0(), sensitivity=1.0, specificity=1.0)
    ctx = PatientContext(patient_id=0, arrival_time=0.0, ground_truth=(1, 0, 1, 0, 1))
    return AssistedDoctor(base=base, policy=policy), ctx


def test_assisted_doctor_without_hint_returns_base_opinion():
    doc, ctx = _perfect_assisted(LogOddsFusionPolicy())
    out = doc.diagnose([np.zeros((1, 1))], context=ctx, ai_hint=None)
    assert out.diagnosis.labels == (1, 0, 1, 0, 1)  # base == ground truth when perfect
    assert "assisted" not in out.extra  # untouched base output


def test_assisted_doctor_fuses_and_stashes_solo_counterfactual():
    doc, ctx = _perfect_assisted(LogOddsFusionPolicy())
    ai = _out(
        labels=(0, 1, 1, 0, 1),
        confidence=(0.2, 0.8, 0.9, 0.1, 0.7),
        uncertainty=(0.1, 0.1, 0.1, 0.1, 0.1),
    )
    out = doc.diagnose([np.zeros((1, 1))], context=ctx, ai_hint=ai)
    assert out.extra["assisted"] is True
    assert out.extra["policy"] == "log_odds_fusion"
    assert out.extra["doctor_solo_labels"] == (1, 0, 1, 0, 1)
    assert len(out.diagnosis.labels) == 5
    assert doc.service_time([np.zeros((1, 1))], ctx) > 0.0
