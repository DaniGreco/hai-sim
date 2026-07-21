"""AI-assisted doctor: own opinion + a swappable ``IntegrationPolicy``.

AI = artificial intelligence (here: the automated image reader).

This is the second half of the *in-series* arrangement: the AI has already read the
image, and now the doctor reads it while seeing what the AI concluded. An
`AssistedDoctor` wraps a *base* doctor — its own, AI-unaided opinion (e.g.
`FixedAccuracyDoctor`) — and an `IntegrationPolicy` that fuses that opinion with the AI
``ai_hint`` into the doctor's final, aided decision.

**The base opinion is always formed first, unaided.** That ordering is not incidental:
keeping the solo verdict alongside the aided one is what lets the metrics layer ask
whether the advice was taken *when it should have been*, instead of only whether the
final answer was right. Design and literature: ``../TESI.md`` §4.5, §6.6.

Policies are interchangeable — a fixed-weight blend, a confidence gate, log-odds
fusion, later a trust that moves with experience — and swapping one touches neither the
doctor nor the engine.

**Belief probabilities.** Fusion needs each actor's per-label ``P(label=1)``:

- The AI emits a genuine probability (ensemble mean); use it directly.
- A *simulated reader* (`FixedAccuracyDoctor`) emits a decision whose noise lives
  in the label, not the confidence (its ``confidence`` reports ``P(positive)`` but
  is decoupled from the realized, possibly-flipped label). To make the fusion
  carry that decision noise, orient the confidence *magnitude* by the emitted
  label: ``strength = max(c, 1-c)`` (the reliability), then
  ``p = strength if label==1 else 1-strength``. Thresholding ``p`` at 0.5 then
  reproduces the label exactly, and a label-flip becomes a *confidently wrong*
  belief — the behaviourally correct input to the fuser.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from hai_sim.core.base import BaseDiagnosticActor
from hai_sim.core.types import Diagnosis, Image, ModelOutput, PatientContext

if TYPE_CHECKING:
    from hai_sim.core.protocols import DiagnosticActor, IntegrationPolicy

_EPS = 1e-6


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) of a Bernoulli(p), elementwise."""
    p = np.clip(p, _EPS, 1.0 - _EPS)
    h: np.ndarray = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return h


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    out: np.ndarray = np.log(p / (1.0 - p))
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = 1.0 / (1.0 + np.exp(-x))
    return out


def doctor_belief(output: ModelOutput) -> np.ndarray:
    """Per-label ``P(label=1)`` for a *decision-with-reliability* reader.

    Orients the confidence magnitude by the emitted (noisy) label so the belief is
    label-consistent and carries the decision noise (see module docstring).
    """
    labels = np.asarray(output.diagnosis.labels)
    conf = np.asarray(output.diagnosis.confidence, dtype=float)
    strength = np.maximum(conf, 1.0 - conf)  # reliability magnitude in [0.5, 1]
    p = np.where(labels == 1, strength, 1.0 - strength)
    belief: np.ndarray = np.clip(p, _EPS, 1.0 - _EPS)
    return belief


def ai_probability(output: ModelOutput) -> np.ndarray:
    """Per-label ``P(label=1)`` for the AI: its confidence is already a probability.

    ``NaN`` is preserved (an uncovered label no member can score) so a policy can
    down-weight it to zero rather than fabricate a value.
    """
    return np.asarray(output.diagnosis.confidence, dtype=float)


def _team_output(
    p_team: np.ndarray,
    doctor_out: ModelOutput,
    policy_name: str,
    weight_ai: np.ndarray,
) -> ModelOutput:
    """Assemble the fused `ModelOutput`, stashing the doctor-solo counterfactual.

    ``extra["doctor_solo_labels"]`` is what the collector/KPI layer reads to keep
    the unaided-doctor counterfactual alongside the aided (team) verdict.
    """
    labels = tuple(int(x) for x in (p_team >= 0.5))
    extra: dict[str, Any] = {
        "assisted": True,
        "policy": policy_name,
        "ai_hint_used": True,
        "doctor_solo_labels": tuple(int(x) for x in doctor_out.diagnosis.labels),
        "doctor_solo_confidence": tuple(float(x) for x in doctor_out.diagnosis.confidence),
        "fusion_weight_ai": tuple(float(x) for x in weight_ai),
    }
    return ModelOutput(
        diagnosis=Diagnosis(
            labels=labels,
            confidence=tuple(float(x) for x in p_team),
        ),
        uncertainty=tuple(float(x) for x in _binary_entropy(p_team)),
        extra=extra,
    )


class LinearPoolPolicy:
    """Baseline trust: blend the two opinions in probability space, at a fixed weight.

    ``p_team = (1-w)·p_doctor + w·p_ai``. ``w`` is the **weight of advice (WOA)**: how
    much of an advisor's opinion a judge adopts, 0 = ignored, 1 = adopted wholesale. The
    human advice-taking literature puts its pooled average at ≈ 0.39.

    Chosen as the baseline because a blend **cannot become more confident than the more
    confident of the two readers**. That makes it tolerant of an AI whose probabilities
    are over-stated — unlike `LogOddsFusionPolicy`, which such an AI can dominate.
    """

    def __init__(self, weight_ai: float = 0.4, name: str = "linear_pool") -> None:
        if not 0.0 <= weight_ai <= 1.0:
            raise ValueError(f"weight_ai must be in [0, 1], got {weight_ai}")
        self.weight_ai = float(weight_ai)
        self.name = name

    def combine(
        self,
        doctor_out: ModelOutput,
        ai_hint: ModelOutput,
        context: PatientContext | None = None,
    ) -> ModelOutput:
        p_doctor = doctor_belief(doctor_out)
        p_ai = ai_probability(ai_hint)
        uncovered = ~np.isfinite(p_ai)  # a label no ensemble member scored → doctor only
        w_ai = np.where(uncovered, 0.0, self.weight_ai)
        p_ai = np.where(uncovered, 0.0, p_ai)
        p_team = (1.0 - w_ai) * p_doctor + w_ai * p_ai
        return _team_output(p_team, doctor_out, self.name, w_ai)


class ConfidenceGatedPoolPolicy:
    """Conditional trust: listen to the AI only on the labels where it sounds sure.

    Per label, the AI's confidence in its own call is ``max(p_ai, 1-p_ai)`` — distance
    from the 0.5 fence, not the probability itself. At or above
    ``confidence_threshold`` the doctor blends toward the AI at ``weight_ai``, exactly
    as `LinearPoolPolicy` would; below it the AI is ignored outright and the doctor
    keeps its own opinion.

    Grounded in the advice-taking literature, where uptake rises with how confident the
    advisor *appears* (Sniezek & Van Swol 2001) — note that this tracks displayed
    confidence, which is only as trustworthy as the model's calibration.
    """

    def __init__(
        self,
        weight_ai: float = 0.4,
        confidence_threshold: float = 0.7,
        name: str = "confidence_gated",
    ) -> None:
        if not 0.0 <= weight_ai <= 1.0:
            raise ValueError(f"weight_ai must be in [0, 1], got {weight_ai}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
            )
        self.weight_ai = float(weight_ai)
        self.confidence_threshold = float(confidence_threshold)
        self.name = name

    def combine(
        self,
        doctor_out: ModelOutput,
        ai_hint: ModelOutput,
        context: PatientContext | None = None,
    ) -> ModelOutput:
        p_doctor = doctor_belief(doctor_out)
        p_ai = ai_probability(ai_hint)
        uncovered = ~np.isfinite(p_ai)
        ai_confidence = np.maximum(p_ai, 1.0 - p_ai)  # confidence in its own call
        heeded = np.isfinite(p_ai) & (ai_confidence >= self.confidence_threshold)
        w_ai = np.where(heeded, self.weight_ai, 0.0)
        p_ai = np.where(uncovered, 0.0, p_ai)
        p_team = (1.0 - w_ai) * p_doctor + w_ai * p_ai
        return _team_output(p_team, doctor_out, self.name, w_ai)


class LogOddsFusionPolicy:
    """Evidence-adding trust: sum the two opinions on the log-odds scale, weighted by
    how uncertain the AI is.

    Each reader contributes a **log-likelihood ratio (LLR)** — its evidence relative to
    the prior, ``LLR = logit(p) - logit(prior)``:

        logit(p_team) = logit(prior) + w_doctor·LLR_doctor + w_ai(x)·LLR_ai
        w_ai(x)       = exp(-beta · H_ai)

    ``H_ai`` is the AI's own uncertainty on that label, so the AI's vote is discounted
    exactly where it is unsure. With ``prior=0.5, w_doctor=1, beta=0`` this collapses to
    ``logit(p_team) = logit(p_doctor) + logit(p_ai)`` — the optimal way to combine two
    readers whose errors are independent.

    ⚠️ **This one assumes the AI's probabilities mean what they say.** Adding log-odds
    gives an over-confident AI an unbounded voice, so an uncalibrated model can drag the
    team *below* the doctor alone. Calibration (temperature scaling) is not implemented
    yet — see ``../TESI.md`` §8.2, which flags this as blocking for this policy. Prefer
    `LinearPoolPolicy` until then.
    """

    def __init__(
        self,
        beta: float = 1.0,
        weight_doctor: float = 1.0,
        prior: float = 0.5,
        name: str = "log_odds_fusion",
    ) -> None:
        if beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        if weight_doctor < 0.0:
            raise ValueError(f"weight_doctor must be >= 0, got {weight_doctor}")
        if not 0.0 < prior < 1.0:
            raise ValueError(f"prior must be in (0, 1), got {prior}")
        self.beta = float(beta)
        self.weight_doctor = float(weight_doctor)
        self.prior = float(prior)
        self.name = name

    def combine(
        self,
        doctor_out: ModelOutput,
        ai_hint: ModelOutput,
        context: PatientContext | None = None,
    ) -> ModelOutput:
        p_doctor = doctor_belief(doctor_out)
        p_ai = ai_probability(ai_hint)
        ai_uncertainty = np.asarray(ai_hint.uncertainty, dtype=float)

        logit_prior = _logit(np.full(p_doctor.shape, self.prior))
        llr_doctor = _logit(p_doctor) - logit_prior

        # Drop the AI on labels it cannot score (a NaN probability or uncertainty);
        # elsewhere discount its evidence by exp(-beta * uncertainty). The nested
        # np.where keeps NaNs out of exp() and logit(), which would propagate them.
        uncovered = ~np.isfinite(p_ai) | ~np.isfinite(ai_uncertainty)
        w_ai = np.where(
            uncovered, 0.0, np.exp(-self.beta * np.where(uncovered, 0.0, ai_uncertainty))
        )
        llr_ai = np.where(
            uncovered, 0.0, _logit(np.where(uncovered, 0.5, p_ai)) - logit_prior
        )

        p_team = _sigmoid(logit_prior + self.weight_doctor * llr_doctor + w_ai * llr_ai)
        return _team_output(p_team, doctor_out, self.name, w_ai)


class AssistedDoctor(BaseDiagnosticActor):
    """A doctor that forms its own opinion, then integrates the AI hint via a policy.

    ``diagnose`` runs the *base* doctor (unaided) to get its own opinion, then —
    if an ``ai_hint`` is present — applies the `IntegrationPolicy` to produce the
    aided, final decision. With no hint it returns the base opinion unchanged, so
    an assisted scenario degrades gracefully to solo reading.
    """

    def __init__(
        self,
        base: DiagnosticActor,
        policy: IntegrationPolicy,
        name: str = "AssistedDoctor",
    ) -> None:
        super().__init__(name=name)
        self.base = base
        self.policy = policy

    def diagnose(
        self,
        images: Sequence[Image],
        *,
        context: PatientContext | None = None,
        ai_hint: ModelOutput | None = None,
    ) -> ModelOutput:
        own = self.base.diagnose(images, context=context)
        if ai_hint is None:
            return own
        return self.policy.combine(own, ai_hint, context)

    def service_time(
        self,
        images: Sequence[Image],
        context: PatientContext | None = None,
    ) -> float:
        return self.base.service_time(images, context)
