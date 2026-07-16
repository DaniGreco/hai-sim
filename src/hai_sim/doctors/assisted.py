"""AI-assisted doctor: own opinion + a swappable ``IntegrationPolicy``.

Runtime spine of the human-AI integration study (design and literature: see
``../TESI.md`` §4.5, §6.6). An `AssistedDoctor` wraps a *base*
doctor — its own, AI-unaided opinion (e.g. `FixedAccuracyDoctor`) — and an
`IntegrationPolicy` that fuses that opinion with the AI ``ai_hint`` into the
doctor's final, aided decision. Swap the policy (fixed-weight pool,
uncertainty-weighted log-odds fusion, later trust dynamics) without touching the
doctor or the engine.

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
    """M1 baseline: convex mix in probability space (fixed Weight of Advice).

    ``p_team = (1-w)·p_D + w·p_M``. The fused confidence cannot exceed the more
    confident reader, so it is robust to an over-confident (mis-calibrated) AI —
    the reason it is the calibration-tolerant baseline. ``w`` is the Weight of
    Advice; the advice-taking literature's empirical mean is ≈0.39.
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
        p_d = doctor_belief(doctor_out)
        p_m = ai_probability(ai_hint)
        uncovered = ~np.isfinite(p_m)  # an AI label no member scored → doctor only
        w = np.where(uncovered, 0.0, self.weight_ai)
        p_m = np.where(uncovered, 0.0, p_m)
        p_team = (1.0 - w) * p_d + w * p_m
        return _team_output(p_team, doctor_out, self.name, w)


class ConfidenceGatedPoolPolicy:
    """Confidence-gated Weight of Advice: listen to the AI only when it is confident.

    Per label, the AI's confidence in its own call is ``max(p_M, 1-p_M)``. Where that
    is at least ``confidence_threshold`` the reader blends toward the AI with
    ``weight_ai`` (a linear pool); below it the AI is ignored (weight 0) and the reader
    keeps its own opinion. Grounded in advice-taking: uptake rises with the advisor's
    displayed confidence (Sniezek & Van Swol 2001). A swappable `IntegrationPolicy`
    like the others — same seam, richer trust behaviour.
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
        p_d = doctor_belief(doctor_out)
        p_m = ai_probability(ai_hint)
        uncovered = ~np.isfinite(p_m)
        ai_confidence = np.maximum(p_m, 1.0 - p_m)  # confidence in the AI's own call
        gated_in = np.isfinite(p_m) & (ai_confidence >= self.confidence_threshold)
        w = np.where(gated_in, self.weight_ai, 0.0)
        p_m = np.where(uncovered, 0.0, p_m)
        p_team = (1.0 - w) * p_d + w * p_m
        return _team_output(p_team, doctor_out, self.name, w)


class LogOddsFusionPolicy:
    """M2 default: uncertainty-weighted log-odds (naïve-Bayes / SDT) fusion.

    ``logit(p_team) = logit(pi) + w_D*LLR_D + w_M(x)*LLR_M`` with
    ``LLR = logit(p) - logit(pi)`` and ``w_M(x) = exp(-beta*H_AI)`` -- the AI's vote is
    down-weighted on labels where its predictive entropy / mutual information
    ``H_AI`` is high. With ``pi=0.5, w_D=1, beta=0`` this is exactly
    ``logit(post) = logit(p_D) + logit(p_M)``, the optimal fusion of two
    conditionally-independent readers. Assumes the AI probabilities are reasonably
    calibrated — an over-confident AI dominates the log-odds sum (see §9 of the
    literature note).
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
        p_d = doctor_belief(doctor_out)
        p_m = ai_probability(ai_hint)
        h = np.asarray(ai_hint.uncertainty, dtype=float)

        lp = _logit(np.full(p_d.shape, self.prior))
        llr_d = _logit(p_d) - lp

        # Drop the AI on labels it cannot score (NaN prob or NaN uncertainty);
        # elsewhere weight its log-likelihood ratio by exp(-beta * uncertainty).
        uncovered = ~np.isfinite(p_m) | ~np.isfinite(h)
        w_m = np.where(uncovered, 0.0, np.exp(-self.beta * np.where(uncovered, 0.0, h)))
        llr_m = np.where(uncovered, 0.0, _logit(np.where(uncovered, 0.5, p_m)) - lp)

        p_team = _sigmoid(lp + self.weight_doctor * llr_d + w_m * llr_m)
        return _team_output(p_team, doctor_out, self.name, w_m)


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
