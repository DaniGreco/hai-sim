"""Trivial doctor twins: controllable, ground-truth-based baselines.

These are *simulated readers*, not image models. They derive their decision
from the patient's ground truth (injected by the engine into the context) and
corrupt it at a configured error rate. This makes them a clean, fully
controllable baseline to swap against the (future) fatigue-aware composite twin
— the whole point of the modular design.

``FixedAccuracyDoctor`` is the simplest such reader: per label it keeps the true
value with probability ``sensitivity`` (when truth = 1) or ``specificity``
(when truth = 0), otherwise it flips. Sensitivity and specificity may be a
single scalar (applied to every label) or a per-label vector.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hai_sim.core.base import BaseDiagnosticActor
from hai_sim.core.types import Diagnosis, Image, LabelSchema, ModelOutput, PatientContext


def _binary_entropy(p: float) -> float:
    """Shannon entropy (nats) of a Bernoulli(p), used as an uncertainty proxy."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


class FixedAccuracyDoctor(BaseDiagnosticActor):
    """Reads the ground truth and flips each label at a fixed error rate.

    - On a label with truth 1: outputs 1 with prob ``sensitivity`` (a hit),
      else 0 (a false negative).
    - On a label with truth 0: outputs 0 with prob ``specificity`` (a correct
      rejection), else 1 (a false positive).
    - On a label with truth -1 (not annotated): outputs 0; metrics skip it.

    ``confidence`` is reported as the probability the doctor assigns to its own
    positive call, and ``uncertainty`` as the binary entropy of that confidence
    — so the existing sort strategies and KPIs work unchanged.
    """

    def __init__(
        self,
        schema: LabelSchema,
        rng: np.random.Generator | None = None,
        sensitivity: float | Sequence[float] = 0.9,
        specificity: float | Sequence[float] = 0.9,
        service_time_mu: float = 2.5,
        service_time_sigma: float = 0.4,
        name: str = "FixedAccuracyDoctor",
    ) -> None:
        super().__init__(name=name)
        self.schema = schema
        self.rng = rng if rng is not None else np.random.default_rng()
        self.sensitivity = self._as_vector(sensitivity, schema.n, "sensitivity")
        self.specificity = self._as_vector(specificity, schema.n, "specificity")
        self.service_time_mu = service_time_mu
        self.service_time_sigma = service_time_sigma

    @staticmethod
    def _as_vector(v: float | Sequence[float], n: int, what: str) -> np.ndarray:
        if isinstance(v, (int, float)):
            arr = np.full(n, float(v))
        else:
            arr = np.asarray(v, dtype=float)
            if arr.shape != (n,):
                raise ValueError(f"{what} length {arr.shape} != schema.n {n}")
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError(f"{what} must be in [0, 1]")
        return arr

    def diagnose(
        self,
        images: Sequence[Image],
        *,
        context: PatientContext | None = None,
        ai_hint: ModelOutput | None = None,
    ) -> ModelOutput:
        n = self.schema.n
        gt = context.ground_truth if context is not None else None
        if gt is None:
            raise ValueError(
                "FixedAccuracyDoctor requires ground_truth in the context. The "
                "engine injects it into the doctor stage; check the wiring."
            )
        if len(gt) != n:
            raise ValueError(f"ground_truth length {len(gt)} != schema.n {n}")

        labels: list[int] = []
        confidence: list[float] = []
        draws = self.rng.random(n)
        for i, g in enumerate(gt):
            if g == 1:
                correct = draws[i] < self.sensitivity[i]
                pred = 1 if correct else 0
                conf = float(self.sensitivity[i])
            elif g == 0:
                correct = draws[i] < self.specificity[i]
                pred = 0 if correct else 1
                conf = float(1.0 - self.specificity[i])
            else:  # g == -1: not annotated; emit a benign negative.
                pred = 0
                conf = 0.0
            labels.append(pred)
            confidence.append(conf)

        uncertainty = [_binary_entropy(c) for c in confidence]
        return ModelOutput(
            diagnosis=Diagnosis(
                labels=tuple(labels),
                confidence=tuple(confidence),
            ),
            uncertainty=tuple(uncertainty),
            extra={"_trivial_doctor": True},
        )

    def service_time(
        self,
        images: Sequence[Image],
        context: PatientContext | None = None,
    ) -> float:
        return float(self.rng.lognormal(self.service_time_mu, self.service_time_sigma))
