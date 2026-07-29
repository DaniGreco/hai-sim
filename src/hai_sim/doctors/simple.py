"""Trivial doctor twins: controllable, ground-truth-based baselines.

These are *simulated readers*, not image models. They derive their decision
from the patient's ground truth (injected by the engine into the context) and
corrupt it at a configured error rate. This makes them a clean, fully
controllable baseline to swap against the (future) signal-detection twin — the
whole point of the modular design.

``FixedAccuracyDoctor`` is the simplest such reader: per label it keeps the true
value with probability ``sensitivity`` (when truth = 1) or ``specificity``
(when truth = 0), otherwise it flips. Sensitivity and specificity may be a
single scalar (applied to every label) or a per-label vector.

**The signal plane is what stops this reader being blind to the case.** With flat
accuracies it draws its errors from a coin that has never seen the image, which pins
the human-AI error correlation at zero no matter what a scenario asks for — the
measured consequence being that the AI never once flipped a senior's call over 1000
reads (``../TESI.md`` §8.1). Wiring ``difficulty_slope`` to the engine's
``sig.difficulty`` makes accuracy fall on hard cases, and when the difficulty model is
the AI's own uncertainty, human and machine errors concentrate on the same images.
That is a *coupling*, and it is the first version of this reader that can produce one.

⚠️ **It is still not a signal-detection observer.** Accuracy now varies with the case,
but the reported confidence is a parameter rather than a belief formed from evidence:
there is no latent decision variable, so over-confidence on hard cases has to be
configured rather than emerging. §8.1's `SDTDoctor` is what fixes that; this is the
honest intermediate, and any result from it should say which of the two produced it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hai_sim.core.base import BaseDiagnosticActor
from hai_sim.core.types import Image, LabelSchema, ModelOutput, PatientContext
from hai_sim.doctors._shared import lognormal_service_time, signal_penalty


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

    ``difficulty_slope`` and ``fatigue_slope`` are how much accuracy is lost at a
    signal of 1.0. Both degrade sensitivity and specificity *equally*, which in
    signal-detection terms is a reduction in discriminability ``d'`` with the criterion
    held fixed. A pure criterion shift — reading more or less liberally without seeing
    any better — is the other half of what Skinner & Giesbrecht (2025) report for the
    vigilance decrement, and it is not representable here: this reader has no criterion
    to shift. Another reason §8.1 supersedes it.
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
        difficulty_slope: float = 0.0,
        fatigue_slope: float = 0.0,
        floor: float = 0.5,
        service_time_model: object | None = None,
    ) -> None:
        super().__init__(name=name)
        self.schema = schema
        self.rng = rng if rng is not None else np.random.default_rng()
        self.sensitivity = self._as_vector(sensitivity, schema.n, "sensitivity")
        self.specificity = self._as_vector(specificity, schema.n, "specificity")
        self.service_time_mu = service_time_mu
        self.service_time_sigma = service_time_sigma
        if difficulty_slope < 0 or fatigue_slope < 0:
            raise ValueError("difficulty_slope and fatigue_slope must be >= 0")
        if not 0.0 <= floor <= 1.0:
            raise ValueError(f"floor must be in [0, 1], got {floor}")
        self.difficulty_slope = float(difficulty_slope)
        self.fatigue_slope = float(fatigue_slope)
        # Accuracy never falls below `floor`. At 0.5 the reader degrades to a coin
        # flip and no further: a simulated doctor that becomes *anti*-correlated with
        # the truth on hard cases is an artefact of unbounded arithmetic, not a model
        # of anything, and it would silently invert every complementarity result.
        self.floor = float(floor)
        self.service_time_model = service_time_model

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

        sensitivity, specificity, penalty = self._effective_accuracy(context)

        labels: list[int] = []
        confidence: list[float] = []
        draws = self.rng.random(n)
        for i, g in enumerate(gt):
            if g == 1:
                correct = draws[i] < sensitivity[i]
                pred = 1 if correct else 0
                conf = float(sensitivity[i])
            elif g == 0:
                correct = draws[i] < specificity[i]
                pred = 0 if correct else 1
                conf = float(1.0 - specificity[i])
            else:  # g == -1: not annotated; emit a benign negative.
                pred = 0
                conf = 0.0
            labels.append(pred)
            confidence.append(conf)

        uncertainty = [_binary_entropy(c) for c in confidence]
        extra: dict[str, object] = {"_trivial_doctor": True}
        if penalty > 0.0:
            # Published so a study can check the coupling actually bit, rather than
            # inferring it from an outcome. A run where this stays 0 has a difficulty
            # model configured but a reader that ignores it.
            extra["accuracy_penalty"] = penalty
        return ModelOutput.from_arrays(labels, confidence, uncertainty, extra=extra)

    def _effective_accuracy(
        self, context: PatientContext | None
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Apply the signal plane's difficulty and fatigue to this read's accuracy."""
        penalty = signal_penalty(context, self.difficulty_slope, self.fatigue_slope)
        if penalty <= 0.0:
            return self.sensitivity, self.specificity, 0.0
        return (
            np.maximum(self.sensitivity - penalty, self.floor),
            np.maximum(self.specificity - penalty, self.floor),
            float(penalty),
        )

    def service_time(
        self,
        images: Sequence[Image],
        context: PatientContext | None = None,
    ) -> float:
        return lognormal_service_time(
            self.rng,
            self.service_time_mu,
            self.service_time_sigma,
            model=self.service_time_model,
            images=images,
            context=context,
        )
