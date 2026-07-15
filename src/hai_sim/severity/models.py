"""Swappable per-patient severity models.

Severity is a *produced* signal, deliberately decoupled from its consumers: the
engine computes it after AI inference and logs it in telemetry whether or not any
downstream component (triage, KPI conditioning, cost) uses it.

The workhorse `ActivationSeverity` follows the decomposition

    severity = aggregate_l( criticality_l * activation_l(patient) )

- ``criticality_l`` is a *pre-assigned* per-label clinical weight (a config vector,
  NOT computed from the image): a pneumothorax outweighs a chronic cardiomegaly.
- ``activation_l`` is a per-image, per-label signal from a swappable source: the
  AI's probabilities / binary calls / uncertainty, or the ground-truth vector (an
  oracle severity used to *condition* metrics, not to drive decisions).
- the aggregator collapses the weighted vector to a scalar, or — in ``vector``
  mode — returns it per-label (for per-pathology triage).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from hai_sim.core.types import ModelOutput, Patient, PatientContext

ActivationSource = Literal["ai_probs", "ai_binary", "ai_uncertainty", "gt"]
Aggregator = Literal["mean", "max", "sum"]
Mode = Literal["scalar", "vector"]


class ConstantSeverity:
    """Baseline: the same severity for every patient (default 0.0)."""

    def __init__(self, value: float = 0.0, name: str | None = None) -> None:
        self.value = float(value)
        self.name = name or f"ConstantSeverity({self.value:g})"

    def score(
        self, *, patient: Patient, ai_output: ModelOutput | None, context: PatientContext
    ) -> float:
        return self.value


class ActivationSeverity:
    """``severity = aggregate_l( criticality_l * activation_l )`` — see module docstring."""

    def __init__(
        self,
        source: ActivationSource = "ai_probs",
        criticality: Sequence[float] | None = None,
        aggregator: Aggregator = "max",
        mode: Mode = "scalar",
        name: str | None = None,
    ) -> None:
        self.source: ActivationSource = source
        self.criticality = None if criticality is None else np.asarray(criticality, dtype=float)
        self.aggregator: Aggregator = aggregator
        self.mode: Mode = mode
        self.name = name or f"ActivationSeverity({source},{aggregator},{mode})"

    def _activation(self, patient: Patient, ai_output: ModelOutput | None) -> np.ndarray:
        if self.source == "gt":
            gt = np.asarray(patient.ground_truth, dtype=float)
            # sentinel -1 = not annotated for this label → contributes no severity.
            return np.where(gt < 0.0, 0.0, gt)
        if ai_output is None:
            raise ValueError(f"severity source {self.source!r} needs an ai_output")
        if self.source == "ai_probs":
            return np.asarray(ai_output.diagnosis.confidence, dtype=float)
        if self.source == "ai_binary":
            return np.asarray(ai_output.diagnosis.labels, dtype=float)
        if self.source == "ai_uncertainty":
            return np.asarray(ai_output.uncertainty, dtype=float)
        raise ValueError(f"unknown activation source: {self.source}")

    def score(
        self, *, patient: Patient, ai_output: ModelOutput | None, context: PatientContext
    ) -> float | np.ndarray:
        a = self._activation(patient, ai_output)
        if self.criticality is not None:
            if self.criticality.shape != a.shape:
                raise ValueError(
                    f"criticality len {self.criticality.shape} != activation len {a.shape}"
                )
            weighted = np.asarray(self.criticality * a, dtype=float)
        else:
            weighted = a
        if self.mode == "vector":
            return weighted
        if weighted.size == 0:
            return 0.0
        if self.aggregator == "mean":
            return float(np.mean(weighted))
        if self.aggregator == "max":
            return float(np.max(weighted))
        if self.aggregator == "sum":
            return float(np.sum(weighted))
        raise ValueError(f"unknown aggregator: {self.aggregator}")


class ImageQualitySeverity:
    """Severity as a function of image quality (a swappable proxy).

    ``laplacian_var`` = variance of a 4-neighbour discrete Laplacian → high for
    sharp images, low for blurred ones; ``intensity_std`` = pixel-intensity
    spread. The quality→severity *semantics* belong to the consumer; this returns
    the raw proxy scalar.
    """

    def __init__(
        self,
        metric: Literal["laplacian_var", "intensity_std"] = "laplacian_var",
        name: str | None = None,
    ) -> None:
        self.metric = metric
        self.name = name or f"ImageQualitySeverity({metric})"

    def score(
        self, *, patient: Patient, ai_output: ModelOutput | None, context: PatientContext
    ) -> float:
        img = np.asarray(patient.images[0], dtype=float)
        if img.ndim == 3:
            # (C, H, W) channel-first if a small leading axis, else (H, W, C).
            img = img.mean(axis=0) if img.shape[0] <= 4 else img.mean(axis=2)
        if self.metric == "intensity_std":
            return float(img.std())
        lap = (
            -4.0 * img
            + np.roll(img, 1, axis=0)
            + np.roll(img, -1, axis=0)
            + np.roll(img, 1, axis=1)
            + np.roll(img, -1, axis=1)
        )
        return float(lap.var())


class EmbeddingSeverity:
    """Severity as a scalar reduction of the model's penultimate embedding.

    Requires ``ai_output.embedding`` (set the AI's ``emit_embedding=True``; the
    factory does this automatically). A first, simple "function of the feature
    array": ``l2_norm`` or ``mean`` of the embedding. Returns 0.0 when no
    embedding is available.
    """

    def __init__(
        self, reduce: Literal["l2_norm", "mean"] = "l2_norm", name: str | None = None
    ) -> None:
        self.reduce = reduce
        self.name = name or f"EmbeddingSeverity({reduce})"

    def score(
        self, *, patient: Patient, ai_output: ModelOutput | None, context: PatientContext
    ) -> float:
        if ai_output is None or ai_output.embedding is None:
            return 0.0
        emb = np.asarray(ai_output.embedding, dtype=float)
        if self.reduce == "mean":
            return float(emb.mean())
        return float(np.linalg.norm(emb))
