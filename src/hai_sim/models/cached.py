"""Cache-replay AI: real ensemble predictions from an `InferenceCache` (no re-inference).

For a study over personas / triage / threshold, the AI's predictions on each image are
identical across runs — only the doctor and the queue change. Re-running the 5-member
ensemble (~5-6 s/image) for every run would waste hours. `CachedEnsembleAI` instead serves
each image's cached per-member probabilities through the same swappable `UQMethod`,
reconstructing exactly the `ModelOutput` a live `TorchXRayVisionAI` would produce — real NIH
predictions, no torch, seconds not hours. The patient's cache index rides in
``context.patient_id`` (set by `CachedPatientGenerator`), so AI and generator stay aligned
as long as they load the same cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from hai_sim.core.base import BaseDiagnosticActor
from hai_sim.core.types import Diagnosis, Image, LabelSchema, ModelOutput, PatientContext
from hai_sim.uq.methods import EntropyUQ

if TYPE_CHECKING:
    from hai_sim.analysis.inference_cache import InferenceCache
    from hai_sim.core.protocols import UQMethod


class CachedEnsembleAI(BaseDiagnosticActor):
    """Serve cached per-member probabilities through a swappable UQ method."""

    def __init__(
        self,
        cache: InferenceCache,
        schema: LabelSchema,
        uq: UQMethod | None = None,
        decision_threshold: float = 0.5,
        service_time: float = 0.05,
        name: str | None = None,
    ) -> None:
        if tuple(cache.pathologies) != schema.labels:
            raise ValueError(
                "cache pathologies do not match the active LabelSchema; use label_schema: xrv"
            )
        resolved = name if name is not None else f"CachedEnsembleAI({'+'.join(cache.weights)})"
        super().__init__(name=resolved, service_time=service_time)
        self.cache = cache
        self.schema = schema
        self.uq: UQMethod = uq if uq is not None else EntropyUQ()
        self.decision_threshold = decision_threshold

    def diagnose(
        self,
        images: Sequence[Image],
        *,
        context: PatientContext | None = None,
        ai_hint: ModelOutput | None = None,
    ) -> ModelOutput:
        if context is None:
            raise ValueError("CachedEnsembleAI needs a PatientContext carrying the cache index")
        idx = int(context.patient_id)
        member_probs = np.asarray(self.cache.member_probs[idx], dtype=float)  # (M, L)
        probs, uncertainty, uq_extra = self.uq.estimate(member_probs)
        probs = np.nan_to_num(np.asarray(probs, dtype=float), nan=0.0)
        uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0)
        labels = (probs >= self.decision_threshold).astype(int)
        extra: dict[str, Any] = {"uq_method": self.uq.name, "cache_index": idx, **uq_extra}
        return ModelOutput(
            diagnosis=Diagnosis(
                labels=tuple(int(x) for x in labels),
                confidence=tuple(float(x) for x in probs),
            ),
            uncertainty=tuple(float(x) for x in uncertainty),
            extra=extra,
        )
