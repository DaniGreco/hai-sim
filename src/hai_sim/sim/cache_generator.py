"""Patient stream that replays an `InferenceCache`.

Yields patients whose ground truth is the cached label vector and whose **cache index rides
in ``context.patient_id``** so a paired `CachedEnsembleAI` can look up the same image's
cached predictions. The image pixels are a placeholder: in a cache-replay study the AI is
served from the cache and the simulated reader reads ground truth, so nobody needs them.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

import numpy as np

from hai_sim.core.types import Image, Patient, PatientContext

if TYPE_CHECKING:
    from hai_sim.analysis.inference_cache import InferenceCache
    from hai_sim.core.types import LabelSchema

SampleMode = Literal["sequential", "random"]


class CachedPatientGenerator:
    """Yields `Patient` objects replayed from an `InferenceCache` with Poisson arrivals."""

    def __init__(
        self,
        cache: InferenceCache,
        schema: LabelSchema,
        arrival_rate: float,
        n_patients: int,
        rng: np.random.Generator | None = None,
        sample_mode: SampleMode = "sequential",
    ) -> None:
        if arrival_rate <= 0:
            raise ValueError(f"arrival_rate must be > 0, got {arrival_rate}")
        if n_patients <= 0:
            raise ValueError(f"n_patients must be > 0, got {n_patients}")
        if cache.n_images == 0:
            raise ValueError("inference cache is empty")
        if tuple(cache.pathologies) != schema.labels:
            raise ValueError(
                "cache pathologies do not match the active LabelSchema; use label_schema: xrv"
            )
        self.cache = cache
        self.schema = schema
        self.arrival_rate = arrival_rate
        self.n_patients = n_patients
        self.rng = rng if rng is not None else np.random.default_rng()
        self.sample_mode: SampleMode = sample_mode

    def _index_stream(self) -> list[int]:
        n = self.cache.n_images
        if self.sample_mode == "random":
            return [int(i) for i in self.rng.integers(0, n, size=self.n_patients)]
        return [i % n for i in range(self.n_patients)]

    def __iter__(self) -> Iterator[tuple[float, Patient]]:
        for idx in self._index_stream():
            gt = tuple(int(x) for x in np.asarray(self.cache.ground_truth[idx]).ravel())
            image: Image = np.zeros((1, 1), dtype=np.uint8)  # unused: AI cached, reader reads GT
            # The cache index rides in patient_id so CachedEnsembleAI can look this image up.
            context = PatientContext(patient_id=int(idx), arrival_time=0.0)
            delay = float(self.rng.exponential(1.0 / self.arrival_rate))
            yield delay, Patient(id=int(idx), images=(image,), ground_truth=gt, context=context)
