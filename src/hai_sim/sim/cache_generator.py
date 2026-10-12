"""Patient stream that replays an `InferenceCache`.

Yields patients whose ground truth is the cached label vector and whose **image id travels in
``context.example_id``** so a paired `CachedEnsembleAI` can look up that image's cached
predictions. ``patient_id`` stays a plain arrival counter, so sampling the same image twice
still produces two distinguishable patients. The image pixels are a placeholder: in a
cache-replay study the AI is served from the cache and the simulated reader reads ground
truth, so nobody needs them.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

import numpy as np

from hai_sim.core.types import Image, Patient, PatientContext
from hai_sim.sim.patient_generator import resolve_arrivals

if TYPE_CHECKING:
    from hai_sim.analysis.inference_cache import InferenceCache
    from hai_sim.core.protocols import ArrivalProcess
    from hai_sim.core.types import LabelSchema

SampleMode = Literal["sequential", "random", "permutation"]


class CachedPatientGenerator:
    """Yields `Patient` objects replayed from an `InferenceCache` with Poisson arrivals."""

    def __init__(
        self,
        cache: InferenceCache,
        schema: LabelSchema,
        arrival_rate: float | None = None,
        n_patients: int = 1,
        rng: np.random.Generator | None = None,
        sample_mode: SampleMode = "sequential",
        arrivals: ArrivalProcess | None = None,
        split: str = "all",
    ) -> None:
        if split not in ("all", "even", "odd"):
            raise ValueError(f"split must be 'all', 'even' or 'odd', got {split!r}")
        if arrival_rate is not None and arrival_rate <= 0:
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
        # The rows this stream may draw from. Parity of the cache row is the same split
        # scripts/fit_calibration.py uses, so "fit on even, study on odd" is one word.
        rows = np.arange(cache.n_images)
        if split != "all":
            rows = rows[rows % 2 == (0 if split == "even" else 1)]
        if rows.size == 0:
            raise ValueError(f"split={split!r} leaves no cache rows to draw from")
        self.split = split
        self._rows = rows
        self.arrivals = resolve_arrivals(arrivals, arrival_rate, self.rng)

    def _index_stream(self) -> list[int]:
        n = int(self._rows.size)
        if self.sample_mode == "random":
            picks = self.rng.integers(0, n, size=self.n_patients)
        elif self.sample_mode == "permutation":
            # Every row once, in a random order; with n_patients equal to the pool this is
            # "the whole dataset arrives, shuffled". Longer streams start a new shuffle.
            rounds = -(-self.n_patients // n)
            picks = np.concatenate([self.rng.permutation(n) for _ in range(rounds)])[: self.n_patients]
        else:
            picks = np.arange(self.n_patients) % n
        return [int(self._rows[i]) for i in picks]

    def __iter__(self) -> Iterator[tuple[float, Patient]]:
        now = 0.0
        for pid, idx in enumerate(self._index_stream()):
            gt = tuple(int(x) for x in np.asarray(self.cache.ground_truth[idx]).ravel())
            image: Image = np.zeros((1, 1), dtype=np.uint8)  # unused: AI cached, reader reads GT
            # The image id travels in example_id so CachedEnsembleAI can look this image up;
            # patient_id counts arrivals, so the same image sampled twice = two patients.
            context = PatientContext(
                patient_id=pid,
                arrival_time=0.0,
                example_id=str(self.cache.image_ids[idx]),
            )
            delay = float(self.arrivals.next_delay(now))
            now += delay
            yield delay, Patient(id=pid, images=(image,), ground_truth=gt, context=context)
