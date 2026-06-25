"""Patient generator backed by a real CXR dataset (TorchXRayVision loaders).

Wraps any TorchXRayVision dataset object (NIH, CheXpert, RSNA, ...) and turns it
into the simulation's patient stream: Poisson inter-arrival timing, real
pre-processed images, and ground-truth label vectors aligned to the dataset's
``pathologies``. Labels the dataset does not annotate arrive as ``-1`` (the
"not annotated" sentinel) and are skipped by the metrics.

The dataset object is expected to:
  - expose ``pathologies`` (a sequence aligned to the active ``LabelSchema``),
  - support ``len(dataset)`` and ``dataset[i] -> {"img": ndarray, "lab": ndarray}``
    where ``img`` is the model-ready ``(1, 224, 224)`` array and ``lab`` is a
    float vector with ``1.0`` / ``0.0`` / ``NaN`` entries.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

import numpy as np

from hai_sim.core.types import Image, LabelSchema, Patient, PatientContext

SampleMode = Literal["sequential", "random"]


def _to_sentinel(v: float) -> int:
    """Map a dataset label cell to the simulation's {1, 0, -1} convention."""
    if v != v:  # NaN
        return -1
    return 1 if float(v) >= 0.5 else 0


class XRVDatasetPatientGenerator:
    """Yields `Patient` objects sampled from a TorchXRayVision dataset."""

    def __init__(
        self,
        dataset: Any,
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
        if len(dataset) == 0:
            raise ValueError("dataset is empty")
        if tuple(dataset.pathologies) != schema.labels:
            raise ValueError(
                "dataset.pathologies do not match the active LabelSchema; "
                "align the schema to the dataset (use label_schema: xrv)."
            )
        self.dataset = dataset
        self.schema = schema
        self.arrival_rate = arrival_rate
        self.n_patients = n_patients
        self.rng = rng if rng is not None else np.random.default_rng()
        self.sample_mode: SampleMode = sample_mode

    def _index_stream(self) -> list[int]:
        n = len(self.dataset)
        if self.sample_mode == "random":
            return [int(i) for i in self.rng.integers(0, n, size=self.n_patients)]
        # sequential, wrapping around if more patients than images are requested
        return [i % n for i in range(self.n_patients)]

    def __iter__(self) -> Iterator[tuple[float, Patient]]:
        for pid, idx in enumerate(self._index_stream()):
            item = self.dataset[idx]
            image: Image = np.asarray(item["img"], dtype=np.float32)
            gt = tuple(_to_sentinel(v) for v in np.asarray(item["lab"]).ravel())
            if len(gt) != self.schema.n:
                raise ValueError(
                    f"dataset label length {len(gt)} != schema.n {self.schema.n}"
                )
            delay = float(self.rng.exponential(1.0 / self.arrival_rate))
            context = PatientContext(patient_id=pid, arrival_time=0.0)
            yield delay, Patient(
                id=pid,
                images=(image,),
                ground_truth=gt,
                context=context,
            )
