"""Patient stream over a capability-aware `Dataset`.

`DatasetPatientGenerator` turns any `Dataset` into the simulation's patient
stream: it samples examples (sequential or random), resolves the requested
ground-truth source against the dataset's *declared* capabilities, and stamps
Poisson inter-arrival delays. It is dataset-agnostic — the same generator streams
a `MockDataset`, an `XRVDataset`, or any future `Dataset`.

`XRVDatasetPatientGenerator` is a thin back-compat shim that wraps a raw
TorchXRayVision dataset object in an `XRVDataset` first.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from hai_sim.core.types import DatasetExample, Patient, PatientContext
from hai_sim.data.xrv import XRVDataset
from hai_sim.sim.patient_generator import resolve_arrivals

if TYPE_CHECKING:
    from hai_sim.core.protocols import ArrivalProcess, Dataset
    from hai_sim.core.types import LabelSchema

SampleMode = Literal["sequential", "random"]


class DatasetPatientGenerator:
    """Yields `Patient` objects sampled from a `Dataset` with Poisson arrivals."""

    def __init__(
        self,
        dataset: Dataset,
        arrival_rate: float | None = None,
        n_patients: int = 1,
        rng: np.random.Generator | None = None,
        sample_mode: SampleMode = "sequential",
        gt_source: str = "native",
        arrivals: ArrivalProcess | None = None,
    ) -> None:
        if arrival_rate is not None and arrival_rate <= 0:
            raise ValueError(f"arrival_rate must be > 0, got {arrival_rate}")
        if n_patients <= 0:
            raise ValueError(f"n_patients must be > 0, got {n_patients}")
        if len(dataset) == 0:
            raise ValueError("dataset is empty")
        available = dataset.capabilities.gt_sources
        if gt_source not in available:
            raise ValueError(
                f"dataset {dataset.name!r} does not provide gt_source {gt_source!r}; "
                f"available: {available}"
            )
        self.dataset = dataset
        self.arrival_rate = arrival_rate
        self.n_patients = n_patients
        self.rng = rng if rng is not None else np.random.default_rng()
        self.sample_mode: SampleMode = sample_mode
        self.gt_source = gt_source
        self.arrivals = resolve_arrivals(arrivals, arrival_rate, self.rng)
        # Fix the sampled indices once, so the stream a study replays and the ids a cache
        # layer pre-resolves are the same draw — random sampling must not advance the rng
        # differently between planning and iteration.
        self._indices = self._index_stream()

    def planned_indices(self) -> list[int]:
        """The dataset indices this run will stream, in order."""
        return list(self._indices)

    def planned_example_ids(self) -> list[str]:
        """The example ids this run will stream, resolved without loading pixels where the
        dataset exposes ``example_ids`` — so a cache layer can pre-resolve coverage."""
        ids = getattr(self.dataset, "example_ids", None)
        if ids is not None:
            return [str(ids[i]) for i in self._indices]
        return [self.dataset[i].example_id for i in self._indices]

    def _index_stream(self) -> list[int]:
        n = len(self.dataset)
        if self.sample_mode == "random":
            return [int(i) for i in self.rng.integers(0, n, size=self.n_patients)]
        # sequential, wrapping around if more patients than examples are requested
        return [i % n for i in range(self.n_patients)]

    def _resolve_gt(self, example: DatasetExample) -> tuple[int, ...]:
        if self.gt_source == "native":
            return example.ground_truth
        if self.gt_source.startswith("reader:") and example.reader_labels is not None:
            return example.reader_labels[self.gt_source.split(":", 1)[1]]
        if example.label_sources is not None and self.gt_source in example.label_sources:
            return example.label_sources[self.gt_source]
        raise ValueError(f"cannot resolve gt_source {self.gt_source!r} for this example")

    def __iter__(self) -> Iterator[tuple[float, Patient]]:
        now = 0.0
        for pid, idx in enumerate(self._indices):
            example = self.dataset[idx]
            gt = self._resolve_gt(example)
            delay = float(self.arrivals.next_delay(now))
            now += delay
            meta = example.metadata
            context = PatientContext(
                patient_id=pid,
                arrival_time=0.0,  # set by the engine at arrival
                age=meta.get("age"),
                sex=meta.get("sex"),
                example_id=example.example_id,
            )
            yield delay, Patient(
                id=pid, images=(example.image,), ground_truth=gt, context=context
            )


class XRVDatasetPatientGenerator:
    """Back-compat shim: wrap a raw TorchXRayVision dataset object, then stream it."""

    def __init__(
        self,
        dataset: Any,
        schema: LabelSchema,
        arrival_rate: float,
        n_patients: int,
        rng: np.random.Generator | None = None,
        sample_mode: SampleMode = "sequential",
    ) -> None:
        self._gen = DatasetPatientGenerator(
            XRVDataset(dataset, schema),
            arrival_rate=arrival_rate,
            n_patients=n_patients,
            rng=rng,
            sample_mode=sample_mode,
        )

    def __iter__(self) -> Iterator[tuple[float, Patient]]:
        return iter(self._gen)
