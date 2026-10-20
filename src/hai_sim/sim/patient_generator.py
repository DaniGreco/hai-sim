"""Synthetic patient arrivals — the dependency-free stream used by tests and smoke runs.

*When* patients arrive is delegated to an `ArrivalProcess`, so the same stream can be
driven by a flat rate, a 24x7 time-varying one, or a metronome. Passing ``arrival_rate``
instead builds a homogeneous Poisson process, which stays the default baseline.

Ground truth is drawn per label from a configured prevalence, and the image is a
placeholder: nothing in this stream looks at pixels. For real images and real ground
truth use `DatasetPatientGenerator` (any `Dataset`) or `CachedPatientGenerator` (replay
of pre-computed model outputs).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from hai_sim.arrivals.processes import PoissonArrival
from hai_sim.core.types import Image, LabelSchema, Patient, PatientContext

if TYPE_CHECKING:
    from hai_sim.core.protocols import ArrivalProcess


def resolve_arrivals(
    arrivals: ArrivalProcess | None,
    arrival_rate: float | None,
    rng: np.random.Generator,
) -> ArrivalProcess:
    """Take an explicit process, or build the default homogeneous Poisson one.

    Shared by all three patient generators so ``arrival_rate: 0.5`` keeps meaning
    exactly what it always did while the richer processes become available.
    """
    if arrivals is not None:
        return arrivals
    if arrival_rate is None:
        raise ValueError("pass either an arrival process or an arrival_rate")
    return PoissonArrival(rate=arrival_rate, rng=rng)


class PoissonPatientGenerator:
    """Yields `Patient` objects with synthetic ground truth and configurable arrivals.

    `prevalence[i]` is the prior probability that label i is positive in the
    ground truth. Pass a scalar to apply uniformly to all labels.
    """

    def __init__(
        self,
        schema: LabelSchema,
        arrival_rate: float | None = None,
        n_patients: int = 1,
        prevalence: float | tuple[float, ...] = 0.2,
        image_shape: tuple[int, ...] = (1, 1),
        rng: np.random.Generator | None = None,
        arrivals: ArrivalProcess | None = None,
        embed_truth: bool = False,
    ) -> None:
        if arrival_rate is not None and arrival_rate <= 0:
            raise ValueError(f"arrival_rate must be > 0, got {arrival_rate}")
        if n_patients <= 0:
            raise ValueError(f"n_patients must be > 0, got {n_patients}")
        self.schema = schema
        self.arrival_rate = arrival_rate
        self.n_patients = n_patients
        if isinstance(prevalence, (int, float)):
            self.prevalence: tuple[float, ...] = tuple([float(prevalence)] * schema.n)
        else:
            if len(prevalence) != schema.n:
                raise ValueError(
                    f"prevalence length {len(prevalence)} != schema.n {schema.n}"
                )
            self.prevalence = tuple(float(p) for p in prevalence)
        self.image_shape = image_shape
        self.rng = rng if rng is not None else np.random.default_rng()
        self.arrivals = resolve_arrivals(arrivals, arrival_rate, self.rng)
        self.embed_truth = embed_truth

    def __iter__(self) -> Iterator[tuple[float, Patient]]:
        """Yield (inter_arrival_delay, patient) pairs."""
        now = 0.0
        for pid in range(self.n_patients):
            delay = float(self.arrivals.next_delay(now))
            now += delay
            gt = tuple(
                int(self.rng.random() < p) for p in self.prevalence
            )
            # Normally a placeholder — nothing in this stream reads pixels. With
            # ``embed_truth`` the "image" carries the case's true label vector instead, the
            # clean channel by which a *synthetic* reader (e.g. `FixedAccuracyAI`) learns
            # the case without the engine ever handing the AI stage the ground truth a real
            # model must not see.
            image: Image = (
                np.asarray(gt, dtype=np.float64)
                if self.embed_truth
                else np.zeros(self.image_shape, dtype=np.uint8)
            )
            # No example_id: these patients are synthesised, not drawn from a dataset, so
            # there is no image identity to record. arrival_time is stamped by the engine.
            context = PatientContext(patient_id=pid, arrival_time=0.0)
            yield delay, Patient(
                id=pid,
                images=(image,),
                ground_truth=gt,
                context=context,
            )
