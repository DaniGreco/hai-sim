"""Patient arrival generator.

For v0: Poisson inter-arrival times, random multi-label ground truth with a
configurable prevalence per label, placeholder image array.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from hai_sim.core.types import Image, LabelSchema, Patient, PatientContext


class PoissonPatientGenerator:
    """Yields `Patient` objects with Poisson inter-arrival times.

    `prevalence[i]` is the prior probability that label i is positive in the
    ground truth. Pass a scalar to apply uniformly to all labels.
    """

    def __init__(
        self,
        schema: LabelSchema,
        arrival_rate: float,
        n_patients: int,
        prevalence: float | tuple[float, ...] = 0.2,
        image_shape: tuple[int, ...] = (1, 1),
        rng: np.random.Generator | None = None,
    ) -> None:
        if arrival_rate <= 0:
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

    def __iter__(self) -> Iterator[tuple[float, Patient]]:
        """Yield (inter_arrival_delay, patient) pairs."""
        for pid in range(self.n_patients):
            delay = float(self.rng.exponential(1.0 / self.arrival_rate))
            gt = tuple(
                int(self.rng.random() < p) for p in self.prevalence
            )
            image: Image = np.zeros(self.image_shape, dtype=np.uint8)
            context = PatientContext(patient_id=pid, arrival_time=0.0)  # arrival_time set by engine
            yield delay, Patient(
                id=pid,
                images=(image,),
                ground_truth=gt,
                context=context,
            )
