"""Random sort: baseline of comparison for any uncertainty-based strategy."""

from __future__ import annotations

import numpy as np

from hai_sim.core.types import ModelOutput, Patient


class RandomSort:
    """Assigns a uniform-random priority key. Patient/output ignored."""

    name: str = "RandomSort"

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def priority(self, output: ModelOutput, patient: Patient) -> float:
        return float(self.rng.random())
