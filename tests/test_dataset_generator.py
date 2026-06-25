"""XRVDatasetPatientGenerator over a stub dataset (no real data / torch needed)."""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.core.types import LabelSchema
from hai_sim.sim.dataset_generator import XRVDatasetPatientGenerator


class _StubDataset:
    """Duck-typed xrv-like dataset: fixed labels with one NaN to exercise masking."""

    def __init__(self, schema: LabelSchema, n: int = 4) -> None:
        self.pathologies = list(schema.labels)
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        lab = np.array([1.0, 0.0, np.nan, 1.0, 0.0])  # len 5 == default_v0
        return {"img": np.zeros((1, 224, 224), dtype=np.float32), "lab": lab}


def test_generator_yields_and_maps_nan_to_sentinel() -> None:
    schema = LabelSchema.default_v0()
    gen = XRVDatasetPatientGenerator(
        _StubDataset(schema), schema, arrival_rate=1.0, n_patients=3,
        rng=np.random.default_rng(0),
    )
    items = list(gen)
    assert len(items) == 3
    delay, patient = items[0]
    assert delay > 0.0
    assert patient.ground_truth == (1, 0, -1, 1, 0)
    assert patient.images[0].shape == (1, 224, 224)


def test_generator_rejects_mismatched_schema() -> None:
    schema = LabelSchema.default_v0()
    bad = _StubDataset(LabelSchema(labels=("x", "y")))
    with pytest.raises(ValueError, match="pathologies"):
        XRVDatasetPatientGenerator(bad, schema, arrival_rate=1.0, n_patients=1)


def test_sequential_wraps_around() -> None:
    schema = LabelSchema.default_v0()
    gen = XRVDatasetPatientGenerator(
        _StubDataset(schema, n=2), schema, arrival_rate=1.0, n_patients=5,
        sample_mode="sequential", rng=np.random.default_rng(0),
    )
    assert len(list(gen)) == 5
