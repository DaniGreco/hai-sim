"""Capability-aware Dataset contract: MockDataset / XRVDataset + agnostic streaming."""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.config.scenario import ScenarioConfig
from hai_sim.core.protocols import Dataset
from hai_sim.core.types import DatasetExample, LabelSchema
from hai_sim.data.mock import MockDataset
from hai_sim.data.xrv import XRVDataset
from hai_sim.factory import build_from_scenario
from hai_sim.sim.dataset_generator import DatasetPatientGenerator, XRVDatasetPatientGenerator
from hai_sim.sim.engine import SimulationEngine

_SCHEMA = LabelSchema.default_v0()


class _StubXRV:
    """Duck-typed xrv-like dataset: fixed labels with one NaN to exercise masking."""

    def __init__(self, schema: LabelSchema, n: int = 4) -> None:
        self.pathologies = list(schema.labels)
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        lab = np.array([1.0, 0.0, np.nan, 1.0, 0.0])  # len 5 == default_v0
        return {"img": np.zeros((1, 224, 224), dtype=np.float32), "lab": lab}


# --- MockDataset -------------------------------------------------------------


def test_mock_dataset_is_dataset_protocol():
    ds = MockDataset(schema=_SCHEMA, n_examples=5, rng=np.random.default_rng(0))
    assert isinstance(ds, Dataset)
    assert len(ds) == 5
    ex = ds[0]
    assert isinstance(ex, DatasetExample)
    assert len(ex.ground_truth) == _SCHEMA.n
    assert ds.capabilities.gt_sources == ("native",)
    assert ds.capabilities.has_masks is False


def test_mock_dataset_is_stable_across_access_order():
    ds = MockDataset(schema=_SCHEMA, n_examples=6, prevalence=0.5, rng=np.random.default_rng(1))
    first = ds[3].ground_truth
    _ = [ds[i] for i in range(6)]
    assert ds[3].ground_truth == first  # index → same example regardless of order


# --- XRVDataset --------------------------------------------------------------


def test_xrv_dataset_wraps_and_maps_nan():
    ds = XRVDataset(_StubXRV(_SCHEMA), _SCHEMA, name="stub")
    assert isinstance(ds, Dataset)
    ex = ds[0]
    assert ex.ground_truth == (1, 0, -1, 1, 0)
    assert ex.image.shape == (1, 224, 224)


def test_xrv_dataset_rejects_schema_mismatch():
    with pytest.raises(ValueError, match="pathologies"):
        XRVDataset(_StubXRV(LabelSchema(labels=("x", "y"))), _SCHEMA)


# --- DatasetPatientGenerator (agnostic) --------------------------------------


def test_generator_streams_patients_from_mock_dataset():
    ds = MockDataset(schema=_SCHEMA, n_examples=8, rng=np.random.default_rng(2))
    gen = DatasetPatientGenerator(ds, arrival_rate=1.0, n_patients=5, rng=np.random.default_rng(3))
    items = list(gen)
    assert len(items) == 5
    delay, patient = items[0]
    assert delay > 0.0
    assert len(patient.ground_truth) == _SCHEMA.n


def test_generator_sequential_wraps_around():
    ds = MockDataset(schema=_SCHEMA, n_examples=2, rng=np.random.default_rng(4))
    gen = DatasetPatientGenerator(ds, arrival_rate=1.0, n_patients=5, rng=np.random.default_rng(5))
    assert len(list(gen)) == 5


def test_generator_gates_unavailable_gt_source():
    ds = MockDataset(schema=_SCHEMA, n_examples=4, rng=np.random.default_rng(6))
    with pytest.raises(ValueError, match="does not provide gt_source"):
        DatasetPatientGenerator(ds, arrival_rate=1.0, n_patients=2, gt_source="adjudicated")


def test_xrv_shim_still_works():
    gen = XRVDatasetPatientGenerator(
        _StubXRV(_SCHEMA), _SCHEMA, arrival_rate=1.0, n_patients=3,
        rng=np.random.default_rng(0),
    )
    items = list(gen)
    assert len(items) == 3
    assert items[0][1].ground_truth == (1, 0, -1, 1, 0)


# --- factory / end-to-end ----------------------------------------------------


def test_build_from_scenario_mock_dataset_runs():
    sc = ScenarioConfig.model_validate(
        {
            "name": "ds_test",
            "seed": 0,
            "label_schema": "default_v0",
            "patient_generator": {
                "kind": "mock_dataset",
                "arrival_rate": 0.5,
                "n_patients": 12,
                "n_examples": 20,
                "prevalence": 0.3,
            },
            "ai": {"kind": "random_mock"},
            "persona": {"capability": {"kind": "random_mock"}},
            "sort": {"kind": "random"},
        }
    )
    schema, gen, ai, doctor, sort, sev, n_ai, n_doc = build_from_scenario(sc)
    res = SimulationEngine(
        patient_generator=gen, ai=ai, doctor=doctor, sort_strategy=sort,
        severity_model=sev, n_ai_workers=n_ai, n_doctor_workers=n_doc,
    ).run()
    assert len(res.collector.records) == 12
    assert all(len(r.ground_truth) == schema.n for r in res.collector.records)


def test_permutation_is_refused_for_an_image_dataset():
    """`permutation` exists for cached streams only; elsewhere it must fail, not run sequentially."""
    sc = ScenarioConfig.model_validate(
        {
            "name": "perm_test",
            "seed": 0,
            "label_schema": "default_v0",
            "patient_generator": {
                "kind": "mock_dataset",
                "arrival_rate": 0.5,
                "n_patients": 12,
                "n_examples": 20,
                "prevalence": 0.3,
                "sample_mode": "permutation",
            },
            "ai": {"kind": "random_mock"},
            "persona": {"capability": {"kind": "random_mock"}},
            "sort": {"kind": "random"},
        }
    )
    with pytest.raises(ValueError, match="permutation"):
        build_from_scenario(sc)
