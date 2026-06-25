"""End-to-end test driving the REAL TorchXRayVision model through the full
engine + KPI pipeline. Skipped automatically when the [models] extra is absent.

Uses a stub 18-label dataset so no Kaggle download is required: it validates the
wiring (real inference + trivial doctor + dataset generator + metrics), not the
clinical numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torchxrayvision")

from hai_sim.core.types import LabelSchema
from hai_sim.doctors.simple import FixedAccuracyDoctor
from hai_sim.metrics.kpi import compute_kpis
from hai_sim.models.torchxrv import TorchXRayVisionAI
from hai_sim.sim.dataset_generator import XRVDatasetPatientGenerator
from hai_sim.sim.engine import SimulationEngine
from hai_sim.sort.uncertainty import UncertaintyDescSort
from hai_sim.uq.methods import EntropyUQ


class _Stub18:
    """xrv-like dataset returning model-ready images + 18-label vectors."""

    def __init__(self, schema: LabelSchema, n: int = 5) -> None:
        self.pathologies = list(schema.labels)
        self._n = n
        self._rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        lab = self._rng.choice([0.0, 1.0, np.nan], size=18)
        img = (self._rng.random((1, 224, 224)) * 2048.0 - 1024.0).astype(np.float32)
        return {"img": img, "lab": lab}


def test_real_model_runs_end_to_end() -> None:
    schema = LabelSchema.xrv()
    ai = TorchXRayVisionAI(schema=schema, weights="densenet121-res224-all", uq=EntropyUQ())
    doctor = FixedAccuracyDoctor(
        schema, rng=np.random.default_rng(0), sensitivity=0.9, specificity=0.9
    )
    gen = XRVDatasetPatientGenerator(
        _Stub18(schema), schema, arrival_rate=1.0, n_patients=5,
        rng=np.random.default_rng(0),
    )
    engine = SimulationEngine(gen, ai, doctor, UncertaintyDescSort())
    result = engine.run()
    kpis = compute_kpis(result.collector.records)

    assert kpis.n_patients == 5
    assert 0.0 <= kpis.ai.f1_macro <= 1.0
    # Every record carries an 18-length probability + uncertainty vector.
    rec = result.collector.records[0]
    assert len(rec.ai_confidence) == 18
    assert len(rec.ai_uncertainty) == 18


def test_schema_must_be_18() -> None:
    with pytest.raises(ValueError, match="18"):
        TorchXRayVisionAI(schema=LabelSchema.default_v0(), weights="densenet121-res224-all")
