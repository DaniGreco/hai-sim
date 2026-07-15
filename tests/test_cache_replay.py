"""Cache-replay AI + generator: real predictions from an InferenceCache, no re-inference."""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.analysis.inference_cache import InferenceCache
from hai_sim.config.scenario import ScenarioConfig
from hai_sim.core.types import LabelSchema, PatientContext
from hai_sim.factory import build_from_scenario
from hai_sim.models.cached import CachedEnsembleAI
from hai_sim.sim.cache_generator import CachedPatientGenerator
from hai_sim.sim.engine import SimulationEngine
from hai_sim.uq.methods import EntropyUQ

_SCHEMA = LabelSchema.default_v0()


def _cache() -> InferenceCache:
    return InferenceCache(
        member_probs=np.array(
            [
                [[0.9, 0.1, 0.5, 0.2, 0.8], [0.8, 0.2, 0.5, 0.3, 0.7]],
                [[0.1, 0.9, 0.4, 0.6, 0.2], [0.2, 0.8, 0.5, 0.5, 0.3]],
                [[0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5]],
            ],
            dtype=np.float32,
        ),
        ground_truth=np.array([[1, 0, -1, 0, 1], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=np.int8),
        inference_seconds=np.array([5.0, 5.1, 4.9]),
        image_ids=("a", "b", "c"),
        dataset_indices=np.array([0, 1, 2]),
        weights=("w1", "w2"),
        pathologies=("L0", "L1", "L2", "L3", "L4"),
        dataset="mock",
        batch_size=1,
    )


# --- CachedEnsembleAI ---------------------------------------------------------


def test_cached_ai_reconstructs_output_from_index():
    ai = CachedEnsembleAI(_cache(), _SCHEMA, uq=EntropyUQ(), decision_threshold=0.5)
    out = ai.diagnose([np.zeros((1, 1))], context=PatientContext(patient_id=0, arrival_time=0.0))
    # probs = mean over the 2 members of member_probs[0] = [0.85, 0.15, 0.5, 0.25, 0.75]
    assert out.diagnosis.labels == (1, 0, 1, 0, 1)
    assert abs(out.diagnosis.confidence[0] - 0.85) < 1e-6
    assert out.extra["cache_index"] == 0


def test_cached_ai_needs_context():
    ai = CachedEnsembleAI(_cache(), _SCHEMA)
    with pytest.raises(ValueError, match="cache index"):
        ai.diagnose([np.zeros((1, 1))], context=None)


def test_cached_ai_rejects_schema_mismatch():
    with pytest.raises(ValueError, match="pathologies"):
        CachedEnsembleAI(_cache(), LabelSchema(labels=("x", "y")))


# --- CachedPatientGenerator ---------------------------------------------------


def test_cached_generator_replays_gt_and_index():
    gen = CachedPatientGenerator(
        _cache(), _SCHEMA, arrival_rate=1.0, n_patients=3, rng=np.random.default_rng(0)
    )
    items = list(gen)
    assert len(items) == 3
    _delay, patient = items[0]
    assert patient.ground_truth == (1, 0, -1, 0, 1)  # cache row 0
    assert patient.context.patient_id == 0  # cache index rides in patient_id


# --- end-to-end via the factory (save → load → build → run) -------------------


def test_cache_replay_scenario_runs(tmp_path):
    stem = tmp_path / "c"
    _cache().save(stem)
    sc = ScenarioConfig.model_validate(
        {
            "name": "cr",
            "seed": 0,
            "label_schema": "default_v0",
            "patient_generator": {
                "kind": "inference_cache",
                "cache_stem": str(stem),
                "arrival_rate": 1.0,
                "n_patients": 3,
            },
            "ai": {"kind": "cached", "cache_stem": str(stem), "uq": "entropy"},
            "persona": {"capability": {"kind": "fixed_accuracy", "sensitivity": 0.9, "specificity": 0.9}},
            "sort": {"kind": "random"},
        }
    )
    _schema, gen, ai, doctor, sort, sev, n_ai, n_doc = build_from_scenario(sc)
    res = SimulationEngine(
        patient_generator=gen, ai=ai, doctor=doctor, sort_strategy=sort,
        severity_model=sev, n_ai_workers=n_ai, n_doctor_workers=n_doc,
    ).run()
    assert len(res.collector.records) == 3
    # AI label vectors came from the cache (patient 0 → labels reflect its cached probs)
    rec0 = next(r for r in res.collector.records if r.patient_id == 0)
    assert rec0.ai_labels == (1, 0, 1, 0, 1)
