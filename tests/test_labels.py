"""Label-view projection: subset + binary-abnormal, and its factory wiring."""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.config.scenario import ScenarioConfig
from hai_sim.core.types import Diagnosis, LabelSchema, ModelOutput, Patient, PatientContext
from hai_sim.factory import build_from_scenario
from hai_sim.labels import ProjectedActor, build_projection, project_generator
from hai_sim.models.mock import RandomMockAI
from hai_sim.sim.engine import SimulationEngine
from hai_sim.sim.patient_generator import PoissonPatientGenerator

_BASE = LabelSchema(labels=("a", "b", "c", "d"))
_CTX = PatientContext(patient_id=0, arrival_time=0.0)


def _out(conf: list[float], labels: tuple[int, ...], embedding: np.ndarray | None = None) -> ModelOutput:
    return ModelOutput(
        diagnosis=Diagnosis(labels=labels, confidence=tuple(conf)),
        uncertainty=tuple([0.1] * len(conf)),
        embedding=embedding,
    )


# --- projection maths --------------------------------------------------------


def test_full_projection_is_identity():
    p = build_projection(_BASE, mode="full")
    assert p.is_identity
    assert p.active == _BASE
    np.testing.assert_allclose(p.project_probs([0.1, 0.2, 0.3, 0.4]), [0.1, 0.2, 0.3, 0.4])


def test_subset_selects_targets():
    p = build_projection(_BASE, mode="subset", targets=("b", "d"))
    assert p.active.labels == ("b", "d")
    np.testing.assert_allclose(p.project_probs([0.1, 0.2, 0.3, 0.4]), [0.2, 0.4])
    np.testing.assert_array_equal(p.project_labels([0, 1, 0, 1]), [1, 1])
    # sentinel -1 preserved through a subset
    assert p.project_gt((0, -1, 0, 1)) == (-1, 1)


def test_subset_requires_targets():
    with pytest.raises(ValueError):
        build_projection(_BASE, mode="subset", targets=None)


def test_unknown_target_raises():
    with pytest.raises(ValueError):
        build_projection(_BASE, mode="subset", targets=("nope",))


def test_binary_abnormal_noisy_or():
    p = build_projection(_BASE, mode="binary_abnormal", combine="noisy_or")
    assert p.active.labels == ("Abnormal",)
    # 1 - (0.9 * 0.8 * 1.0 * 0.5) = 0.64
    np.testing.assert_allclose(p.project_probs([0.1, 0.2, 0.0, 0.5]), [0.64])


def test_binary_abnormal_max():
    p = build_projection(_BASE, mode="binary_abnormal", combine="max")
    np.testing.assert_allclose(p.project_probs([0.1, 0.2, 0.0, 0.5]), [0.5])


def test_binary_abnormal_label_is_or():
    p = build_projection(_BASE, mode="binary_abnormal")
    np.testing.assert_array_equal(p.project_labels([0, 0, 1, 0]), [1])
    np.testing.assert_array_equal(p.project_labels([0, 0, 0, 0]), [0])


def test_binary_abnormal_gt():
    p = build_projection(_BASE, mode="binary_abnormal")
    assert p.project_gt((0, -1, 1, 0)) == (1,)  # any annotated positive
    assert p.project_gt((0, 0, -1, 0)) == (0,)  # annotated, none positive
    assert p.project_gt((-1, -1, -1, -1)) == (-1,)  # no annotated contributor → unknown


def test_binary_abnormal_targets_subset():
    # only labels b,c contribute to "Abnormal"
    p = build_projection(_BASE, mode="binary_abnormal", targets=("b", "c"))
    assert p.indices == (1, 2)
    np.testing.assert_array_equal(p.project_labels([1, 0, 0, 1]), [0])  # a,d ignored


# --- ProjectedActor + generator ---------------------------------------------


def test_project_output_preserves_embedding():
    emb = np.arange(6, dtype=np.float32)
    p = build_projection(_BASE, mode="binary_abnormal")
    out = p.project_output(_out([0.1, 0.2, 0.3, 0.9], (0, 0, 0, 1), embedding=emb))
    assert out.embedding is emb
    assert len(out.diagnosis.labels) == 1
    assert out.diagnosis.labels == (1,)


def test_projected_actor_emits_active_length():
    ai = RandomMockAI(schema=_BASE, rng=np.random.default_rng(0))
    p = build_projection(_BASE, mode="binary_abnormal")
    wrapped = ProjectedActor(ai, p)
    out = wrapped.diagnose([np.zeros((1, 1))])
    assert len(out.diagnosis.labels) == 1
    assert len(out.uncertainty) == 1
    assert "Abnormal" in wrapped.name


def test_project_generator_projects_ground_truth():
    gen = PoissonPatientGenerator(
        schema=_BASE, arrival_rate=1.0, n_patients=6, prevalence=0.5,
        rng=np.random.default_rng(1),
    )
    p = build_projection(_BASE, mode="binary_abnormal")
    for _delay, patient in project_generator(gen, p):
        assert isinstance(patient, Patient)
        assert len(patient.ground_truth) == 1


# --- factory / end-to-end ----------------------------------------------------


def _scenario_dict(label_view: dict | None = None, gt_source: str | None = None) -> dict:
    pg: dict = {"kind": "poisson", "arrival_rate": 0.5, "n_patients": 10, "prevalence": 0.3}
    if gt_source is not None:
        pg["gt_source"] = gt_source
    d: dict = {
        "name": "lv_test",
        "seed": 0,
        "label_schema": "default_v0",
        "patient_generator": pg,
        "ai": {"kind": "random_mock"},
        "persona": {"capability": {"kind": "random_mock"}},
        "sort": {"kind": "random"},
    }
    if label_view is not None:
        d["label_view"] = label_view
    return d


def _run(sc: ScenarioConfig) -> SimulationEngine:
    _schema, gen, ai, doctor, sort, sev, n_ai, n_doc = build_from_scenario(sc)
    return SimulationEngine(
        patient_generator=gen, ai=ai, doctor=doctor, sort_strategy=sort,
        severity_model=sev, n_ai_workers=n_ai, n_doctor_workers=n_doc,
    )


def test_build_from_scenario_full_is_default():
    schema, *_ = build_from_scenario(ScenarioConfig.model_validate(_scenario_dict(None)))
    assert schema.n == 5  # default_v0, unprojected


def test_build_from_scenario_subset_active_schema_and_run():
    sc = ScenarioConfig.model_validate(
        _scenario_dict({"mode": "subset", "targets": ["L1", "L3"]})
    )
    schema, *_ = build_from_scenario(sc)
    assert schema.labels == ("L1", "L3")
    res = _run(sc).run()
    assert res.collector.records
    assert all(len(r.ground_truth) == 2 for r in res.collector.records)
    assert all(len(r.ai_labels) == 2 for r in res.collector.records)
    assert all(len(r.doc_labels) == 2 for r in res.collector.records)


def test_build_from_scenario_binary_abnormal_run():
    sc = ScenarioConfig.model_validate(_scenario_dict({"mode": "binary_abnormal"}))
    schema, *_ = build_from_scenario(sc)
    assert schema.labels == ("Abnormal",)
    res = _run(sc).run()
    assert all(len(r.ground_truth) == 1 for r in res.collector.records)
    assert all(len(r.ai_labels) == 1 for r in res.collector.records)


def test_gt_source_non_native_raises():
    # poisson (synthetic) only provides native GT → capability rejection
    sc = ScenarioConfig.model_validate(_scenario_dict(gt_source="adjudicated"))
    with pytest.raises(ValueError):
        build_from_scenario(sc)
