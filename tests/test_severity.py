"""Swappable severity models + their engine/telemetry wiring."""

from __future__ import annotations

import numpy as np

from hai_sim.config.scenario import ScenarioConfig
from hai_sim.core.protocols import SeverityModel
from hai_sim.core.types import Diagnosis, LabelSchema, ModelOutput, Patient, PatientContext
from hai_sim.doctors.mock import RandomMockDoctor
from hai_sim.factory import build_from_scenario, build_severity
from hai_sim.metrics.collector import MetricsCollector
from hai_sim.models.mock import RandomMockAI
from hai_sim.models.torchxrv import pool_embedding_from_feature_maps
from hai_sim.severity.models import (
    ActivationSeverity,
    ConstantSeverity,
    EmbeddingSeverity,
    ImageQualitySeverity,
)
from hai_sim.sim.engine import SimulationEngine
from hai_sim.sim.patient_generator import PoissonPatientGenerator
from hai_sim.sort.random import RandomSort

_CTX = PatientContext(patient_id=0, arrival_time=0.0)


def _patient(gt: tuple[int, ...] = (0, 1, 0), img: np.ndarray | None = None) -> Patient:
    return Patient(
        id=0,
        images=(np.zeros((1, 1)) if img is None else img,),
        ground_truth=gt,
        context=_CTX,
    )


def _ai_output(
    conf: list[float],
    labels: tuple[int, ...] | None = None,
    unc: tuple[float, ...] | None = None,
    embedding: np.ndarray | None = None,
) -> ModelOutput:
    n = len(conf)
    lab = tuple(int(c >= 0.5) for c in conf) if labels is None else labels
    return ModelOutput(
        diagnosis=Diagnosis(labels=lab, confidence=tuple(conf)),
        uncertainty=tuple([0.0] * n) if unc is None else unc,
        embedding=embedding,
    )


# --- embedding pooling (increment 1) ----------------------------------------


def test_pool_embedding_shape_and_relu():
    # (M=2, C=3, h=2, w=2). Negatives must be ReLU'd to 0 before pooling.
    fm = np.zeros((2, 3, 2, 2), dtype=np.float32)
    fm[0, 0] = np.array([[1.0, 3.0], [-5.0, 0.0]])  # relu → mean = (1+3+0+0)/4 = 1.0
    fm[1, 0] = 2.0
    emb = pool_embedding_from_feature_maps(fm)
    assert emb.shape == (3,)
    # channel 0: member0 mean 1.0, member1 mean 2.0 → across-member mean 1.5
    assert abs(float(emb[0]) - 1.5) < 1e-6


def test_pool_embedding_rejects_non_4d():
    try:
        pool_embedding_from_feature_maps(np.zeros((3, 7, 7)))
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-4D feature maps")


def test_model_output_accepts_embedding():
    out = _ai_output([0.2, 0.8], embedding=np.arange(4, dtype=np.float32))
    assert out.embedding is not None
    assert out.embedding.shape == (4,)


# --- ConstantSeverity -------------------------------------------------------


def test_constant_severity_and_protocol():
    s = ConstantSeverity(value=0.7)
    assert s.score(patient=_patient(), ai_output=None, context=_CTX) == 0.7
    assert isinstance(s, SeverityModel)


# --- ActivationSeverity -----------------------------------------------------


def test_activation_probs_max():
    s = ActivationSeverity(source="ai_probs", aggregator="max")
    out = _ai_output([0.1, 0.9, 0.4])
    assert s.score(patient=_patient(), ai_output=out, context=_CTX) == 0.9


def test_activation_criticality_weighting_sum():
    # criticality up-weights label 2; severity = sum(criticality * activation).
    s = ActivationSeverity(source="ai_probs", criticality=(1.0, 1.0, 5.0), aggregator="sum")
    out = _ai_output([0.2, 0.2, 0.2])
    assert abs(float(s.score(patient=_patient(), ai_output=out, context=_CTX)) - 1.4) < 1e-9


def test_activation_vector_mode_returns_per_label():
    s = ActivationSeverity(source="ai_probs", criticality=(2.0, 1.0, 0.0), mode="vector")
    out = _ai_output([0.5, 0.4, 0.9])
    v = s.score(patient=_patient(), ai_output=out, context=_CTX)
    assert isinstance(v, np.ndarray)
    np.testing.assert_allclose(v, [1.0, 0.4, 0.0])


def test_activation_gt_source_masks_sentinel():
    s = ActivationSeverity(source="gt", aggregator="sum")
    p = _patient(gt=(1, -1, 1))  # -1 = not annotated → excluded from severity
    assert s.score(patient=p, ai_output=None, context=_CTX) == 2.0


def test_activation_criticality_length_mismatch_raises():
    s = ActivationSeverity(source="ai_probs", criticality=(1.0, 1.0))
    out = _ai_output([0.5, 0.5, 0.5])
    try:
        s.score(patient=_patient(), ai_output=out, context=_CTX)
    except ValueError:
        return
    raise AssertionError("expected ValueError on criticality/activation length mismatch")


# --- ImageQualitySeverity ---------------------------------------------------


def test_image_quality_intensity_std():
    img = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    s = ImageQualitySeverity(metric="intensity_std")
    assert abs(s.score(patient=_patient(img=img), ai_output=None, context=_CTX) - 0.5) < 1e-9


def test_image_quality_laplacian_var_is_nonneg_float():
    img = np.random.default_rng(0).random((8, 8))
    s = ImageQualitySeverity(metric="laplacian_var")
    val = s.score(patient=_patient(img=img), ai_output=None, context=_CTX)
    assert isinstance(val, float) and val >= 0.0


# --- EmbeddingSeverity ------------------------------------------------------


def test_embedding_severity_l2_norm():
    out = _ai_output([0.5, 0.5], embedding=np.array([3.0, 4.0], dtype=np.float32))
    s = EmbeddingSeverity(reduce="l2_norm")
    assert abs(s.score(patient=_patient(), ai_output=out, context=_CTX) - 5.0) < 1e-6


def test_embedding_severity_none_returns_zero():
    out = _ai_output([0.5, 0.5], embedding=None)
    assert EmbeddingSeverity().score(patient=_patient(), ai_output=out, context=_CTX) == 0.0


# --- factory ----------------------------------------------------------------


def _scenario_dict(severity: dict | None = None) -> dict:
    d: dict = {
        "name": "sev_test",
        "seed": 0,
        "label_schema": "default_v0",
        "patient_generator": {
            "kind": "poisson",
            "arrival_rate": 0.5,
            "n_patients": 5,
            "prevalence": 0.2,
        },
        "ai": {"kind": "random_mock"},
        "doctor": {"kind": "random_mock"},
        "sort": {"kind": "random"},
    }
    if severity is not None:
        d["severity"] = severity
    return d


def test_build_severity_none():
    assert build_severity(None, LabelSchema.default_v0()) is None


def test_build_from_scenario_returns_severity_slot():
    sc = ScenarioConfig.model_validate(_scenario_dict({"kind": "activation", "source": "ai_probs"}))
    _, _, _, _, _, severity, _, _ = build_from_scenario(sc)
    assert isinstance(severity, SeverityModel)


def test_build_from_scenario_no_severity_slot_is_none():
    sc = ScenarioConfig.model_validate(_scenario_dict(None))
    assert build_from_scenario(sc)[5] is None


def test_build_severity_criticality_length_validated():
    sc_dict = _scenario_dict({"kind": "activation", "criticality": [1.0, 1.0]})  # schema n=5
    sc = ScenarioConfig.model_validate(sc_dict)
    try:
        build_from_scenario(sc)
    except ValueError:
        return
    raise AssertionError("expected ValueError on wrong criticality length")


# --- engine + telemetry wiring ----------------------------------------------


def _mock_engine(severity_model: SeverityModel | None, n: int = 8) -> SimulationEngine:
    schema = LabelSchema.default_v0()
    return SimulationEngine(
        patient_generator=PoissonPatientGenerator(
            schema=schema, arrival_rate=0.5, n_patients=n, prevalence=0.3,
            rng=np.random.default_rng(0),
        ),
        ai=RandomMockAI(schema=schema, rng=np.random.default_rng(1)),
        doctor=RandomMockDoctor(schema=schema, rng=np.random.default_rng(2)),
        sort_strategy=RandomSort(rng=np.random.default_rng(3)),
        severity_model=severity_model,
    )


def test_engine_records_scalar_severity(tmp_path):
    res = _mock_engine(ActivationSeverity(source="ai_probs", aggregator="max")).run()
    assert res.collector.records
    assert all(isinstance(r.severity, float) for r in res.collector.records)
    path = tmp_path / "rec.csv"
    res.collector.write_csv(path, label_names=LabelSchema.default_v0().labels)
    assert "severity" in path.read_text(encoding="utf-8").splitlines()[0].split(",")


def test_engine_without_severity_leaves_none_and_no_column(tmp_path):
    res = _mock_engine(None).run()
    assert all(r.severity is None for r in res.collector.records)
    path = tmp_path / "rec.csv"
    res.collector.write_csv(path, label_names=LabelSchema.default_v0().labels)
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert not any(h.startswith("severity") for h in header)


def test_collector_writes_vector_severity_columns(tmp_path):
    schema = LabelSchema.default_v0()
    coll = MetricsCollector()
    p = _patient(gt=(0, 1, 0, 1, 0))
    out = _ai_output([0.1, 0.2, 0.3, 0.4, 0.5])
    coll.add(
        patient=p, ts_arrival=0.0, ts_model_start=0.0, ts_model_end=0.0,
        ts_doctor_start=0.0, ts_doctor_end=0.0, ai_out=out, doc_out=out,
        priority_key=0.0, severity=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    )
    path = tmp_path / "v.csv"
    coll.write_csv(path, label_names=schema.labels)
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert f"severity__{schema.labels[0]}" in header
    assert f"severity__{schema.labels[-1]}" in header
