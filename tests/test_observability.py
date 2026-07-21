"""The observability layer: nothing a module produces is lost on its way to a metric.

The framework used to compute observables it could not measure — the whole
aleatoric/epistemic decomposition of the ensemble uncertainty was recomputed for every
patient and dropped by the collector. These tests pin the contract that fixes it: an
actor's `ModelOutput.extra`, the patient context and the queue occupancy all reach the
telemetry, namespaced, and survive the trip to CSV.
"""

from __future__ import annotations

import csv

import numpy as np

from hai_sim.analysis.inference_cache import InferenceCache
from hai_sim.core.types import Diagnosis, LabelSchema, ModelOutput, PatientContext
from hai_sim.data.mock import MockDataset
from hai_sim.doctors.assisted import AssistedDoctor, LinearPoolPolicy
from hai_sim.doctors.mock import RandomMockDoctor
from hai_sim.metrics.collector import MetricsCollector, PatientRecord
from hai_sim.models.cached import CachedEnsembleAI
from hai_sim.models.mock import RandomMockAI
from hai_sim.sim.cache_generator import CachedPatientGenerator
from hai_sim.sim.dataset_generator import DatasetPatientGenerator
from hai_sim.sim.engine import SimulationEngine
from hai_sim.sim.patient_generator import PoissonPatientGenerator
from hai_sim.sort.random import RandomSort
from hai_sim.uq.methods import EnsembleUQ

_SCHEMA = LabelSchema.default_v0()
_LABELS = _SCHEMA.labels


def _fast_doctor(seed: int = 2, **kwargs) -> RandomMockDoctor:
    """A doctor with an ~1.0 service time, so queue behaviour is easy to reason about."""
    return RandomMockDoctor(
        schema=_SCHEMA,
        rng=np.random.default_rng(seed),
        service_time_mu=0.0,
        service_time_sigma=0.01,
        **kwargs,
    )


def _cache() -> InferenceCache:
    return InferenceCache(
        member_probs=np.array(
            [
                [[0.9, 0.1, 0.5, 0.2, 0.8], [0.4, 0.2, 0.5, 0.3, 0.7]],
                [[0.1, 0.9, 0.4, 0.6, 0.2], [0.2, 0.8, 0.5, 0.5, 0.3]],
            ],
            dtype=np.float32,
        ),
        ground_truth=np.array([[1, 0, -1, 0, 1], [0, 1, 0, 1, 0]], dtype=np.int8),
        inference_seconds=np.array([5.0, 5.1]),
        image_ids=("img_a.png", "img_b.png"),
        dataset_indices=np.array([0, 1]),
        weights=("w1", "w2"),
        pathologies=_LABELS,
        dataset="mock",
        batch_size=1,
    )


# ── The bag is populated, namespaced, and non-duplicating ──────────────────────


def test_ai_and_doctor_extras_reach_the_record():
    gen = PoissonPatientGenerator(
        schema=_SCHEMA, arrival_rate=1.0, n_patients=5, prevalence=0.3,
        rng=np.random.default_rng(0),
    )
    res = SimulationEngine(
        patient_generator=gen,
        ai=RandomMockAI(schema=_SCHEMA, rng=np.random.default_rng(1)),
        doctor=_fast_doctor(),
        sort_strategy=RandomSort(rng=np.random.default_rng(3)),
    ).run()
    rec = res.collector.records[0]
    assert rec.observations["ai._mock"] is True  # RandomMockAI's own extra key
    assert rec.observation("ai._mock") is True  # and the accessor agrees
    assert rec.observation("nope", "fallback") == "fallback"


def test_uncertainty_decomposition_survives_to_the_telemetry():
    """The headline case: all four ensemble uncertainty measures are computed per
    patient. They used to be discarded, making 'does epistemic uncertainty triage
    better than total entropy?' unaskable even though the numbers existed."""
    cache = _cache()
    gen = CachedPatientGenerator(
        cache, _SCHEMA, arrival_rate=1.0, n_patients=4, rng=np.random.default_rng(0)
    )
    ai = CachedEnsembleAI(cache, _SCHEMA, uq=EnsembleUQ(measure="mutual_information"))
    res = SimulationEngine(
        patient_generator=gen, ai=ai, doctor=_fast_doctor(),
        sort_strategy=RandomSort(rng=np.random.default_rng(3)),
    ).run()

    rec = res.collector.records[0]
    for measure in ("predictive_entropy", "mean_entropy", "mutual_information", "std"):
        assert len(rec.observations[f"ai.{measure}"]) == _SCHEMA.n
    # epistemic = total - aleatoric, and the identity must hold entry by entry
    total = rec.observations["ai.predictive_entropy"]
    aleatoric = rec.observations["ai.mean_entropy"]
    epistemic = rec.observations["ai.mutual_information"]
    for t, a, e in zip(total, aleatoric, epistemic, strict=True):
        assert abs(e - max(t - a, 0.0)) < 1e-9
    # and the image identity came along, so telemetry can be joined back to the image
    assert rec.observations["ctx.example_id"] in ("img_a.png", "img_b.png")
    assert rec.observations["ai.image_id"] == rec.observations["ctx.example_id"]


def test_context_fields_land_under_ctx():
    dataset = MockDataset(_SCHEMA, n_examples=4, rng=np.random.default_rng(0))
    gen = DatasetPatientGenerator(dataset, arrival_rate=1.0, n_patients=4)
    res = SimulationEngine(
        patient_generator=gen,
        ai=RandomMockAI(schema=_SCHEMA, rng=np.random.default_rng(1)),
        doctor=_fast_doctor(),
        sort_strategy=RandomSort(rng=np.random.default_rng(3)),
    ).run()
    ids = {r.observations["ctx.example_id"] for r in res.collector.records}
    assert ids == {"0", "1", "2", "3"}  # example_id used to be dropped entirely


def test_promoted_key_is_not_duplicated_into_the_bag():
    """``doctor_solo_labels`` is a first-class field; it must not also show up as an
    observation, or every assisted run would carry the same vector twice."""
    doctor = AssistedDoctor(base=_fast_doctor(), policy=LinearPoolPolicy(weight_ai=0.5))
    gen = PoissonPatientGenerator(
        schema=_SCHEMA, arrival_rate=1.0, n_patients=4, prevalence=0.3,
        rng=np.random.default_rng(0),
    )
    res = SimulationEngine(
        patient_generator=gen,
        ai=RandomMockAI(schema=_SCHEMA, rng=np.random.default_rng(1)),
        doctor=doctor,
        sort_strategy=RandomSort(rng=np.random.default_rng(3)),
    ).run()
    rec = res.collector.records[0]
    assert rec.doctor_solo_labels is not None
    assert "doc.doctor_solo_labels" not in rec.observations


# ── Workload: the variable that makes order effects studiable ──────────────────


def _mean_queue_at_arrival(arrival_rate: float, n_patients: int = 60) -> float:
    gen = PoissonPatientGenerator(
        schema=_SCHEMA, arrival_rate=arrival_rate, n_patients=n_patients, prevalence=0.3,
        rng=np.random.default_rng(0),
    )
    res = SimulationEngine(
        patient_generator=gen,
        ai=RandomMockAI(schema=_SCHEMA, rng=np.random.default_rng(1)),
        doctor=_fast_doctor(),
        sort_strategy=RandomSort(rng=np.random.default_rng(3)),
    ).run()
    waits = [r.observations["load.in_system_at_arrival"] for r in res.collector.records]
    return float(np.mean(waits))


def test_load_observations_are_recorded_at_every_boundary():
    gen = PoissonPatientGenerator(
        schema=_SCHEMA, arrival_rate=1.0, n_patients=5, prevalence=0.3,
        rng=np.random.default_rng(0),
    )
    res = SimulationEngine(
        patient_generator=gen,
        ai=RandomMockAI(schema=_SCHEMA, rng=np.random.default_rng(1)),
        doctor=_fast_doctor(),
        sort_strategy=RandomSort(rng=np.random.default_rng(3)),
    ).run()
    for r in res.collector.records:
        for key in (
            "load.input_queue_at_arrival",
            "load.in_system_at_arrival",
            "load.triaged_at_enqueue",
            "load.triaged_at_read",
        ):
            assert key in r.observations
    # the first arrival finds an empty department, by construction
    first = min(res.collector.records, key=lambda r: r.ts_arrival)
    assert first.observations["load.in_system_at_arrival"] == 0.0


def test_congestion_shows_up_in_the_load_observations():
    """Order effects only exist under saturation, so a triage study has to be able to
    tell a busy department from an idle one. Same service, 40x the arrival rate."""
    idle = _mean_queue_at_arrival(arrival_rate=0.05)
    busy = _mean_queue_at_arrival(arrival_rate=2.0)
    assert busy > idle
    assert idle < 2.0  # service ~1.0, arrivals every ~20: essentially no queue


# ── Flattening to CSV ──────────────────────────────────────────────────────────


def _record(observations: dict) -> PatientRecord:
    n = _SCHEMA.n
    zeros = tuple([0] * n)
    floats = tuple([0.0] * n)
    return PatientRecord(
        patient_id=0,
        ground_truth=zeros,
        ts_arrival=0.0,
        ts_model_start=0.0,
        ts_model_end=0.0,
        ts_doctor_start=0.0,
        ts_doctor_end=0.0,
        ai_labels=zeros,
        ai_confidence=floats,
        ai_uncertainty=floats,
        doc_labels=zeros,
        doc_confidence=floats,
        doc_uncertainty=floats,
        priority_key=0.0,
        observations=observations,
    )


def _read_csv(path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def test_per_label_observation_expands_one_column_per_label(tmp_path):
    col = MetricsCollector(
        records=[_record({"ai.mutual_information": tuple(0.5 for _ in _LABELS)})]
    )
    path = tmp_path / "r.csv"
    col.write_csv(path, _LABELS)
    header, rows = _read_csv(path)
    for name in _LABELS:
        assert f"ai.mutual_information__{name}" in header
    assert rows[0][header.index(f"ai.mutual_information__{_LABELS[0]}")] == "0.500000"


def test_scalar_observation_is_one_column(tmp_path):
    col = MetricsCollector(records=[_record({"ai.n_members": 5, "ctx.sex": "F"})])
    path = tmp_path / "r.csv"
    col.write_csv(path, _LABELS)
    header, rows = _read_csv(path)
    assert header.count("ai.n_members") == 1
    assert rows[0][header.index("ai.n_members")] == "5"
    assert rows[0][header.index("ctx.sex")] == "F"


def test_records_with_disjoint_keys_stay_writable(tmp_path):
    """A study may swap the AI between runs, so records legitimately carry different
    keys. Missing cells must be blank, never a shifted row."""
    col = MetricsCollector(
        records=[_record({"ai.alpha": 1.0}), _record({"ai.beta": 2.0})]
    )
    path = tmp_path / "r.csv"
    col.write_csv(path, _LABELS)
    header, rows = _read_csv(path)
    assert all(len(row) == len(header) for row in rows)
    assert rows[0][header.index("ai.alpha")] == "1.000000"
    assert rows[0][header.index("ai.beta")] == ""
    assert rows[1][header.index("ai.alpha")] == ""
    assert rows[1][header.index("ai.beta")] == "2.000000"


def test_mixed_shape_key_falls_back_to_one_json_column(tmp_path):
    """Same key per-label in one record and scalar in another: a single JSON column,
    so a column can never silently misalign with the label axis."""
    col = MetricsCollector(
        records=[_record({"ai.x": tuple(1.0 for _ in _LABELS)}), _record({"ai.x": 3.0})]
    )
    path = tmp_path / "r.csv"
    col.write_csv(path, _LABELS)
    header, rows = _read_csv(path)
    assert header.count("ai.x") == 1
    assert f"ai.x__{_LABELS[0]}" not in header
    assert all(len(row) == len(header) for row in rows)


def test_unflattenable_value_is_json_encoded(tmp_path):
    col = MetricsCollector(records=[_record({"ai.matrix": {"a": [1, 2]}})])
    path = tmp_path / "r.csv"
    col.write_csv(path, _LABELS)
    header, rows = _read_csv(path)
    assert rows[0][header.index("ai.matrix")] == '{"a": [1, 2]}'


def test_empty_observations_keep_the_legacy_header(tmp_path):
    """No observations → the CSV is exactly what it was before this layer existed."""
    col = MetricsCollector(records=[_record({})])
    path = tmp_path / "r.csv"
    col.write_csv(path, _LABELS)
    header, _rows = _read_csv(path)
    assert header[0] == "patient_id"
    assert header[-1] == f"doc_unc__{_LABELS[-1]}"


def test_output_model_extra_is_the_documented_channel():
    """A hand-rolled actor needs no framework change to publish an observable."""
    out = ModelOutput(
        diagnosis=Diagnosis(labels=(1,) * _SCHEMA.n, confidence=(0.9,) * _SCHEMA.n),
        uncertainty=(0.1,) * _SCHEMA.n,
        extra={"my_new_signal": 42},
    )
    collector = MetricsCollector()
    from hai_sim.core.types import Patient

    patient = Patient(
        id=0,
        images=(np.zeros((1, 1)),),
        ground_truth=(0,) * _SCHEMA.n,
        context=PatientContext(patient_id=0, arrival_time=0.0, age=61.0, sex="M"),
    )
    collector.add(
        patient=patient, ts_arrival=0.0, ts_model_start=0.0, ts_model_end=1.0,
        ts_doctor_start=1.0, ts_doctor_end=2.0, ai_out=out, doc_out=out, priority_key=0.0,
    )
    rec = collector.records[0]
    assert rec.observations["ai.my_new_signal"] == 42
    assert rec.observations["doc.my_new_signal"] == 42
    assert rec.observations["ctx.age"] == 61.0
    assert rec.observations["ctx.sex"] == "M"
