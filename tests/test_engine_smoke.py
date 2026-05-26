"""End-to-end smoke test of the SimPy engine using mocks."""

from __future__ import annotations

import numpy as np

from hai_sim.core.types import LabelSchema
from hai_sim.doctors.mock import RandomMockDoctor
from hai_sim.metrics.kpi import compute_kpis
from hai_sim.models.mock import RandomMockAI
from hai_sim.sim.engine import SimulationEngine
from hai_sim.sim.patient_generator import PoissonPatientGenerator
from hai_sim.sort.random import RandomSort
from hai_sim.sort.uncertainty import UncertaintyDescSort


def _engine(sort_kind: str, seed: int = 0, n_patients: int = 30):
    schema = LabelSchema.default_v0()
    rng_p = np.random.default_rng(seed)
    rng_a = np.random.default_rng(seed + 1)
    rng_d = np.random.default_rng(seed + 2)
    rng_s = np.random.default_rng(seed + 3)
    gen = PoissonPatientGenerator(
        schema=schema, arrival_rate=0.5, n_patients=n_patients, prevalence=0.2, rng=rng_p
    )
    ai = RandomMockAI(schema=schema, rng=rng_a)
    doctor = RandomMockDoctor(schema=schema, rng=rng_d)
    sort = RandomSort(rng=rng_s) if sort_kind == "random" else UncertaintyDescSort()
    return SimulationEngine(
        patient_generator=gen, ai=ai, doctor=doctor, sort_strategy=sort
    )


def test_engine_runs_to_completion_random_sort():
    res = _engine("random").run()
    assert len(res.collector.records) == 30
    for r in res.collector.records:
        assert r.ts_arrival <= r.ts_model_start <= r.ts_model_end <= r.ts_doctor_start <= r.ts_doctor_end


def test_engine_runs_to_completion_uncertainty_sort():
    res = _engine("uncertainty_desc").run()
    assert len(res.collector.records) == 30


def test_kpis_have_expected_keys():
    res = _engine("random").run()
    kpis = compute_kpis(res.collector.records)
    assert kpis.n_patients == 30
    assert kpis.sim_duration > 0
    assert 0.0 <= kpis.ai.f1_macro <= 1.0
    assert 0.0 <= kpis.doctor.f1_macro <= 1.0


def test_uncertainty_sort_serves_high_uncertainty_earlier():
    """Mean uncertainty at the doctor station should be decreasing over time
    when we sort by uncertainty descending (apart from queue dynamics)."""
    res = _engine("uncertainty_desc", n_patients=200).run()
    records = sorted(res.collector.records, key=lambda r: r.ts_doctor_start)
    if len(records) < 20:
        return
    first_half_unc = [sum(r.ai_uncertainty) / len(r.ai_uncertainty) for r in records[: len(records) // 2]]
    second_half_unc = [sum(r.ai_uncertainty) / len(r.ai_uncertainty) for r in records[len(records) // 2 :]]
    # Loose check: first half should have higher mean uncertainty on average.
    # (Not strict because Poisson arrivals + per-patient stochasticity blur the signal.)
    assert sum(first_half_unc) / len(first_half_unc) > sum(second_half_unc) / len(second_half_unc)
