"""Doctor personas = capability x visibility x trust, + the confidence-gated policy."""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.config.scenario import CapabilityConfig, PersonaConfig, ScenarioConfig
from hai_sim.core.types import Diagnosis, LabelSchema, ModelOutput
from hai_sim.doctors.assisted import AssistedDoctor, ConfidenceGatedPoolPolicy
from hai_sim.doctors.simple import FixedAccuracyDoctor
from hai_sim.factory import build_from_scenario, build_persona
from hai_sim.personas import PERSONA_LIBRARY, resolve_persona
from hai_sim.sim.engine import SimulationEngine

_SCHEMA = LabelSchema.default_v0()


# --- library / resolution ----------------------------------------------------


def test_library_is_the_six_grid():
    assert set(PERSONA_LIBRARY) == {
        "solo_senior", "solo_junior",
        "senior_trusting", "senior_skeptical",
        "junior_trusting", "junior_skeptical",
    }
    # solo personas do not see the AI; the rest do
    assert PERSONA_LIBRARY["solo_senior"].sees_ai is False
    assert PERSONA_LIBRARY["senior_trusting"].sees_ai is True
    # experts discount more than novices at the same attitude (WOA senior < junior)
    assert (
        PERSONA_LIBRARY["senior_skeptical"].trust.weight_ai
        < PERSONA_LIBRARY["junior_skeptical"].trust.weight_ai
    )


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="unknown persona"):
        resolve_persona("nope")


def test_resolve_inline_passthrough():
    p = PersonaConfig(sees_ai=False)
    assert resolve_persona(p) is p


# --- build_persona -----------------------------------------------------------


def test_solo_persona_is_bare_reader():
    doc = build_persona("solo_senior", _SCHEMA, np.random.default_rng(0))
    assert isinstance(doc, FixedAccuracyDoctor)
    assert not isinstance(doc, AssistedDoctor)


def test_assisted_persona_wraps_in_assisted_doctor():
    doc = build_persona("senior_trusting", _SCHEMA, np.random.default_rng(0))
    assert isinstance(doc, AssistedDoctor)
    assert "senior_trusting" in doc.name


def test_capability_source_reader_is_gated():
    persona = PersonaConfig(capability=CapabilityConfig(source="reader", reader_id="R1"))
    with pytest.raises(ValueError, match="per-reader"):
        build_persona(persona, _SCHEMA, np.random.default_rng(0))


# --- confidence-gated policy -------------------------------------------------


def _out(labels, conf, unc=None) -> ModelOutput:
    return ModelOutput(
        diagnosis=Diagnosis(labels=labels, confidence=conf),
        uncertainty=unc if unc is not None else tuple(0.0 for _ in conf),
    )


def test_confidence_gated_listens_only_when_ai_is_confident():
    # doctor neutral (belief 0.5 on both labels); AI confident on L0 (0.9), not L1 (0.6).
    doctor_out = _out((0, 0), (0.5, 0.5))
    ai_hint = _out((1, 0), (0.9, 0.6))
    policy = ConfidenceGatedPoolPolicy(weight_ai=0.4, confidence_threshold=0.7)
    fused = policy.combine(doctor_out, ai_hint)
    # L0 gated in: 0.6*0.5 + 0.4*0.9 = 0.66 ; L1 gated out: stays the doctor's 0.5
    assert abs(fused.diagnosis.confidence[0] - 0.66) < 1e-9
    assert abs(fused.diagnosis.confidence[1] - 0.5) < 1e-9
    assert fused.extra["fusion_weight_ai"] == (0.4, 0.0)


def test_confidence_gated_rejects_bad_params():
    with pytest.raises(ValueError):
        ConfidenceGatedPoolPolicy(weight_ai=1.5)
    with pytest.raises(ValueError):
        ConfidenceGatedPoolPolicy(confidence_threshold=2.0)


# --- end-to-end via scenario -------------------------------------------------


def _scenario(persona) -> ScenarioConfig:
    return ScenarioConfig.model_validate(
        {
            "name": "persona_test",
            "seed": 0,
            "label_schema": "default_v0",
            "persona": persona,
            "patient_generator": {"kind": "poisson", "arrival_rate": 0.5, "n_patients": 10},
            "ai": {"kind": "random_mock"},
            "sort": {"kind": "random"},
        }
    )


def test_scenario_with_library_persona_runs():
    sc = _scenario("solo_junior")
    _schema, gen, ai, doctor, sort, sev, n_ai, n_doc = build_from_scenario(sc)
    assert isinstance(doctor, FixedAccuracyDoctor)
    res = SimulationEngine(
        patient_generator=gen, ai=ai, doctor=doctor, sort_strategy=sort,
        severity_model=sev, n_ai_workers=n_ai, n_doctor_workers=n_doc,
    ).run()
    assert len(res.collector.records) == 10


def test_scenario_with_inline_confidence_gated_persona_runs():
    sc = _scenario(
        {
            "capability": {"kind": "random_mock"},
            "sees_ai": True,
            "trust": {"kind": "confidence_gated", "weight_ai": 0.4, "confidence_threshold": 0.7},
        }
    )
    _schema, gen, ai, doctor, sort, sev, n_ai, n_doc = build_from_scenario(sc)
    assert isinstance(doctor, AssistedDoctor)
    res = SimulationEngine(
        patient_generator=gen, ai=ai, doctor=doctor, sort_strategy=sort,
        severity_model=sev, n_ai_workers=n_ai, n_doctor_workers=n_doc,
    ).run()
    assert len(res.collector.records) == 10
