"""Swappable KPI bundle registry + the AIxDoctorxGT joint contingency table."""

from __future__ import annotations

import pytest

from hai_sim.config.scenario import ScenarioConfig
from hai_sim.metrics.collector import PatientRecord
from hai_sim.metrics.kpi import (
    DEFAULT_KPI_BUNDLES,
    KPI_BUNDLES,
    JointTable,
    KPIBundle,
    compute_report,
)


def _rec(gt, ai, doc, t: float = 1.0) -> PatientRecord:
    z = tuple(0.0 for _ in gt)
    return PatientRecord(
        patient_id=0,
        ground_truth=gt,
        ts_arrival=0.0,
        ts_model_start=0.0,
        ts_model_end=0.0,
        ts_doctor_start=0.0,
        ts_doctor_end=t,
        ai_labels=ai,
        ai_confidence=z,
        ai_uncertainty=z,
        doc_labels=doc,  # hybrid = the doctor's final call
        doc_confidence=z,
        doc_uncertainty=z,
        priority_key=0.0,
    )


# --- registry / selection ----------------------------------------------------


def test_default_report_has_all_bundles():
    report = compute_report([_rec((1, 0), (1, 0), (1, 0))])
    assert tuple(report.keys()) == DEFAULT_KPI_BUNDLES
    assert set(KPI_BUNDLES) == set(DEFAULT_KPI_BUNDLES)


def test_report_selects_subset():
    report = compute_report([_rec((1, 0), (1, 0), (1, 0))], ["system", "joint_table"])
    assert set(report.keys()) == {"system", "joint_table"}
    assert "actors" not in report


def test_unknown_bundle_raises():
    with pytest.raises(ValueError, match="unknown KPI bundle"):
        compute_report([_rec((1,), (1,), (1,))], ["nope"])


def test_bundles_satisfy_protocol():
    assert all(isinstance(b, KPIBundle) for b in KPI_BUNDLES.values())


def test_system_bundle_matches_record_count():
    recs = [_rec((1, 0), (1, 0), (1, 0), t=float(i + 1)) for i in range(4)]
    sysm = compute_report(recs, ["system"])["system"]
    assert sysm["n_patients"] == 4.0
    assert sysm["sim_duration"] == 4.0


# --- joint table -------------------------------------------------------------


def test_joint_table_cells():
    # gt=(1,1,0,0), ai=(1,0,1,0), doc=(1,1,0,0):
    #   L0 gt1 ai1 doc1 | L1 gt1 ai0 doc1 | L2 gt0 ai1 doc0 | L3 gt0 ai0 doc0
    jt = compute_report([_rec((1, 1, 0, 0), (1, 0, 1, 0), (1, 1, 0, 0))], ["joint_table"])[
        "joint_table"
    ]
    assert isinstance(jt, JointTable)
    assert jt.n_entries == 4
    assert jt.ai_confusion == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert jt.doctor_confusion == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}
    assert jt.joint["gt1_ai1_doc1"] == 1
    assert jt.joint["gt1_ai0_doc1"] == 1
    assert jt.joint["gt0_ai1_doc0"] == 1
    assert jt.joint["gt0_ai0_doc0"] == 1
    # the 8 cells partition the labeled entries
    assert sum(jt.joint.values()) == jt.n_entries


def test_joint_table_skips_unannotated():
    jt = compute_report([_rec((1, -1, 0), (1, 1, 0), (1, 0, 0))], ["joint_table"])[
        "joint_table"
    ]
    assert jt.n_entries == 2  # the -1 label is not scored


# --- config wiring -----------------------------------------------------------


def test_scenario_kpis_field_selects_bundles():
    sc = ScenarioConfig.model_validate(
        {
            "name": "k",
            "seed": 0,
            "label_schema": "default_v0",
            "kpis": ["system", "team"],
            "patient_generator": {"kind": "poisson", "arrival_rate": 0.5, "n_patients": 3},
            "ai": {"kind": "random_mock"},
            "doctor": {"kind": "random_mock"},
            "sort": {"kind": "random"},
        }
    )
    assert sc.kpis == ("system", "team")
    report = compute_report([_rec((1, 0), (1, 0), (1, 0))], sc.kpis)
    assert set(report.keys()) == {"system", "team"}
