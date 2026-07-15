"""Team KPIs: complementarity gap + appropriate-reliance (RAIR/RSR) from records."""

from __future__ import annotations

from hai_sim.metrics.collector import PatientRecord
from hai_sim.metrics.kpi import compute_kpis


def _rec(gt, ai, solo, team, t=1.0) -> PatientRecord:
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
        doc_labels=team,  # the doctor stage's actual output = team (aided)
        doc_confidence=z,
        doc_uncertainty=z,
        priority_key=0.0,
        doctor_solo_labels=solo,  # the unaided counterfactual
    )


def test_reliance_ratios_and_opportunity_counts():
    # gt all-positive; each label places an entry in a specific 2x2 cell.
    # r1: L0 doc-wrong/AI-right (adopt→right), L1 doc-right/AI-wrong (keep→right),
    #     L2 both right, L3 both wrong.
    r1 = _rec(gt=(1, 1, 1, 1), ai=(1, 0, 1, 0), solo=(0, 1, 1, 0), team=(1, 1, 1, 1))
    # r2: L0 doc-wrong/AI-right but team WRONG (under-reliance),
    #     L1 doc-right/AI-wrong but team WRONG (over-reliance).
    r2 = _rec(gt=(1, 1), ai=(1, 0), solo=(0, 1), team=(0, 0))
    tm = compute_kpis([r1, r2]).team

    assert tm.n_ai_reliance_opportunities == 2
    assert tm.n_self_reliance_opportunities == 2
    assert tm.rair == 0.5  # 1 of 2 AI-reliance opportunities harvested
    assert tm.rsr == 0.5  # 1 of 2 self-reliance opportunities kept
    assert tm.over_reliance == 0.5
    assert tm.under_reliance == 0.5


def test_complementarity_gap_positive_when_team_beats_both():
    # AI-solo 2/4, doctor-solo 2/4 (different labels), team 4/4.
    r = _rec(gt=(1, 1, 1, 1), ai=(1, 0, 1, 0), solo=(0, 1, 1, 0), team=(1, 1, 1, 1))
    tm = compute_kpis([r]).team
    assert tm.complementarity_gap_accuracy == 0.5  # 1.0 - max(0.5, 0.5)
    assert isinstance(tm.complementarity_gap_f1, float)


def test_unassisted_records_have_no_reliance_opportunities():
    # doctor_solo_labels=None -> "doctor" falls back to team; no divergence to score.
    z = (0.0, 0.0)
    rec = PatientRecord(
        patient_id=0,
        ground_truth=(1, 0),
        ts_arrival=0.0,
        ts_model_start=0.0,
        ts_model_end=0.0,
        ts_doctor_start=0.0,
        ts_doctor_end=1.0,
        ai_labels=(1, 0),
        ai_confidence=z,
        ai_uncertainty=z,
        doc_labels=(1, 0),
        doc_confidence=z,
        doc_uncertainty=z,
        priority_key=0.0,
    )
    tm = compute_kpis([rec]).team
    assert tm.rair is None
    assert tm.rsr is None
    assert tm.over_reliance is None
    assert tm.under_reliance is None
    assert tm.n_ai_reliance_opportunities == 0
    assert tm.n_self_reliance_opportunities == 0
