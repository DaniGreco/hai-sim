"""KPI must skip unannotated (-1) ground-truth entries."""

from __future__ import annotations

from hai_sim.metrics.collector import PatientRecord
from hai_sim.metrics.kpi import compute_kpis


def _rec(gt: tuple[int, ...], ai: tuple[int, ...], doc: tuple[int, ...], t: float) -> PatientRecord:
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
        doc_labels=doc,
        doc_confidence=z,
        doc_uncertainty=z,
        priority_key=0.0,
    )


def test_minus_one_is_not_scored() -> None:
    # Only mismatch is on the masked label 1 → no error counted anywhere.
    records = [_rec((1, -1), (1, 1), (1, 1), 1.0)]
    k = compute_kpis(records)
    assert k.ai.n_fp == 0
    assert k.ai.n_fn == 0
    assert k.ai.ttfe is None
    assert k.ai.accuracy == 1.0  # the single scored entry (label 0) is a TP


def test_real_error_still_counts_with_masking() -> None:
    # label 0: gt1/pred0 = FN (real error); label 1 masked.
    records = [_rec((1, -1), (0, 1), (0, 1), 2.0)]
    k = compute_kpis(records)
    assert k.ai.n_fn == 1
    assert k.ai.ttfe == 2.0
