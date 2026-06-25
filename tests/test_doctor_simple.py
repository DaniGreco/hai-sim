"""FixedAccuracyDoctor: controllable ground-truth-based reader."""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.core.types import LabelSchema, PatientContext
from hai_sim.doctors.simple import FixedAccuracyDoctor


def _ctx(gt: tuple[int, ...]) -> PatientContext:
    return PatientContext(patient_id=0, arrival_time=0.0, ground_truth=gt)


def test_perfect_doctor_reproduces_truth() -> None:
    schema = LabelSchema.default_v0()
    doc = FixedAccuracyDoctor(
        schema, rng=np.random.default_rng(0), sensitivity=1.0, specificity=1.0
    )
    gt = (1, 0, 1, 0, 1)
    out = doc.diagnose([np.zeros((1, 1))], context=_ctx(gt))
    assert out.diagnosis.labels == gt


def test_unannotated_label_emits_zero() -> None:
    schema = LabelSchema.default_v0()
    doc = FixedAccuracyDoctor(schema, sensitivity=1.0, specificity=1.0)
    gt = (1, -1, 0, -1, 1)
    out = doc.diagnose([np.zeros((1, 1))], context=_ctx(gt))
    assert out.diagnosis.labels == (1, 0, 0, 0, 1)


def test_requires_ground_truth() -> None:
    schema = LabelSchema.default_v0()
    doc = FixedAccuracyDoctor(schema)
    with pytest.raises(ValueError, match="ground_truth"):
        doc.diagnose([np.zeros((1, 1))], context=PatientContext(patient_id=0, arrival_time=0.0))


def test_sensitivity_holds_in_aggregate() -> None:
    schema = LabelSchema(labels=("a",))
    doc = FixedAccuracyDoctor(
        schema, rng=np.random.default_rng(1), sensitivity=0.8, specificity=1.0
    )
    hits = sum(
        doc.diagnose([np.zeros((1, 1))], context=_ctx((1,))).diagnosis.labels[0]
        for _ in range(3000)
    )
    assert 0.77 < hits / 3000 < 0.83


def test_per_label_vectors() -> None:
    schema = LabelSchema(labels=("a", "b"))
    doc = FixedAccuracyDoctor(
        schema, rng=np.random.default_rng(0), sensitivity=(1.0, 1.0), specificity=(1.0, 1.0)
    )
    out = doc.diagnose([np.zeros((1, 1))], context=_ctx((1, 0)))
    assert out.diagnosis.labels == (1, 0)
