"""The `follow_when_certain` policy must override the doctor's call on labels
where the AI's uncertainty is below the threshold, and leave the rest alone."""

from __future__ import annotations

import numpy as np

from hai_sim.core.types import Diagnosis, LabelSchema, ModelOutput
from hai_sim.doctors.mock import RandomMockDoctor


def _ai_hint(labels: tuple[int, ...], uncertainty: tuple[float, ...]) -> ModelOutput:
    return ModelOutput(
        diagnosis=Diagnosis(labels=labels, confidence=tuple([0.99] * len(labels))),
        uncertainty=uncertainty,
    )


def test_noop_policy_does_not_change_doctor_labels():
    schema = LabelSchema.default_v0()
    doc = RandomMockDoctor(
        schema=schema,
        rng=np.random.default_rng(0),
        use_ai_hint=True,
        assisted_policy="noop",
    )
    # Make AI fully certain on every label, but policy is noop → doctor ignores.
    hint = _ai_hint(labels=(1, 1, 1, 1, 1), uncertainty=(0.0, 0.0, 0.0, 0.0, 0.0))
    without = RandomMockDoctor(schema=schema, rng=np.random.default_rng(0)).diagnose([np.zeros((1, 1))])
    with_hint = doc.diagnose([np.zeros((1, 1))], ai_hint=hint)
    assert with_hint.diagnosis.labels == without.diagnosis.labels
    assert with_hint.extra.get("ai_hint_seen") is True


def test_follow_when_certain_copies_ai_label_below_threshold():
    schema = LabelSchema.default_v0()
    doc = RandomMockDoctor(
        schema=schema,
        rng=np.random.default_rng(0),
        use_ai_hint=True,
        assisted_policy="follow_when_certain",
        ai_hint_uncertainty_threshold=0.3,
    )
    # AI says all 1s with low uncertainty on positions 0, 2 and high uncertainty elsewhere.
    hint = _ai_hint(labels=(1, 1, 1, 1, 1), uncertainty=(0.1, 0.9, 0.1, 0.9, 0.9))
    out = doc.diagnose([np.zeros((1, 1))], ai_hint=hint)
    # On low-uncertainty labels (0 and 2) we must see the AI's label.
    assert out.diagnosis.labels[0] == 1
    assert out.diagnosis.labels[2] == 1
    assert out.extra["ai_hint_followed_labels"] == (0, 2)


def test_follow_when_certain_with_no_hint_falls_back_silently():
    schema = LabelSchema.default_v0()
    doc = RandomMockDoctor(
        schema=schema,
        rng=np.random.default_rng(0),
        use_ai_hint=True,
        assisted_policy="follow_when_certain",
    )
    out = doc.diagnose([np.zeros((1, 1))], ai_hint=None)
    assert "ai_hint_seen" not in out.extra
    assert "ai_hint_followed_labels" not in out.extra
