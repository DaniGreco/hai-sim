from __future__ import annotations

import numpy as np

from hai_sim.core.types import LabelSchema
from hai_sim.doctors.mock import RandomMockDoctor
from hai_sim.models.mock import RandomMockAI


def _img():
    return [np.zeros((1, 1), dtype=np.uint8)]


def test_mock_ai_shape():
    schema = LabelSchema.default_v0()
    ai = RandomMockAI(schema=schema, rng=np.random.default_rng(0))
    out = ai.diagnose(_img())
    assert len(out.diagnosis.labels) == schema.n
    assert len(out.diagnosis.confidence) == schema.n
    assert len(out.uncertainty) == schema.n
    assert all(0.0 <= c <= 1.0 for c in out.diagnosis.confidence)
    assert all(0.0 <= u <= 1.0 for u in out.uncertainty)


def test_mock_doctor_shape_and_service_time():
    schema = LabelSchema.default_v0()
    doc = RandomMockDoctor(schema=schema, rng=np.random.default_rng(0))
    out = doc.diagnose(_img())
    assert len(out.diagnosis.labels) == schema.n
    t = doc.service_time(_img())
    assert t > 0


def test_seeded_mocks_are_deterministic():
    schema = LabelSchema.default_v0()
    a = RandomMockAI(schema=schema, rng=np.random.default_rng(123))
    b = RandomMockAI(schema=schema, rng=np.random.default_rng(123))
    assert a.diagnose(_img()).diagnosis.confidence == b.diagnose(_img()).diagnosis.confidence
