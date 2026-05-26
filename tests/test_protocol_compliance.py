"""Mocks must satisfy the structural Protocols at runtime."""

from __future__ import annotations

import numpy as np

from hai_sim.core.protocols import DiagnosticActor, SortStrategy
from hai_sim.core.types import LabelSchema
from hai_sim.doctors.mock import RandomMockDoctor
from hai_sim.models.mock import RandomMockAI
from hai_sim.sort.random import RandomSort
from hai_sim.sort.uncertainty import UncertaintyDescSort


def test_mock_ai_is_diagnostic_actor():
    schema = LabelSchema.default_v0()
    ai = RandomMockAI(schema=schema, rng=np.random.default_rng(0))
    assert isinstance(ai, DiagnosticActor)


def test_mock_doctor_is_diagnostic_actor():
    schema = LabelSchema.default_v0()
    doc = RandomMockDoctor(schema=schema, rng=np.random.default_rng(0))
    assert isinstance(doc, DiagnosticActor)


def test_sort_strategies_satisfy_protocol():
    assert isinstance(RandomSort(np.random.default_rng(0)), SortStrategy)
    assert isinstance(UncertaintyDescSort(), SortStrategy)
