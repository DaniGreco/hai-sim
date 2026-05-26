"""Structural contracts for swappable modules.

Protocols are the *public contract* of the framework. Anything that walks like
a `DiagnosticActor` is one — no inheritance required. Use the `BaseDiagnosticActor`
ABC in `hai_sim.core.base` if you want a base class with default behavior.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from hai_sim.core.types import Image, ModelOutput, Patient, PatientContext


@runtime_checkable
class DiagnosticActor(Protocol):
    """Unified contract for any entity that produces a diagnosis from images.

    Implemented by AI models and by doctor digital twins. The `ai_hint` kwarg
    is only meaningful for doctors operating in assisted mode; AI models
    should accept it and ignore it.
    """

    name: str

    def diagnose(
        self,
        images: Sequence[Image],
        *,
        context: PatientContext | None = None,
        ai_hint: ModelOutput | None = None,
    ) -> ModelOutput: ...

    def service_time(
        self,
        images: Sequence[Image],
        context: PatientContext | None = None,
    ) -> float: ...


@runtime_checkable
class SortStrategy(Protocol):
    """Computes the priority key used by the triaged PriorityStore.

    Lower priority value → served first (SimPy `PriorityStore` convention).
    Implementations decide how to collapse the per-label uncertainty vector
    into a scalar (mean, max, weighted, etc.).
    """

    name: str

    def priority(self, output: ModelOutput, patient: Patient) -> float: ...
