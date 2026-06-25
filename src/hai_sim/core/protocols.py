"""Structural contracts for swappable modules.

Protocols are the *public contract* of the framework. Anything that walks like
a `DiagnosticActor` is one — no inheritance required. Use the `BaseDiagnosticActor`
ABC in `hai_sim.core.base` if you want a base class with default behavior.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from hai_sim.core.types import Image, ModelOutput, Patient, PatientContext

if TYPE_CHECKING:
    import numpy as np


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


@runtime_checkable
class UQMethod(Protocol):
    """Turns a stack of per-member probabilities into (probs, uncertainty).

    ``member_probs`` has shape ``(n_members, n_labels)``: one row per ensemble
    member (or one row for a single forward pass). The method returns the fused
    per-label probability vector and a per-label uncertainty vector, plus an
    ``extra`` dict of any auxiliary measures (e.g. the full entropy / mutual
    information decomposition). Keeping uncertainty per-label is deliberate: the
    `SortStrategy` is what collapses it to a scalar.
    """

    name: str

    def estimate(
        self, member_probs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]: ...
