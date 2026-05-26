"""Optional ABC base class for `DiagnosticActor` implementations.

Inheriting is *not required* — any duck-typed class satisfies the Protocol.
Use this when you want the default `service_time` and `name` behavior so you
only have to implement `diagnose`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from hai_sim.core.types import Image, ModelOutput, PatientContext


class BaseDiagnosticActor(ABC):
    """Default-laden base class for in-framework actor implementations."""

    name: str = ""
    _constant_service_time: float = 1.0

    def __init__(self, name: str | None = None, service_time: float | None = None) -> None:
        self.name = name if name is not None else type(self).__name__
        if service_time is not None:
            self._constant_service_time = service_time

    @abstractmethod
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
    ) -> float:
        return self._constant_service_time
