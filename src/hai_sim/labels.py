"""Label-view projection: study a subset of labels, or collapse to binary abnormal.

The model runs in its native label space (e.g. the 18-pathology xrv schema); a
`LabelProjection` maps that native output — and the ground truth — down to the
*active* label set the simulation scores on. This is how single-label vs
multi-label studies are expressed without retraining or fabricating per-label
models: ``subset`` keeps K native labels, ``binary_abnormal`` collapses them to a
single "Abnormal" label (any finding). ``full`` is the identity (no projection).

Applied uniformly: the patient generator's ground truth is projected, and any AI
actor is wrapped in a `ProjectedActor` so its ModelOutput comes out in the active
space. The doctor, sort, severity and KPIs are then built directly on the active
schema, so nothing downstream needs to know a projection happened. A single-label
AI is thus ``subset`` with one target; an "ensemble of single-label models" over a
multi-label net is expressed as one such projected study per label. Genuinely
distinct per-label model backends (MedGemma prompts, per-pathology CheXNet) plug in
later behind the same active-schema contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

from hai_sim.core.types import Diagnosis, LabelSchema, ModelOutput

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from hai_sim.core.protocols import DiagnosticActor
    from hai_sim.core.types import Image, Patient, PatientContext

Mode = Literal["full", "subset", "binary_abnormal"]
Combine = Literal["noisy_or", "max"]


@dataclass(frozen=True)
class LabelProjection:
    """Maps native (base-schema) vectors to the active study schema."""

    base: LabelSchema
    active: LabelSchema
    indices: tuple[int, ...]
    mode: Mode
    combine: Combine = "noisy_or"

    @property
    def is_identity(self) -> bool:
        return self.mode == "full"

    def _select_float(self, values: Sequence[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.asarray([arr[i] for i in self.indices], dtype=float)

    def _select_int(self, values: Sequence[int]) -> np.ndarray:
        arr = np.asarray(values, dtype=int)
        return np.asarray([arr[i] for i in self.indices], dtype=int)

    def project_probs(self, probs: Sequence[float]) -> np.ndarray:
        sel = self._select_float(probs)
        if self.mode != "binary_abnormal":
            return sel
        if sel.size == 0:
            return np.array([0.0], dtype=float)
        if self.combine == "max":
            p = float(np.max(sel))
        else:  # noisy_or: P(any) = 1 - prod(1 - p_i)
            p = float(1.0 - np.prod(1.0 - np.clip(sel, 0.0, 1.0)))
        return np.array([p], dtype=float)

    def project_uncertainty(self, unc: Sequence[float]) -> np.ndarray:
        sel = self._select_float(unc)
        if self.mode != "binary_abnormal":
            return sel
        return np.array([float(np.mean(sel)) if sel.size else 0.0], dtype=float)

    def project_labels(self, labels: Sequence[int]) -> np.ndarray:
        sel = self._select_int(labels)
        if self.mode != "binary_abnormal":
            return sel
        return np.array([1 if bool(np.any(sel == 1)) else 0], dtype=int)

    def project_gt(self, gt: Sequence[int]) -> tuple[int, ...]:
        sel = self._select_int(gt)
        if self.mode != "binary_abnormal":
            return tuple(int(x) for x in sel)
        annotated = sel[sel != -1]
        if annotated.size == 0:
            return (-1,)  # no annotated contributor → abnormal-status unknown
        return (1 if bool(np.any(annotated == 1)) else 0,)

    def project_output(self, out: ModelOutput) -> ModelOutput:
        return ModelOutput(
            diagnosis=Diagnosis(
                labels=tuple(int(x) for x in self.project_labels(out.diagnosis.labels)),
                confidence=tuple(float(x) for x in self.project_probs(out.diagnosis.confidence)),
            ),
            uncertainty=tuple(float(x) for x in self.project_uncertainty(out.uncertainty)),
            extra=out.extra,
            embedding=out.embedding,
        )


def _index_of(schema: LabelSchema, label: str) -> int:
    try:
        return schema.labels.index(label)
    except ValueError:
        raise ValueError(f"label {label!r} not in base schema {schema.labels}") from None


def build_projection(
    base: LabelSchema,
    mode: Mode = "full",
    targets: Sequence[str] | None = None,
    combine: Combine = "noisy_or",
) -> LabelProjection:
    if mode == "full":
        return LabelProjection(base, base, tuple(range(base.n)), "full", combine)
    if mode == "subset":
        if not targets:
            raise ValueError("label_view mode 'subset' requires non-empty 'targets'")
        indices = tuple(_index_of(base, t) for t in targets)
        return LabelProjection(base, LabelSchema(labels=tuple(targets)), indices, "subset", combine)
    if mode == "binary_abnormal":
        indices = (
            tuple(range(base.n)) if not targets else tuple(_index_of(base, t) for t in targets)
        )
        active = LabelSchema(labels=("Abnormal",))
        return LabelProjection(base, active, indices, "binary_abnormal", combine)
    raise ValueError(f"unknown label_view mode: {mode}")


class ProjectedActor:
    """Wraps a DiagnosticActor so its ModelOutput is projected to the active schema."""

    def __init__(
        self, inner: DiagnosticActor, projection: LabelProjection, name: str | None = None
    ) -> None:
        self.inner = inner
        self.projection = projection
        self.name = name or f"Projected[{'+'.join(projection.active.labels)}]({inner.name})"

    def diagnose(
        self,
        images: Sequence[Image],
        *,
        context: PatientContext | None = None,
        ai_hint: ModelOutput | None = None,
    ) -> ModelOutput:
        out = self.inner.diagnose(images, context=context, ai_hint=ai_hint)
        return self.projection.project_output(out)

    def service_time(
        self, images: Sequence[Image], context: PatientContext | None = None
    ) -> float:
        return self.inner.service_time(images, context)


def project_generator(
    generator: Iterable[tuple[float, Patient]], projection: LabelProjection
) -> Iterator[tuple[float, Patient]]:
    """Yield the generator's patients with ground truth projected to the active schema."""
    for delay, patient in generator:
        yield delay, replace(patient, ground_truth=projection.project_gt(patient.ground_truth))
