"""Cache-replay AI: real ensemble predictions from an `InferenceCache` (no re-inference).

For a study over personas / triage / threshold, the AI's predictions on each image are
identical across runs — only the doctor and the queue change. Re-running the 5-member
ensemble (~5-6 s/image) for every run would waste hours. `CachedEnsembleAI` instead serves
each image's cached per-member probabilities through the same swappable `UQMethod`,
reconstructing exactly the `ModelOutput` a live `TorchXRayVisionAI` would produce — real NIH
predictions, no torch, seconds not hours. The patient carries its **image id** in
``context.example_id`` (set by `CachedPatientGenerator`) and this actor resolves it back to a
cache row, so AI and generator stay aligned as long as they load the same cache — and the
telemetry keeps the real image identity rather than a run-local counter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from hai_sim.core.base import BaseDiagnosticActor
from hai_sim.core.types import Image, LabelSchema, ModelOutput, PatientContext
from hai_sim.models._finalize import model_output_from_uq
from hai_sim.uq.methods import EntropyUQ

if TYPE_CHECKING:
    from hai_sim.analysis.inference_cache import InferenceCache
    from hai_sim.core.protocols import UQMethod


class CachedEnsembleAI(BaseDiagnosticActor):
    """Serve cached per-member probabilities through a swappable UQ method."""

    def __init__(
        self,
        cache: InferenceCache,
        schema: LabelSchema,
        uq: UQMethod | None = None,
        decision_threshold: float = 0.5,
        service_time: float = 0.05,
        name: str | None = None,
    ) -> None:
        if tuple(cache.pathologies) != schema.labels:
            raise ValueError(
                "cache pathologies do not match the active LabelSchema; use label_schema: xrv"
            )
        image_ids = tuple(str(x) for x in cache.image_ids)
        if len(set(image_ids)) != len(image_ids):
            raise ValueError(
                "inference cache has duplicate image_ids; the replay AI resolves a patient's "
                "image by id, so ids must be unique"
            )
        resolved = name if name is not None else f"CachedEnsembleAI({'+'.join(cache.weights)})"
        super().__init__(name=resolved, service_time=service_time)
        self.cache = cache
        self.schema = schema
        self.uq: UQMethod = uq if uq is not None else EntropyUQ()
        self.decision_threshold = decision_threshold
        self._row_of: dict[str, int] = {img_id: i for i, img_id in enumerate(image_ids)}

    def diagnose(
        self,
        images: Sequence[Image],
        *,
        context: PatientContext | None = None,
        ai_hint: ModelOutput | None = None,
    ) -> ModelOutput:
        if context is None or context.example_id is None:
            raise ValueError(
                "CachedEnsembleAI needs a PatientContext carrying the image id in "
                "example_id (CachedPatientGenerator sets it)"
            )
        idx = self._row_of.get(context.example_id)
        if idx is None:
            raise ValueError(
                f"image id {context.example_id!r} is not in this inference cache; "
                "AI and patient generator must load the same cache"
            )
        member_probs = np.asarray(self.cache.member_probs[idx], dtype=float)  # (M, L)
        return model_output_from_uq(
            self.uq,
            member_probs,
            self.decision_threshold,
            extra={"cache_index": idx, "image_id": context.example_id},
        )


class CompositeCachedAI(BaseDiagnosticActor):
    """Serve an ensemble from *several* single-model caches, composed by image id.

    The cache matrix stores one file per atomic model (`hai_sim.analysis.cache_matrix`);
    the ensemble is assembled here at replay time — each member's probabilities are
    looked up by image id and stacked along the member axis, then run through the same
    swappable `UQMethod` a live multi-weight `TorchXRayVisionAI` would use, reproducing
    its `ModelOutput` exactly. Dropping or adding a member is a change of *which caches
    are passed in*, never a rebuild — which is what made dropping the leaky NIH member a
    one-line decision.

    Only images every member covers can be served, so the actor is defined over the
    **intersection** of the members' ids (a study run under ``cache_mode: require`` has
    already guaranteed that intersection contains everything it will ask for).
    """

    def __init__(
        self,
        caches: Sequence[InferenceCache],
        schema: LabelSchema,
        uq: UQMethod | None = None,
        decision_threshold: float = 0.5,
        service_time: float = 0.05,
        name: str | None = None,
    ) -> None:
        if not caches:
            raise ValueError("CompositeCachedAI needs at least one member cache")
        for c in caches:
            if tuple(c.pathologies) != schema.labels:
                raise ValueError(
                    "a member cache's pathologies do not match the active LabelSchema; "
                    "use label_schema: xrv"
                )
        later = [set(c.image_ids) for c in caches[1:]]
        common = [img for img in caches[0].image_ids if all(img in s for s in later)]
        if not common:
            raise ValueError("member caches share no common image id to compose")
        blocks: list[np.ndarray] = []
        for c in caches:
            pos = {img: r for r, img in enumerate(c.image_ids)}
            rows = [pos[img] for img in common]
            blocks.append(np.asarray(c.member_probs, dtype=float)[rows])  # (N, M_c, L)
        probs = np.concatenate(blocks, axis=1)  # (N, sum M_c, L)

        n_members = int(probs.shape[1])
        resolved = name or f"CompositeCachedAI({n_members} members / {len(caches)} caches)"
        super().__init__(name=resolved, service_time=service_time)
        self.schema = schema
        self.uq: UQMethod = uq if uq is not None else EntropyUQ()
        self.decision_threshold = decision_threshold
        self._probs = probs
        self._row: dict[str, int] = {img: r for r, img in enumerate(common)}

    @property
    def covered_ids(self) -> frozenset[str]:
        return frozenset(self._row)

    def diagnose(
        self,
        images: Sequence[Image],
        *,
        context: PatientContext | None = None,
        ai_hint: ModelOutput | None = None,
    ) -> ModelOutput:
        if context is None or context.example_id is None:
            raise ValueError(
                "CompositeCachedAI needs a PatientContext carrying the image id in "
                "example_id (a dataset patient generator sets it)"
            )
        idx = self._row.get(context.example_id)
        if idx is None:
            raise ValueError(
                f"image id {context.example_id!r} is not covered by every composed cache"
            )
        member_probs = self._probs[idx]  # (M_total, L)
        return model_output_from_uq(
            self.uq,
            member_probs,
            self.decision_threshold,
            extra={"image_id": context.example_id, "n_members": int(member_probs.shape[0])},
        )
