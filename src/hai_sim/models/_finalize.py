"""Shared tail for the real-AI wrappers: UQ estimate → threshold → `ModelOutput`.

`TorchXRayVisionAI`, `CachedEnsembleAI` and `MedGemmaAI` all end their ``diagnose``
the same way: run the stack of per-member probabilities through the swappable
`UQMethod`, replace uncovered-label NaNs with a benign negative, threshold to a binary
call, and assemble a `ModelOutput`. Only the ``extra`` bag differs (an embedding here, a
cache row there). This is that one shared tail; each wrapper passes its own ``extra``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from hai_sim.core.types import ModelOutput

if TYPE_CHECKING:
    from hai_sim.core.protocols import UQMethod


def model_output_from_uq(
    uq: UQMethod,
    member_probs: np.ndarray,
    decision_threshold: float,
    *,
    extra: dict[str, Any] | None = None,
    embedding: np.ndarray | None = None,
) -> ModelOutput:
    """Fuse ``member_probs`` through ``uq``, threshold, and build the `ModelOutput`.

    ``extra`` is merged after the always-present ``uq_method`` key and before the UQ
    method's own ``extra`` (so a decomposition key can never be shadowed), matching what
    each wrapper built inline. A pathology no member covers comes back NaN and is emitted
    as a benign negative.
    """
    probs, uncertainty, uq_extra = uq.estimate(member_probs)
    probs = np.nan_to_num(np.asarray(probs, dtype=float), nan=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0)
    full_extra: dict[str, Any] = {"uq_method": uq.name, **(extra or {}), **uq_extra}
    return ModelOutput.from_arrays(
        probs >= decision_threshold,
        probs,
        uncertainty,
        extra=full_extra,
        embedding=embedding,
    )
