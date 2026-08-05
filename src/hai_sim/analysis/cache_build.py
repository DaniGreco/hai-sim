"""Materialize a cache cell: run the model only over the images the cell is missing.

`materialize_cell` is the reusable *fill* core the standalone build command and the
study-time resolver both call. Given a cell and the dataset indices a study needs, it
loads whatever the cell already holds, runs the model only over the ids still missing,
appends them and saves — so a cell **grows** to cover new requests instead of being
rebuilt. The identity that makes "missing" well-defined lives in
`hai_sim.analysis.cache_matrix`.

The model builder and the dataset are injectable, so the orchestration is testable with
a fake model over a tiny dataset — no multi-gigabyte network, no torch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hai_sim.analysis.cache_matrix import (
    DEFAULT_CACHE_ROOT,
    CellSpec,
    dataset_cell_fingerprint,
    dataset_id,
    inspect_cell,
)
from hai_sim.analysis.inference_cache import (
    CacheableModel,
    InferenceCache,
    _image_id,
    build_inference_cache,
)

if TYPE_CHECKING:
    from hai_sim.config.scenario import AIModelConfig, PatientGenConfig
    from hai_sim.core.types import LabelSchema

ModelBuilder = Callable[..., CacheableModel]
"""Builds the atomic model for a cell from its single-model config + the schema."""


def _default_model_builder(cfg: AIModelConfig, schema: LabelSchema) -> CacheableModel:
    """Build the real backend through the factory, checking it can be cached."""
    import numpy as np

    from hai_sim.factory import build_ai

    model = build_ai(cfg, schema, np.random.default_rng(0))
    if not isinstance(model, CacheableModel):
        raise ValueError(
            f"ai.kind={cfg.kind!r} produces no member probabilities and cannot be cached"
        )
    return model


def _id_at(dataset: Any) -> Callable[[int], str]:
    """A cheap index → image-id lookup: the folder's stems, else the xrv csv id."""
    ids = getattr(dataset, "example_ids", None)
    if ids is not None:
        return lambda i: str(ids[i])
    return lambda i: _image_id(dataset, i)


def materialize_cell(
    spec: CellSpec,
    pg: PatientGenConfig,
    schema: LabelSchema,
    *,
    indices: Sequence[int],
    root: Path = DEFAULT_CACHE_ROOT,
    batch_size: int = 1,
    device: str | None = None,
    checkpoint_every: int = 0,
    model_builder: ModelBuilder | None = None,
    dataset: Any | None = None,
) -> InferenceCache:
    """Ensure ``spec``'s cell covers ``indices`` for ``pg``'s dataset; build only the gap.

    Raises on a fingerprint conflict (a cell at this stem built from a different
    model/preprocess) rather than silently mixing two experiments in one file. Returns
    the resulting cache (the existing one untouched when nothing was missing).
    """
    ds_id = dataset_id(pg)
    ds_fp = dataset_cell_fingerprint(pg)
    status = inspect_cell(spec, ds_id, ds_fp, root)
    if status.conflict is not None:
        raise ValueError(
            f"cache cell {status.stem} conflicts with the requested model/dataset: "
            f"{status.conflict}. Rename the alias or build with a fresh stem."
        )
    prior = InferenceCache.load(status.stem) if status.exists else None
    ds = dataset if dataset is not None else _build_dataset(pg, schema)

    # Only the requested indices whose image id the cell does not already hold, keeping
    # the request's order and dropping duplicates.
    id_at = _id_at(ds)
    seen: set[str] = set()
    remaining: list[int] = []
    for i in indices:
        img_id = id_at(int(i))
        if img_id not in status.present_ids and img_id not in seen:
            seen.add(img_id)
            remaining.append(int(i))

    if not remaining:
        if prior is None:
            raise ValueError(
                "no indices to build and no existing cache at "
                f"{status.stem} to return"
            )
        return prior

    build = model_builder or _default_model_builder
    ai_cfg = spec.ai_cfg if device is None else spec.ai_cfg.model_copy(update={"device": device})
    model = build(ai_cfg, schema)

    stem = status.stem
    on_ckpt: Callable[[InferenceCache], None] | None = (
        (lambda c: c.save(stem)) if checkpoint_every else None
    )
    cache = build_inference_cache(
        model,
        ds,
        indices=remaining,
        dataset_name=ds_id,
        batch_size=batch_size,
        checkpoint_every=checkpoint_every,
        on_checkpoint=on_ckpt,
        prior=prior,
        dataset_fingerprint=ds_fp,
        model_fingerprint=spec.fingerprint,
    )
    cache.save(stem)
    return cache


def _build_dataset(pg: PatientGenConfig, schema: LabelSchema) -> Any:
    from hai_sim.factory import build_cacheable_dataset

    return build_cacheable_dataset(pg, schema)
