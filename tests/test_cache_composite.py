"""The composite replay actor and the transparent cache wiring: composing several
single-model caches by image id, and `build_cached_ai` resolving a study's cells against
disk under `cache_mode` (no live inference — caches are pre-built on disk)."""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.analysis.cache_matrix import cells_for_ai, dataset_cell_fingerprint
from hai_sim.analysis.inference_cache import InferenceCache
from hai_sim.config.scenario import AIModelConfig, PatientGenConfig
from hai_sim.core.types import LabelSchema, PatientContext
from hai_sim.factory import build_cached_ai
from hai_sim.models.cached import CompositeCachedAI

SCHEMA = LabelSchema(labels=("A", "B", "C"))
IMG = [np.zeros((1, 4, 4), dtype=np.float32)]


def _ctx(example_id: str) -> PatientContext:
    return PatientContext(patient_id=0, arrival_time=0.0, example_id=example_id)


def _cache(ids, *, fill_value, model_fp, dataset_fp="ds", pathologies=("A", "B", "C")) -> InferenceCache:
    n = len(ids)
    return InferenceCache(
        member_probs=np.full((n, 1, len(pathologies)), fill_value, dtype=np.float32),
        ground_truth=np.zeros((n, len(pathologies)), dtype=np.int8),
        inference_seconds=np.zeros(n),
        image_ids=tuple(ids),
        dataset_indices=np.arange(n, dtype=np.int64),
        weights=("w",),
        pathologies=tuple(pathologies),
        dataset="tds",
        batch_size=1,
        dataset_fingerprint=dataset_fp,
        model_fingerprint=model_fp,
    )


class FakeExampleDataset:
    def __init__(self, ids) -> None:
        self._ids = list(ids)

    @property
    def example_ids(self) -> list[str]:
        return list(self._ids)

    def __len__(self) -> int:
        return len(self._ids)


# --- CompositeCachedAI -------------------------------------------------------


def test_composite_serves_intersection_by_id() -> None:
    c1 = _cache(["a", "b"], fill_value=0.2, model_fp="torchxrv|w1")
    c2 = _cache(["a", "c"], fill_value=0.8, model_fp="torchxrv|w2")
    ai = CompositeCachedAI([c1, c2], SCHEMA)
    # only 'a' is in both member caches
    assert ai.covered_ids == frozenset({"a"})
    out = ai.diagnose(IMG, context=_ctx("a"))
    assert len(out.diagnosis.confidence) == 3
    # 'b' is in c1 but not c2 -> not composable
    with pytest.raises(ValueError, match="not covered"):
        ai.diagnose(IMG, context=_ctx("b"))


def test_composite_stacks_members() -> None:
    c1 = _cache(["a"], fill_value=0.2, model_fp="torchxrv|w1")
    c2 = _cache(["a"], fill_value=0.8, model_fp="torchxrv|w2")
    ai = CompositeCachedAI([c1, c2], SCHEMA)
    # two single-member caches compose into a two-member stack
    assert ai._probs.shape == (1, 2, 3)


def test_composite_rejects_schema_mismatch() -> None:
    wrong = _cache(["a"], fill_value=0.2, model_fp="m", pathologies=("A", "B"))
    with pytest.raises(ValueError, match="pathologies"):
        CompositeCachedAI([wrong], SCHEMA)


def test_composite_needs_example_id() -> None:
    ai = CompositeCachedAI([_cache(["a"], fill_value=0.2, model_fp="m")], SCHEMA)
    with pytest.raises(ValueError, match="example_id"):
        ai.diagnose(IMG, context=PatientContext(patient_id=0, arrival_time=0.0))


# --- build_cached_ai wiring (pre-built caches, no torch) ---------------------


def _study_cfg(tmp_path, cache_mode: str) -> AIModelConfig:
    return AIModelConfig(
        kind="torchxrv",
        weights=("densenet121-res224-chex", "densenet121-res224-pc"),
        cache_mode=cache_mode,
        cache_root=str(tmp_path),
    )


def _study_pg() -> PatientGenConfig:
    return PatientGenConfig(
        kind="image_folder", arrival_rate=1.0, n_patients=1, dataset_id="tds", resize=8
    )


def _prebuild(tmp_path, cfg, pg, ids) -> None:
    ds_fp = dataset_cell_fingerprint(pg)
    for cell in cells_for_ai(cfg):
        _cache(ids, fill_value=0.3, model_fp=cell.fingerprint, dataset_fp=ds_fp).save(
            cell.stem("tds", tmp_path)
        )


def test_require_composes_when_all_cells_present(tmp_path) -> None:
    cfg, pg = _study_cfg(tmp_path, "require"), _study_pg()
    _prebuild(tmp_path, cfg, pg, ["a", "b"])
    ai = build_cached_ai(cfg, pg, SCHEMA, dataset=FakeExampleDataset(["a", "b"]), indices=[0, 1])
    assert isinstance(ai, CompositeCachedAI)
    assert ai.covered_ids == frozenset({"a", "b"})
    assert len(ai.diagnose(IMG, context=_ctx("b")).diagnosis.confidence) == 3


def test_require_errors_on_a_missing_cell(tmp_path) -> None:
    cfg, pg = _study_cfg(tmp_path, "require"), _study_pg()
    # only the chex cell is present; pc is missing -> require must refuse
    ds_fp = dataset_cell_fingerprint(pg)
    chex = cells_for_ai(cfg)[0]
    _cache(["a", "b"], fill_value=0.3, model_fp=chex.fingerprint, dataset_fp=ds_fp).save(
        chex.stem("tds", tmp_path)
    )
    with pytest.raises(ValueError, match="incomplete"):
        build_cached_ai(cfg, pg, SCHEMA, dataset=FakeExampleDataset(["a", "b"]), indices=[0, 1])


def test_report_prints_plan_and_stops(tmp_path) -> None:
    cfg, pg = _study_cfg(tmp_path, "report"), _study_pg()
    with pytest.raises(ValueError, match="report"):
        build_cached_ai(cfg, pg, SCHEMA, dataset=FakeExampleDataset(["a", "b"]), indices=[0, 1])
