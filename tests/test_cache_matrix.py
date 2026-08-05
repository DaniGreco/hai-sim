"""Addressing and identity of the cache matrix: expanding an AI config into atomic
cells, naming their stems, fingerprinting what must match, and resolving a cell against
disk (missing / matching-with-coverage / conflicting)."""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.analysis.cache_matrix import (
    apply_cache_mode,
    cells_for_ai,
    dataset_cell_fingerprint,
    dataset_id,
    inspect_cell,
    plan_summary,
    resolve_matrix,
)
from hai_sim.analysis.inference_cache import InferenceCache
from hai_sim.config.scenario import AIModelConfig, PatientGenConfig

ENSEMBLE_4 = (
    "densenet121-res224-chex",
    "densenet121-res224-mimic_ch",
    "densenet121-res224-pc",
    "densenet121-res224-rsna",
)


def _xrv(weights) -> AIModelConfig:
    return AIModelConfig(kind="torchxrv", weights=weights)


def _pg(**kw) -> PatientGenConfig:
    base = dict(arrival_rate=1.0, n_patients=1)
    base.update(kw)
    return PatientGenConfig(**base)


# --- expansion into cells ----------------------------------------------------


def test_single_weight_is_one_cell() -> None:
    (cell,) = cells_for_ai(_xrv("densenet121-res224-chex"))
    assert cell.alias == "chex"
    assert cell.fingerprint == "torchxrv|densenet121-res224-chex"
    assert cell.ai_cfg.weights == "densenet121-res224-chex"


def test_ensemble_expands_per_member() -> None:
    cells = cells_for_ai(_xrv(ENSEMBLE_4))
    assert [c.alias for c in cells] == ["chex", "mimic_ch", "pc", "rsna"]
    # each member cell builds exactly one weight set
    assert [c.ai_cfg.weights for c in cells] == list(ENSEMBLE_4)
    # fingerprints are per-member and distinct
    assert len({c.fingerprint for c in cells}) == 4


def test_medgemma_is_one_cell_with_config_in_fingerprint() -> None:
    cfg = AIModelConfig(
        kind="medgemma",
        model_id="google/medgemma-1.5-4b-it",
        prompt_set="triple",
        readout="independent_question",
    )
    (cell,) = cells_for_ai(cfg)
    assert cell.alias == "medgemma-1.5-4b"
    # the things that change the raw probabilities are all in the identity
    assert "google/medgemma-1.5-4b-it" in cell.fingerprint
    assert "prompt_set=triple" in cell.fingerprint
    assert "readout=independent_question" in cell.fingerprint


def test_27b_alias() -> None:
    (cell,) = cells_for_ai(AIModelConfig(kind="medgemma", model_id="google/medgemma-27b-it"))
    assert cell.alias == "medgemma-27b"


@pytest.mark.parametrize("kind", ["random_mock", "fixed_accuracy", "cached"])
def test_non_cacheable_kinds_raise(kind) -> None:
    with pytest.raises(ValueError):
        cells_for_ai(AIModelConfig(kind=kind))


def test_duplicate_alias_raises() -> None:
    # Two weight sets that shorten to the same alias ("pc") would collide on disk.
    with pytest.raises(ValueError):
        cells_for_ai(_xrv(("densenet121-res224-pc", "densenet121-res500-pc")))


# --- stems -------------------------------------------------------------------


def test_stem_naming(tmp_path) -> None:
    (cell,) = cells_for_ai(_xrv("densenet121-res224-pc"))
    assert cell.stem("vindr", tmp_path).name == "vindr__pc"


# --- dataset id / fingerprint ------------------------------------------------


def test_dataset_id_sources() -> None:
    assert dataset_id(_pg(kind="xrv_dataset", dataset="nih")) == "nih"
    assert dataset_id(_pg(kind="image_folder", imgpath="data/vindr/images")) == "vindr"
    # explicit override wins
    assert dataset_id(_pg(kind="image_folder", imgpath="x/y/z", dataset_id="vindr")) == "vindr"


def test_dataset_fingerprint_has_no_image_count() -> None:
    fp = dataset_cell_fingerprint(_pg(kind="xrv_dataset", dataset="nih", resize=224))
    assert "resize=224" in fp and "ndata" not in fp and "n_images" not in fp


def test_dataset_fingerprint_changes_with_preprocess() -> None:
    a = dataset_cell_fingerprint(_pg(kind="xrv_dataset", dataset="nih", resize=224))
    b = dataset_cell_fingerprint(_pg(kind="xrv_dataset", dataset="nih", resize=896))
    c = dataset_cell_fingerprint(
        _pg(kind="xrv_dataset", dataset="nih", resize=224, views=("PA", "AP"))
    )
    assert a != b and a != c


# --- resolve against disk ----------------------------------------------------


def _save_cell(stem, *, ids, model_fp, dataset_fp) -> None:
    n = len(ids)
    InferenceCache(
        member_probs=np.zeros((n, 1, 3), dtype=np.float32),
        ground_truth=np.zeros((n, 3), dtype=np.int8),
        inference_seconds=np.zeros(n),
        image_ids=tuple(ids),
        dataset_indices=np.arange(n, dtype=np.int64),
        weights=("densenet121-res224-chex",),
        pathologies=("A", "B", "C"),
        dataset="nih",
        batch_size=1,
        dataset_fingerprint=dataset_fp,
        model_fingerprint=model_fp,
    ).save(stem)


def test_inspect_missing_cell(tmp_path) -> None:
    (cell,) = cells_for_ai(_xrv("densenet121-res224-chex"))
    st = inspect_cell(cell, "nih", "xrv:nih|resize=224", root=tmp_path)
    assert not st.exists
    assert st.conflict is None
    assert st.missing(["a", "b"]) == ["a", "b"]


def test_inspect_matching_cell_reports_coverage(tmp_path) -> None:
    (cell,) = cells_for_ai(_xrv("densenet121-res224-chex"))
    dataset_fp = "xrv:nih|resize=224"
    _save_cell(cell.stem("nih", tmp_path), ids=["a", "b"], model_fp=cell.fingerprint, dataset_fp=dataset_fp)
    st = inspect_cell(cell, "nih", dataset_fp, root=tmp_path)
    assert st.exists and st.conflict is None
    assert st.present_ids == frozenset({"a", "b"})
    # only the uncovered ids come back, de-duplicated and in order
    assert st.missing(["a", "c", "c", "b", "d"]) == ["c", "d"]


def test_inspect_model_conflict(tmp_path) -> None:
    (cell,) = cells_for_ai(_xrv("densenet121-res224-chex"))
    _save_cell(
        cell.stem("nih", tmp_path),
        ids=["a"],
        model_fp="torchxrv|densenet121-res224-pc",  # different weight set at this stem
        dataset_fp="xrv:nih|resize=224",
    )
    st = inspect_cell(cell, "nih", "xrv:nih|resize=224", root=tmp_path)
    assert st.exists and st.conflict is not None and "model identity" in st.conflict


def test_inspect_dataset_conflict(tmp_path) -> None:
    (cell,) = cells_for_ai(_xrv("densenet121-res224-chex"))
    _save_cell(
        cell.stem("nih", tmp_path),
        ids=["a"],
        model_fp=cell.fingerprint,
        dataset_fp="xrv:nih|resize=896",  # built at a different resolution
    )
    st = inspect_cell(cell, "nih", "xrv:nih|resize=224", root=tmp_path)
    assert st.conflict is not None and "dataset identity" in st.conflict


# --- resolve a whole study's cells + the cache_mode policy --------------------

DFP = "xrv:nih|resize=224"


def _pair_cells():
    return cells_for_ai(_xrv(("densenet121-res224-chex", "densenet121-res224-pc")))


def test_resolve_reports_incomplete(tmp_path) -> None:
    chex, pc = _pair_cells()
    _save_cell(chex.stem("nih", tmp_path), ids=["a", "b"], model_fp=chex.fingerprint, dataset_fp=DFP)
    # pc has no file on disk -> a miss
    plan = resolve_matrix([chex, pc], "nih", DFP, ["a", "b"], root=tmp_path)
    assert not plan.is_complete
    assert [c.spec.alias for c in plan.incomplete] == ["pc"]
    with pytest.raises(ValueError, match="incomplete"):
        apply_cache_mode(plan, "require")


def test_require_passes_when_complete(tmp_path) -> None:
    cells = _pair_cells()
    for c in cells:
        _save_cell(c.stem("nih", tmp_path), ids=["a", "b"], model_fp=c.fingerprint, dataset_fp=DFP)
    plan = resolve_matrix(cells, "nih", DFP, ["a", "b"], root=tmp_path)
    assert plan.is_complete
    assert apply_cache_mode(plan, "require") is plan


def test_fill_calls_back_for_incomplete_cells_only(tmp_path) -> None:
    chex, pc = _pair_cells()
    _save_cell(chex.stem("nih", tmp_path), ids=["a", "b"], model_fp=chex.fingerprint, dataset_fp=DFP)
    plan = resolve_matrix([chex, pc], "nih", DFP, ["a", "b"], root=tmp_path)
    filled: list[str] = []
    apply_cache_mode(plan, "fill", fill=lambda spec: filled.append(spec.alias))
    assert filled == ["pc"]


def test_report_never_raises_and_summarizes(tmp_path) -> None:
    (cell,) = cells_for_ai(_xrv("densenet121-res224-chex"))
    plan = resolve_matrix([cell], "nih", DFP, ["a", "b"], root=tmp_path)  # nothing on disk
    assert apply_cache_mode(plan, "report") is plan
    assert "miss" in plan_summary(plan)


def test_conflict_raises_in_every_mode(tmp_path) -> None:
    (cell,) = cells_for_ai(_xrv("densenet121-res224-chex"))
    _save_cell(cell.stem("nih", tmp_path), ids=["a"], model_fp="torchxrv|OTHER", dataset_fp=DFP)
    plan = resolve_matrix([cell], "nih", DFP, ["a"], root=tmp_path)
    assert plan.conflicts
    for mode in ("require", "fill", "report"):
        with pytest.raises(ValueError, match="conflict"):
            apply_cache_mode(plan, mode, fill=lambda _s: None)
