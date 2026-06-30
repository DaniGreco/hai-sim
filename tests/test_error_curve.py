"""Tests for the cumulative error-capture curve math and rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hai_sim.analysis.error_curve import (
    _auroc,
    compute_curve,
    per_pathology_curves,
    render_error_curves,
    render_per_pathology,
)
from hai_sim.analysis.inference_cache import InferenceCache
from hai_sim.core.types import LabelSchema


def _curve(unc: list[float], err: list[bool]):
    return compute_curve(
        np.array(unc),
        np.array(err),
        error_type="any",
        mode="entry",
        measure="mutual_information",
        threshold=0.5,
    )


def test_perfect_ordering_gives_skill_one() -> None:
    # Errors are exactly the most uncertain units → curve == oracle.
    cr = _curve([0.9, 0.8, 0.1, 0.05], [True, True, False, False])
    np.testing.assert_array_equal(cr.actual, cr.oracle)
    assert cr.skill == pytest.approx(1.0)
    assert cr.auroc == pytest.approx(1.0)
    assert cr.n_errors == 2
    assert cr.actual[-1] == 2  # plateau at total errors


def test_anti_correlated_ordering_is_worse_than_random() -> None:
    # Errors are the LEAST uncertain units → below the random diagonal.
    cr = _curve([0.1, 0.05, 0.9, 0.8], [True, True, False, False])
    assert cr.skill == pytest.approx(-1.0)
    assert cr.auroc == pytest.approx(0.0)


def test_actual_is_monotone_and_ends_at_total() -> None:
    rng = np.random.default_rng(0)
    unc = rng.random(50)
    err = rng.random(50) < 0.3
    cr = compute_curve(
        unc, err, error_type="any", mode="entry", measure="std", threshold=0.5
    )
    assert np.all(np.diff(cr.actual) >= 0)  # non-decreasing
    assert cr.actual[-1] == cr.n_errors
    assert cr.random[-1] == pytest.approx(cr.n_errors)


def test_auroc_ties_average_to_half() -> None:
    # All scores equal → no discrimination → AUROC 0.5.
    assert _auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 1, 0, 0], dtype=bool)) == pytest.approx(0.5)


def test_auroc_degenerate_is_nan() -> None:
    assert np.isnan(_auroc(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0], dtype=bool)))


def _synthetic_cache(n: int = 40, m: int = 4) -> InferenceCache:
    rng = np.random.default_rng(3)
    mp = rng.random((n, m, 18)).astype(np.float32)
    mp[:, 0, 7] = np.nan
    gt = rng.choice([1, 0, -1], size=(n, 18)).astype(np.int8)
    return InferenceCache(
        member_probs=mp,
        ground_truth=gt,
        inference_seconds=rng.random(n),
        image_ids=tuple(f"i{i}" for i in range(n)),
        dataset_indices=np.arange(n, dtype=np.int64),
        weights=tuple(f"w{j}" for j in range(m)),
        pathologies=tuple(LabelSchema.xrv().labels),
        dataset="nih",
        batch_size=1,
    )


def test_render_produces_png_and_summary(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    cache = _synthetic_cache()
    out = tmp_path / "curves.png"
    summary = render_error_curves(cache, out, measure="mutual_information", threshold=0.5)

    assert out.exists()
    assert out.with_suffix(".json").exists()
    # 2 modes x 3 error types = 6 panels.
    assert set(summary) == {
        "image/any", "image/fn", "image/fp", "entry/any", "entry/fn", "entry/fp",
    }
    for panel in summary.values():
        assert "skill" in panel and "auroc" in panel


def test_per_pathology_skips_unannotated(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    cache = _synthetic_cache()
    cache.ground_truth[:, 14:] = -1  # last 4 columns unannotated (like NIH)

    res = per_pathology_curves(cache, error_type="any")
    names = LabelSchema.xrv().labels
    assert len(res) <= 14
    for name in names[14:]:  # unannotated labels are skipped
        assert name not in res

    out = tmp_path / "pp.png"
    summary = render_per_pathology(cache, out, error_type="fn")
    assert out.exists() and out.with_suffix(".json").exists()
    assert set(summary) == set(res)  # same annotated labels, any error type
