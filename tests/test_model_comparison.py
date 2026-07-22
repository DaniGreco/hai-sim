"""Comparing AI readers of different natures on the same images.

The load-bearing one here is the error overlap: a pair only beats its members where
their mistakes differ, so κ is the quantity the whole complementarity argument rests on.
"""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.analysis.model_comparison import (
    ModelRun,
    align_runs,
    all_overlaps,
    auroc,
    error_capture,
    error_overlap,
    parse_cache_spec,
    per_pathology_auroc,
    reliability,
    render_all,
)


def _run(name, probs, gt, uncertainty=None, ids=None, labels=("L0",)) -> ModelRun:
    probs = np.asarray(probs, dtype=float)
    gt = np.asarray(gt, dtype=int)
    return ModelRun(
        name=name,
        image_ids=tuple(ids or [f"img{i}" for i in range(probs.shape[0])]),
        pathologies=tuple(labels),
        probs=probs,
        uncertainty=np.asarray(
            uncertainty if uncertainty is not None else np.zeros_like(probs), dtype=float
        ),
        ground_truth=gt,
    )


# ── Discrimination ─────────────────────────────────────────────────────────────


def test_auroc_of_a_perfect_ranking_is_one():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1], dtype=bool)) == 1.0


def test_auroc_of_a_reversed_ranking_is_zero():
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1], dtype=bool)) == 0.0


def test_ties_average_to_chance_rather_than_following_the_sort_order():
    """A model emitting one constant score has ranked nothing; it must score 0.5, not
    whatever order the sort happened to produce."""
    flat = np.array([0.5, 0.5, 0.5, 0.5])
    assert auroc(flat, np.array([1, 0, 1, 0], dtype=bool)) == pytest.approx(0.5)


def test_a_single_class_is_not_measurable_and_says_so():
    """None, not 0.5: 'we could not measure this' and 'no better than chance' are
    different claims and a chart must not merge them."""
    assert auroc(np.array([0.1, 0.9]), np.array([1, 1], dtype=bool)) is None


def test_unannotated_entries_are_excluded_from_the_auroc():
    run = _run("m", [[0.9], [0.1], [0.5]], [[1], [0], [-1]])
    assert per_pathology_auroc(run)["L0"] == 1.0  # the -1 row would have broken it


# ── Calibration ────────────────────────────────────────────────────────────────


def test_a_perfectly_calibrated_model_has_no_expected_calibration_error():
    # half the entries stated 0.0 and are negative, half stated 1.0 and are positive
    probs = np.array([[0.0]] * 50 + [[1.0]] * 50)
    gt = np.array([[0]] * 50 + [[1]] * 50)
    assert reliability(_run("m", probs, gt)).expected_calibration_error == pytest.approx(0.0)


def test_an_overconfident_model_shows_a_gap():
    probs = np.full((100, 1), 0.9)
    gt = np.array([[1]] * 20 + [[0]] * 80)  # says 0.9, is right 20% of the time
    rel = reliability(_run("m", probs, gt))
    assert rel.expected_calibration_error == pytest.approx(0.7, abs=1e-9)


def test_a_stated_probability_of_exactly_one_lands_in_the_last_bin():
    rel = reliability(_run("m", np.ones((4, 1)), np.ones((4, 1), dtype=int)), bins=10)
    assert rel.counts[-1] == 4
    assert int(rel.counts.sum()) == 4  # nothing silently dropped


def test_empty_bins_are_nan_not_zero():
    rel = reliability(_run("m", np.full((5, 1), 0.95), np.ones((5, 1), dtype=int)), bins=10)
    assert np.isnan(rel.observed[0])


# ── Error overlap — the complementarity substrate ──────────────────────────────


def _overlap_fixture():
    """100 entries, all truly negative: A errs on 0-29, B errs on 0-9 and 30-59.
    So both=10, only_a=20, only_b=30, neither=40."""
    gt = np.zeros((100, 1), dtype=int)
    a = np.where(np.arange(100)[:, None] < 30, 0.9, 0.1)
    b_hits = (np.arange(100) < 10) | ((np.arange(100) >= 30) & (np.arange(100) < 60))
    b = np.where(b_hits[:, None], 0.9, 0.1)
    return _run("A", a, gt), _run("B", b, gt)


def test_overlap_counts_the_four_outcomes():
    ov = error_overlap(*_overlap_fixture())
    assert (ov.both, ov.only_a, ov.only_b, ov.neither) == (10, 20, 30, 40)
    assert ov.total == 100


def test_kappa_matches_the_hand_computation():
    ov = error_overlap(*_overlap_fixture())
    # observed agreement 0.50; chance 0.3*0.4 + 0.7*0.6 = 0.54 → (0.50-0.54)/0.46
    assert ov.kappa == pytest.approx(-0.04 / 0.46, abs=1e-9)


def test_identical_models_fail_together_completely():
    a, _ = _overlap_fixture()
    ov = error_overlap(a, _run("copy", a.probs, a.ground_truth))
    assert ov.only_a == ov.only_b == 0
    assert ov.kappa == pytest.approx(1.0)


def test_kappa_is_not_measurable_when_one_model_never_errs():
    """Agreement is then guaranteed and κ carries no information — report that rather
    than a number produced by dividing by zero."""
    gt = np.zeros((10, 1), dtype=int)
    flawless = _run("perfect", np.zeros((10, 1)), gt)
    assert error_overlap(flawless, flawless).kappa is None


def test_entries_neither_model_could_be_scored_on_are_excluded():
    gt = np.array([[1], [-1]])
    a = _run("A", [[0.9], [0.9]], gt)
    b = _run("B", [[0.1], [0.1]], gt)
    assert error_overlap(a, b).total == 1  # the -1 row is unmeasurable, not an outcome


def test_every_pair_is_compared_once():
    a, b = _overlap_fixture()
    c = _run("C", a.probs, a.ground_truth)
    pairs = {(o.a, o.b) for o in all_overlaps([a, b, c])}
    assert pairs == {("A", "B"), ("A", "C"), ("B", "C")}


# ── Does uncertainty find the errors? ──────────────────────────────────────────


def test_uncertainty_that_ranks_errors_first_captures_them_immediately():
    gt = np.zeros((10, 1), dtype=int)
    probs = np.where(np.arange(10)[:, None] < 2, 0.9, 0.1)  # the first two are errors
    doubt = np.where(np.arange(10)[:, None] < 2, 1.0, 0.0)  # and are the most uncertain
    reviewed, caught = error_capture(_run("m", probs, gt, doubt))
    assert caught[1] == pytest.approx(1.0)  # both found after two reviews
    assert reviewed[1] == pytest.approx(0.2)


def test_a_model_with_no_errors_yields_a_degenerate_curve_not_a_crash():
    gt = np.zeros((5, 1), dtype=int)
    reviewed, caught = error_capture(_run("m", np.zeros((5, 1)), gt))
    assert reviewed.shape == caught.shape


# ── Aligning models onto the same images ───────────────────────────────────────


def test_alignment_keeps_only_the_shared_images_in_the_first_run_s_order():
    a = _run("A", [[0.1], [0.2], [0.3]], [[0], [0], [0]], ids=["x", "y", "z"])
    b = _run("B", [[0.4], [0.5]], [[0], [0]], ids=["z", "x"])
    aligned = align_runs([a, b])
    assert all(r.image_ids == ("x", "z") for r in aligned)
    assert aligned[1].probs[:, 0].tolist() == [0.5, 0.4]  # reindexed, not merely trimmed


def test_alignment_keeps_only_the_shared_pathologies():
    a = _run("A", np.zeros((2, 2)), np.zeros((2, 2), dtype=int), labels=("P", "Q"))
    b = _run("B", np.zeros((2, 2)), np.zeros((2, 2), dtype=int), labels=("Q", "R"))
    assert align_runs([a, b])[0].pathologies == ("Q",)


def test_models_over_different_datasets_fail_loudly():
    a = _run("A", [[0.1]], [[0]], ids=["x"])
    b = _run("B", [[0.1]], [[0]], ids=["y"])
    with pytest.raises(ValueError, match="share no images"):
        align_runs([a, b])


# ── Cache specs ────────────────────────────────────────────────────────────────


def test_cache_spec_selects_members():
    assert parse_cache_spec("data/c") == ("data/c", None)
    assert parse_cache_spec("data/c#0") == ("data/c", [0])
    assert parse_cache_spec("data/c#0,2") == ("data/c", [0, 2])


def test_a_malformed_member_selection_is_rejected():
    with pytest.raises(ValueError, match="bad member selection"):
        parse_cache_spec("data/c#")
    with pytest.raises(ValueError, match="bad member selection"):
        parse_cache_spec("data/c#a")


# ── Rendering ──────────────────────────────────────────────────────────────────


def test_every_figure_is_written(tmp_path):
    a, b = _overlap_fixture()
    written = render_all([a, b], tmp_path)
    assert {p.name for p in written} == {
        "discrimination.png", "calibration.png", "error_capture.png", "error_overlap.png"
    }
    assert all(p.stat().st_size > 0 for p in written)


def test_a_single_model_skips_the_pairwise_figure(tmp_path):
    a, _ = _overlap_fixture()
    written = render_all([a], tmp_path)
    assert "error_overlap.png" not in {p.name for p in written}
