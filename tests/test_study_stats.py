"""Replication statistics and paired contrasts.

The engine splits a run's seed into independent per-module substreams so two arms at the
same seed see the same patient stream. That property is what makes a *paired* comparison
valid, and it was going unclaimed: the study layer emitted raw rows and never computed a
per-seed difference.
"""

from __future__ import annotations

import csv

import pytest

from hai_sim.study.runner import (
    run_study,
    write_long_csv,
    write_summary_csv,
    write_tidy_csv,
)
from hai_sim.study.spec import StudyConfig, baseline_of, expand_runs
from hai_sim.study.stats import (
    TidyRow,
    aggregate,
    paired_contrasts,
    summarize_cells,
)

_BASE = {
    "name": "b",
    "seed": 0,
    "label_schema": "default_v0",
    "patient_generator": {
        "kind": "poisson",
        "arrival_rate": 1.0,
        "n_patients": 20,
        "prevalence": 0.3,
    },
    "ai": {"kind": "random_mock"},
    "persona": "solo_senior",
    "sort": {"kind": "random"},
}


# ── Aggregation ────────────────────────────────────────────────────────────────


def test_aggregate_of_nothing_is_none():
    assert aggregate([]) is None


def test_single_replication_reports_no_spread_rather_than_zero():
    """One observation has no spread; reporting 0 would read as 'perfectly
    reproducible' instead of 'unknown'."""
    stat = aggregate([4.0])
    assert stat is not None
    assert (stat.n, stat.mean) == (1, 4.0)
    assert stat.sd is None and stat.ci_low is None and stat.ci_high is None


def test_aggregate_mean_sd_and_interval():
    stat = aggregate([2.0, 4.0, 6.0])
    assert stat is not None
    assert stat.mean == 4.0
    assert stat.sd == pytest.approx(2.0)  # sample sd, n-1 in the denominator
    # t(0.975, df=2) = 4.303 → half-width = 4.303 * 2/sqrt(3)
    assert stat.ci_low == pytest.approx(4.0 - 4.303 * 2.0 / 3**0.5, rel=1e-3)


def test_interval_falls_back_to_the_normal_past_the_table():
    stat = aggregate([1.0, -1.0] * 40)  # n=80, mean 0
    assert stat is not None
    assert stat.ci_high == pytest.approx(1.96 * stat.sd / 80**0.5, rel=1e-6)


def test_excludes_zero_reads_the_interval():
    assert aggregate([1.0, 1.1, 0.9]).excludes_zero
    assert not aggregate([1.0, -1.0, 0.0]).excludes_zero
    assert not aggregate([1.0]).excludes_zero  # unknown, not "no"


# ── Summaries ──────────────────────────────────────────────────────────────────


def _row(persona: str, seed: int, metric: str, value, group: str = "") -> TidyRow:
    return TidyRow(factors={"persona": persona}, seed=seed, group=group, metric=metric, value=value)


def test_summaries_group_by_cell_group_and_metric():
    rows = [
        _row("a", 1, "m", 1.0), _row("a", 2, "m", 3.0),
        _row("b", 1, "m", 10.0), _row("b", 2, "m", 12.0),
    ]
    summaries = {s.factors["persona"]: s.stat for s in summarize_cells(rows)}
    assert summaries["a"].mean == 2.0
    assert summaries["b"].mean == 11.0


def test_non_numeric_metrics_are_skipped_not_coerced():
    rows = [_row("a", 1, "name", "senior"), _row("a", 1, "m", 1.0)]
    assert [s.metric for s in summarize_cells(rows)] == ["m"]


# ── Paired contrasts ───────────────────────────────────────────────────────────


def _baseline_for(factors):
    return None if factors["persona"] == "base" else {"persona": "base"}


def test_contrast_pairs_on_the_seed():
    rows = [
        _row("base", 1, "m", 1.0), _row("base", 2, "m", 5.0),
        _row("arm", 1, "m", 2.0), _row("arm", 2, "m", 7.0),
    ]
    (contrast,) = paired_contrasts(rows, _baseline_for)
    assert contrast.n_pairs == 2
    assert contrast.delta.mean == pytest.approx(1.5)  # (2-1) and (7-5)
    assert contrast.arm_mean == pytest.approx(4.5)
    assert contrast.baseline_mean == pytest.approx(3.0)


def test_pairing_beats_comparing_two_independent_means():
    """The whole reason for common random numbers: a shared per-seed swing cancels in
    the difference, so the paired interval is tight where an unpaired one is not."""
    shared = [0.0, 10.0, 20.0, 30.0]  # a large seed-to-seed swing, common to both arms
    rows = []
    for seed, offset in enumerate(shared, start=1):
        rows.append(_row("base", seed, "m", offset))
        rows.append(_row("arm", seed, "m", offset + 2.0))  # constant +2 effect
    (contrast,) = paired_contrasts(rows, _baseline_for)
    assert contrast.delta.mean == pytest.approx(2.0)
    assert contrast.delta.sd == pytest.approx(0.0)  # the swing cancelled entirely
    unpaired = aggregate([r.value for r in rows if r.factors["persona"] == "arm"])
    assert unpaired is not None and unpaired.sd > 10.0  # ... and would have swamped it


def test_seeds_present_in_only_one_arm_are_skipped():
    """An unpaired difference would put back the noise the pairing exists to remove."""
    rows = [
        _row("base", 1, "m", 1.0),
        _row("arm", 1, "m", 2.0), _row("arm", 9, "m", 99.0),  # seed 9 has no baseline
    ]
    (contrast,) = paired_contrasts(rows, _baseline_for)
    assert contrast.n_pairs == 1
    assert contrast.delta.mean == pytest.approx(1.0)


def test_contrasts_are_per_subgroup():
    rows = [
        _row("base", 1, "m", 1.0, group="g=x"), _row("arm", 1, "m", 2.0, group="g=x"),
        _row("base", 1, "m", 1.0, group="g=y"), _row("arm", 1, "m", 5.0, group="g=y"),
    ]
    by_group = {c.group: c.delta.mean for c in paired_contrasts(rows, _baseline_for)}
    assert by_group == {"g=x": 1.0, "g=y": 4.0}


# ── Which cell is the baseline ─────────────────────────────────────────────────


def _study(**kwargs) -> StudyConfig:
    payload = {"base": _BASE, "factors": {"persona": ["solo_senior", "junior_trusting"]}}
    payload.update(kwargs)
    return StudyConfig.model_validate(payload)


def test_baseline_of_is_none_for_the_baseline_cell_itself():
    study = _study(baseline={"persona": "solo_senior"})
    assert baseline_of({"persona": "solo_senior"}, study) is None
    assert baseline_of({"persona": "junior_trusting"}, study) == {"persona": "solo_senior"}


def test_partial_baseline_holds_the_other_factors_fixed():
    """With a baseline on one factor only, each arm is contrasted against the cell that
    differs in *that* factor alone — otherwise the contrast confounds two changes."""
    study = StudyConfig.model_validate(
        {
            "base": _BASE,
            "factors": {"persona": ["solo_senior", "junior_trusting"], "seed_x": [1, 2]},
            "baseline": {"persona": "solo_senior"},
        }
    )
    assert baseline_of({"persona": "junior_trusting", "seed_x": 2}, study) == {
        "persona": "solo_senior",
        "seed_x": 2,
    }


def test_baseline_must_name_real_factors_and_real_levels():
    with pytest.raises(ValueError, match="non-factors"):
        _study(baseline={"nope": 1})
    with pytest.raises(ValueError, match="not among that factor's levels"):
        _study(baseline={"persona": "nonexistent"})


# ── Designs ────────────────────────────────────────────────────────────────────


def test_random_design_samples_the_grid_and_keeps_the_baseline():
    study = StudyConfig.model_validate(
        {
            "base": _BASE,
            "factors": {"persona": ["solo_senior", "junior_trusting", "senior_skeptical"]},
            "design": "random",
            "n_samples": 2,
            "baseline": {"persona": "solo_senior"},
        }
    )
    personas = [r.factors["persona"] for r in expand_runs(study)]
    assert len(personas) == 2
    # a contrast against an arm that was never run is not a contrast
    assert "solo_senior" in personas


def test_random_design_requires_n_samples():
    with pytest.raises(ValueError, match="requires n_samples"):
        _study(design="random")


def test_identical_scenarios_are_simulated_once_but_still_reported():
    """A repeated level is a plausible typo; it should cost one run, not two, and must
    still produce the row that was asked for."""
    study = StudyConfig.model_validate(
        {
            "base": _BASE,
            "factors": {"ai.decision_threshold": [0.5, 0.5, 0.9]},
            "report": ["system.n_patients"],
        }
    )
    result = run_study(study)
    assert len(result.rows) == 3
    assert result.n_executed == 2


# ── Output files ───────────────────────────────────────────────────────────────


def _header(path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def test_unconditioned_study_writes_the_format_it_always_wrote(tmp_path):
    study = _study(report=["system.n_patients"], replications=[1, 2])
    result = run_study(study)
    path = tmp_path / "s.csv"
    write_tidy_csv(result, path)
    assert _header(path) == ["persona", "seed", "system.n_patients"]


def test_long_and_summary_files_carry_the_statistics(tmp_path):
    study = _study(
        report=["system.n_patients"],
        replications=[1, 2, 3],
        baseline={"persona": "solo_senior"},
    )
    result = run_study(study)

    long_path = tmp_path / "s_long.csv"
    write_long_csv(result, long_path)
    assert _header(long_path) == ["persona", "seed", "group", "metric", "value"]

    summary_path = tmp_path / "s_summary.csv"
    write_summary_csv(result, summary_path)
    header = _header(summary_path)
    for column in ("n", "mean", "sd", "ci_low", "ci_high", "vs_baseline_delta"):
        assert column in header

    contrasts = result.contrasts()
    assert contrasts, "a declared baseline must produce contrasts"
    assert all(c.n_pairs == 3 for c in contrasts)
