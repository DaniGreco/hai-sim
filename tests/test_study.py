"""Study layer: factor expansion, report extraction, and an end-to-end tidy table."""

from __future__ import annotations

import pytest

from hai_sim.metrics.kpi import TeamKPIs
from hai_sim.study.runner import (
    StudyResult,
    _bundles_for_report,
    _get_report_path,
    run_study,
    write_tidy_csv,
)
from hai_sim.study.spec import StudyConfig, _set_path, expand_runs

_BASE = {
    "name": "base",
    "seed": 0,
    "label_schema": "default_v0",
    "patient_generator": {
        "kind": "mock_dataset",
        "arrival_rate": 0.5,
        "n_patients": 12,
        "n_examples": 20,
        "prevalence": 0.3,
    },
    "ai": {"kind": "random_mock"},
    "persona": "solo_senior",
    "sort": {"kind": "uncertainty_desc", "aggregator": "mean"},
}


def _study(**over) -> StudyConfig:
    cfg = {
        "name": "t",
        "base": _BASE,
        "factors": {"persona": ["solo_senior", "junior_trusting"]},
        "report": ["team.rair", "actors.hybrid.n_fn"],
    }
    cfg.update(over)
    return StudyConfig.model_validate(cfg)


# --- config / validation -----------------------------------------------------


def test_empty_factor_levels_rejected():
    with pytest.raises(ValueError):
        _study(factors={"persona": []})


def test_report_is_optional_and_means_every_scalar():
    """Naming paths up front stopped being required: a study that omits `report` gets
    every scalar the KPI report flattens to, so it need not know the names first."""
    study = StudyConfig.model_validate(
        {"base": _BASE, "factors": {"persona": ["solo_senior"]}}
    )
    assert study.report == ()
    result = run_study(study)
    assert "actors.ai.f1_macro" in result.report_paths
    assert "system.n_patients" in result.report_paths


# --- path setting ------------------------------------------------------------


def test_set_path_nested_and_creates():
    d: dict = {}
    _set_path(d, "a.b.c", 5)
    assert d == {"a": {"b": {"c": 5}}}
    _set_path(d, "a.b.d", 9)
    assert d["a"]["b"] == {"c": 5, "d": 9}


# --- expansion ---------------------------------------------------------------


def test_grid_expansion_count():
    study = _study(
        factors={"persona": ["solo_senior", "junior_trusting"], "sort.aggregator": ["mean", "max"]},
        replications=[1, 2, 3],
        design="grid",
    )
    runs = expand_runs(study)
    assert len(runs) == 2 * 2 * 3
    # the override reached the built scenario
    seeds = {r.seed for r in runs}
    assert seeds == {1, 2, 3}


def test_ofat_expansion_count():
    # baseline + (2-1) persona + (2-1) aggregator = 3 cells, x1 seed
    study = _study(
        factors={"persona": ["solo_senior", "junior_trusting"], "sort.aggregator": ["mean", "max"]},
        design="ofat",
        replications=[7],
    )
    runs = expand_runs(study)
    assert len(runs) == 3


def test_override_applies_to_scenario():
    study = _study(factors={"sort.aggregator": ["mean", "max"]})
    runs = expand_runs(study)
    aggs = {r.scenario.sort.aggregator for r in runs}
    assert aggs == {"mean", "max"}


# --- report extraction -------------------------------------------------------


def test_bundles_for_report_dedup_order():
    assert _bundles_for_report(("team.rair", "team.rsr", "actors.hybrid.n_fn")) == (
        "team",
        "actors",
    )


def test_get_report_path_dict_and_dataclass():
    report = {
        "system": {"throughput": 2.0},
        "team": TeamKPIs(0.1, 0.2, 0.5, None, None, None, 3, 0),
    }
    assert _get_report_path(report, "system.throughput") == 2.0
    assert _get_report_path(report, "team.rair") == 0.5
    assert _get_report_path(report, "team.rsr") is None


# --- end-to-end --------------------------------------------------------------


def test_run_study_produces_tidy_rows(tmp_path):
    study = _study(replications=[1, 2])  # 2 personas x 2 seeds
    result = run_study(study)
    assert isinstance(result, StudyResult)
    assert len(result.rows) == 4
    for row in result.rows:
        assert "persona" in row and "seed" in row
        assert "team.rair" in row and "actors.hybrid.n_fn" in row
    # tidy CSV has the factor + seed + report columns
    csv_path = tmp_path / "study.csv"
    write_tidy_csv(result, csv_path)
    header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == ["persona", "seed", "team.rair", "actors.hybrid.n_fn"]
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 1 + 4


def test_run_study_in_a_process_pool_matches_the_sequential_rows():
    study = _study(replications=[1, 2])
    sequential = run_study(study)
    parallel = run_study(study, workers=2)
    assert parallel.rows == sequential.rows
    assert parallel.n_executed == sequential.n_executed
