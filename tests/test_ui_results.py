"""The result browser's reading layer. No Streamlit involved — that is the point.

Every reshaping decision lives in `hai_sim.ui.results`, so it is testable in the normal
suite whether or not the ``[ui]`` extra is installed, and so a future front end (or a
notebook, or a script) inherits the same behaviour rather than reimplementing it.
"""

from __future__ import annotations

import json

import pytest

from hai_sim.ui.results import (
    common_metrics,
    config_diff,
    contrast_rows,
    discover,
    factor_names,
    flatten_config,
    load_records,
    metric_names,
    numeric_column,
    observation_keys,
    run_group_rows,
    run_kpi_rows,
    study_rows,
    summary_rows,
)

STUDY = {
    "study": {
        "name": "s",
        "base": {"seed": 0, "ai": {"kind": "cached", "uq": "ensemble"}},
        "factors": {"persona": ["a", "b"]},
    },
    "factor_names": ["persona"],
    "n_runs": 4,
    "rows": [
        {"persona": "a", "seed": 1, "group": "", "recall": 0.70, "n_fn": 30},
        {"persona": "a", "seed": 2, "group": "", "recall": 0.72, "n_fn": 28},
        {"persona": "b", "seed": 1, "group": "", "recall": 0.60, "n_fn": 40},
        {"persona": "b", "seed": 2, "group": "", "recall": 0.62, "n_fn": 38},
    ],
    "summaries": [
        {
            "factors": {"persona": "a"},
            "group": "",
            "metric": "recall",
            "n": 2,
            "mean": 0.71,
            "sd": 0.014,
            "ci_low": 0.58,
            "ci_high": 0.84,
        }
    ],
    "contrasts": [
        {
            "factors": {"persona": "b"},
            "baseline": {"persona": "a"},
            "group": "",
            "metric": "recall",
            "arm_mean": 0.61,
            "baseline_mean": 0.71,
            "delta_mean": -0.10,
            "delta_ci_low": -0.12,
            "delta_ci_high": -0.08,
            "n_pairs": 2,
            "excludes_zero": True,
        }
    ],
}

RUN = {
    "scenario": {"seed": 3, "ai": {"kind": "cached", "uq": "entropy"}, "persona": "solo_senior"},
    "ai": "CachedEnsembleAI",
    "doctor": "Persona[x]",
    "metadata": {"difficulty_model": "prior:beta(2,5)", "fatigue_curve": "none"},
    "kpis": {
        "system": {"n_patients": 100, "throughput": 0.5},
        "actors": {"hybrid": {"recall_macro": 0.7, "per_label": [0.1, 0.2]}},
    },
    "kpis_by_group": {"sig.difficulty=<0.5": {"system": {"n_patients": 60}}},
}


@pytest.fixture()
def out_dir(tmp_path):
    (tmp_path / "study_s_20260722_101500.json").write_text(json.dumps(STUDY), encoding="utf-8")
    (tmp_path / "summary_r_20260722_120000.json").write_text(json.dumps(RUN), encoding="utf-8")
    (tmp_path / "records_r_20260722_120000.csv").write_text(
        "patient_id,sig.difficulty,sig.fatigue,load.triaged_at_read,doc.note\n"
        "0,0.10,0.00,3,hello\n"
        "1,0.90,0.50,7,hello\n"
        "2,,0.75,,hello\n",
        encoding="utf-8",
    )
    return tmp_path


def test_discover_classifies_by_content_not_by_filename(out_dir, tmp_path):
    """A renamed file must still classify: the JSON is self-describing and the name is
    a convention, so keying on the name would hide exactly the ad-hoc runs one most
    wants to look at."""
    (tmp_path / "whatever.json").write_text(json.dumps(RUN), encoding="utf-8")
    kinds = sorted(a.kind for a in discover(out_dir))
    assert kinds == ["run", "run", "study"]


def test_discover_skips_unreadable_and_unrelated_json(out_dir, tmp_path):
    """An output directory accumulates junk; refusing to start over one stray file
    would make the browser useless precisely when there is most to look at."""
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "unrelated.json").write_text('{"hello": 1}', encoding="utf-8")
    (tmp_path / "a_list.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert len(discover(out_dir)) == 2


def test_discover_on_a_missing_directory_is_empty_not_an_error(tmp_path):
    assert discover(tmp_path / "nope") == []


def test_labels_split_name_from_timestamp(out_dir):
    study = next(a for a in discover(out_dir) if a.kind == "study")
    assert study.name == "s"
    assert study.stamp == "20260722-101500"
    assert "s" in study.label and "20260722" in study.label


def test_a_hand_named_file_keeps_its_whole_stem_as_the_name(out_dir, tmp_path):
    (tmp_path / "my_experiment.json").write_text(json.dumps(RUN), encoding="utf-8")
    names = {a.name for a in discover(out_dir)}
    assert "my_experiment" in names


def test_newest_first(out_dir, tmp_path):
    (tmp_path / "summary_r_20250101_000000.json").write_text(json.dumps(RUN), encoding="utf-8")
    stamps = [a.stamp for a in discover(out_dir)]
    assert stamps == sorted(stamps, reverse=True)


def test_study_exposes_its_factors_and_its_metrics(out_dir):
    study = next(a for a in discover(out_dir) if a.kind == "study")
    assert factor_names(study) == ["persona"]
    assert metric_names(study) == ["recall", "n_fn"]
    assert len(study_rows(study)) == 4


def test_metric_names_never_include_the_identifying_columns(out_dir):
    """`seed` and `group` are numeric-ish and would otherwise be offered as metrics."""
    study = next(a for a in discover(out_dir) if a.kind == "study")
    assert "seed" not in metric_names(study)
    assert "persona" not in metric_names(study)


def test_metric_names_come_from_the_rows_not_from_the_declared_report(out_dir, tmp_path):
    """`report:` is optional — omitting it means 'every scalar', and only the rows know
    what that turned out to be."""
    payload = json.loads(json.dumps(STUDY))
    payload["study"]["report"] = []
    payload["rows"][0]["surprise"] = 1.0
    (tmp_path / "study_t_20260722_101600.json").write_text(json.dumps(payload), encoding="utf-8")
    study = next(a for a in discover(out_dir) if a.name == "t")
    assert "surprise" in metric_names(study)


def test_summary_and_contrast_rows_flatten_the_factor_mapping(out_dir):
    study = next(a for a in discover(out_dir) if a.kind == "study")
    assert summary_rows(study)[0]["persona"] == "a"
    contrast = contrast_rows(study)[0]
    assert contrast["arm"] == "persona=b"
    assert contrast["baseline"] == "persona=a"
    assert contrast["delta_mean"] == -0.10
    assert contrast["excludes_zero"] is True


def test_run_kpis_flatten_to_dotted_paths(out_dir):
    run = next(a for a in discover(out_dir) if a.kind == "run")
    pairs = dict(run_kpi_rows(run))
    assert pairs["system.throughput"] == 0.5
    assert pairs["actors.hybrid.recall_macro"] == 0.7


def test_per_label_vectors_are_skipped_rather_than_averaged(out_dir):
    """A macro average of a vector already has a name and a definition in the bundle
    that produced it; inventing a second one here would put two numbers under one
    label."""
    run = next(a for a in discover(out_dir) if a.kind == "run")
    assert not any(p.startswith("actors.hybrid.per_label") for p, _ in run_kpi_rows(run))


def test_run_group_rows_keep_one_entry_per_subgroup(out_dir):
    run = next(a for a in discover(out_dir) if a.kind == "run")
    groups = run_group_rows(run)
    assert dict(groups["sig.difficulty=<0.5"])["system.n_patients"] == 60.0


def test_records_are_found_next_to_the_summary(out_dir):
    run = next(a for a in discover(out_dir) if a.kind == "run")
    assert len(load_records(run)) == 3


def test_a_missing_records_file_is_not_an_error(out_dir, tmp_path):
    """The summary is useful on its own and must stay readable if its sibling moved."""
    (tmp_path / "records_r_20260722_120000.csv").unlink()
    run = next(a for a in discover(out_dir) if a.kind == "run")
    assert load_records(run) == []


def test_numeric_column_drops_blanks_without_shifting_the_rest(out_dir):
    run = next(a for a in discover(out_dir) if a.kind == "run")
    records = load_records(run)
    assert numeric_column(records, "sig.difficulty") == [0.10, 0.90]
    assert numeric_column(records, "sig.fatigue") == [0.0, 0.5, 0.75]


def test_numeric_column_on_an_absent_key_is_empty(out_dir):
    """Runs legitimately carry different columns — a run with no fatigue curve has no
    `sig.fatigue` — so a missing key is empty, not an error."""
    run = next(a for a in discover(out_dir) if a.kind == "run")
    assert numeric_column(load_records(run), "sig.nonexistent") == []


def test_observation_keys_filter_by_namespace_and_skip_text_columns(out_dir):
    run = next(a for a in discover(out_dir) if a.kind == "run")
    records = load_records(run)
    assert observation_keys(records, "sig.") == ["sig.difficulty", "sig.fatigue"]
    assert "doc.note" not in observation_keys(records)


def test_flatten_config_keeps_a_list_whole():
    """An ensemble's weight list or a rate table is one decision. Expanding it to
    path.0, path.1 would bury the single key that actually differs under fourteen."""
    flat = flatten_config({"ai": {"weights": ["a", "b", "c"]}})
    assert flat == {"ai.weights": ["a", "b", "c"]}


def test_config_diff_reports_only_what_differs(out_dir):
    study = next(a for a in discover(out_dir) if a.kind == "study")
    run = next(a for a in discover(out_dir) if a.kind == "run")
    paths = {d.path for d in config_diff(study.scenario, run.scenario)}
    assert "ai.uq" in paths
    assert "ai.kind" not in paths, "both are 'cached'; an identical key is not a diff"


def test_config_diff_flags_a_block_present_on_only_one_side():
    """An added block — a difficulty model appearing — is exactly the kind of change
    that explains a result, so it must not be silently dropped."""
    deltas = config_diff({"a": 1}, {"a": 1, "difficulty": {"kind": "prior"}})
    assert [(d.path, d.status) for d in deltas] == [("difficulty.kind", "only right")]


def test_config_diff_of_identical_configs_is_empty():
    assert config_diff({"a": {"b": 1}}, {"a": {"b": 1}}) == []


def test_a_study_exposes_its_base_scenario_as_the_thing_to_compare(out_dir):
    study = next(a for a in discover(out_dir) if a.kind == "study")
    assert study.scenario["seed"] == 0
    assert study.scenario["ai"]["kind"] == "cached"


def test_common_metrics_is_the_intersection(out_dir, tmp_path):
    other = json.loads(json.dumps(RUN))
    other["kpis"] = {"system": {"throughput": 0.9}}
    (tmp_path / "summary_q_20260722_130000.json").write_text(json.dumps(other), encoding="utf-8")
    found = discover(out_dir)
    runs = [a for a in found if a.kind == "run"]
    assert common_metrics(runs) == ["system.throughput"]


def test_the_browser_never_writes_into_the_directory_it_reads(out_dir):
    """It is a reader. A result is reproducible because a YAML and a seed produced it;
    a browser that mutated the record would break exactly that guarantee."""
    before = {p.name: p.stat().st_mtime_ns for p in out_dir.iterdir()}
    for artifact in discover(out_dir):
        study_rows(artifact)
        run_kpi_rows(artifact)
        load_records(artifact)
        contrast_rows(artifact)
    after = {p.name: p.stat().st_mtime_ns for p in out_dir.iterdir()}
    assert before == after
