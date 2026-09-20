"""Study runner: execute the expanded scenarios into tidy results and their statistics.

Each `Run` is built and simulated, its KPI report is flattened to ``dotted path ->
scalar`` (once per subgroup when the study conditions on anything), and the result is
emitted as **long** rows — one per (cell, seed, subgroup, metric). Long is the canonical
shape because a conditioned study has no fixed column set; the wide table stays available
for the common unconditioned case, and is exactly what it always was.

Two things that need saying:

- **Identical scenarios run once.** Cells whose *built* scenario is byte-for-byte the
  same — a repeated level, two paths that normalize to one config — execute once and
  share the result, so the table still holds every row the researcher asked for without
  paying twice. Note the limit: a factor that is *inert* rather than duplicated (an
  aggregator under a random sort) still lands in the config and so still counts as a
  distinct scenario. Deduplication is a safety net, not a design analyzer.
- **Contrasts are paired.** See `hai_sim.study.stats`.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hai_sim.factory import build_from_scenario, build_kpi_groupings
from hai_sim.metrics.kpi import DEFAULT_KPI_BUNDLES, compute_grouped_report
from hai_sim.metrics.report import flatten_report
from hai_sim.sim.engine import SimulationEngine
from hai_sim.study.spec import Run, StudyConfig, baseline_of, expand_runs
from hai_sim.study.stats import (
    CellSummary,
    Contrast,
    TidyRow,
    paired_contrasts,
    summarize_cells,
)


def _bundles_for_report(report_paths: Sequence[str]) -> tuple[str, ...]:
    """The KPI bundles a report needs = the first segment of each path (order-preserving)."""
    seen: list[str] = []
    for p in report_paths:
        bundle = p.split(".", 1)[0]
        if bundle not in seen:
            seen.append(bundle)
    return tuple(seen)


def _bundles_for(study: StudyConfig, runs: Sequence[Run]) -> tuple[str, ...]:
    """Which bundles to compute: the study's own choice, else what `report` implies,
    else whatever the base scenario already selected."""
    if study.kpis:
        return tuple(study.kpis)
    if study.report:
        return _bundles_for_report(study.report)
    scenario_kpis = runs[0].scenario.kpis if runs else None
    return tuple(scenario_kpis) if scenario_kpis else DEFAULT_KPI_BUNDLES


def _get_report_path(report: dict[str, Any], path: str) -> Any:
    """Traverse a KPI report by dotted path, via dict keys or dataclass attributes."""
    parts = path.split(".")
    cur: Any = report.get(parts[0])
    for p in parts[1:]:
        if cur is None:
            return None
        try:
            cur = cur[p] if isinstance(cur, dict) else getattr(cur, p)
        except (KeyError, AttributeError):
            return None
    return cur


def _fingerprint(run: Run) -> str:
    return json.dumps(run.scenario.model_dump(), sort_keys=True, default=str)


@dataclass
class StudyResult:
    study: StudyConfig
    factor_names: tuple[str, ...]
    report_paths: tuple[str, ...]
    rows: list[dict[str, Any]] = field(default_factory=list)
    """Wide: one row per (run, subgroup), metric paths as columns."""
    long_rows: list[TidyRow] = field(default_factory=list)
    """Canonical: one row per (cell, seed, subgroup, metric)."""
    n_executed: int = 0
    """Distinct scenarios actually simulated; below ``len(rows)`` when cells collapsed."""

    def summaries(self) -> list[CellSummary]:
        return summarize_cells(self.long_rows)

    def contrasts(self) -> list[Contrast]:
        if not self.study.baseline:
            return []
        return paired_contrasts(self.long_rows, lambda f: baseline_of(f, self.study))


def _execute(scenario: Any, bundles: Sequence[str], by: Any) -> dict[str, dict[str, Any]]:
    """Build one scenario, run it and reduce its records to the grouped KPI report.

    Module-level so that a process pool can pickle it: every run of a study is
    independent (its seed lives in the scenario), so runs can execute in parallel.
    """
    built = build_from_scenario(scenario)
    result = SimulationEngine(
        patient_generator=built.generator,
        ai=built.ai,
        doctor=built.doctor,
        sort_strategy=built.sort,
        severity_model=built.severity,
        difficulty_model=built.difficulty,
        fatigue_curve=built.fatigue,
        n_ai_workers=built.n_ai_workers,
        n_doctor_workers=built.n_doctor_workers,
    ).run()
    return compute_grouped_report(
        result.collector.records,
        bundles,
        build_kpi_groupings(by, built.schema),
    )


def run_study(
    study: StudyConfig,
    progress: Callable[[int, int], None] | None = None,
    workers: int = 1,
) -> StudyResult:
    """Run every cell of a study. ``workers > 1`` executes the distinct scenarios in a
    process pool; results are identical to the sequential run because each scenario
    carries its own seed."""
    runs = expand_runs(study)
    bundles = _bundles_for(study, runs)
    factor_names = tuple(study.factors)
    rows: list[dict[str, Any]] = []
    long_rows: list[TidyRow] = []
    metric_paths: list[str] = []

    unique: dict[str, Run] = {}
    for run in runs:
        unique.setdefault(_fingerprint(run), run)
    cache: dict[str, dict[str, dict[str, Any]]] = {}
    if workers > 1 and len(unique) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=min(workers, len(unique))) as pool:
            futures = {
                pool.submit(_execute, r.scenario, bundles, study.by): fp
                for fp, r in unique.items()
            }
            for k, fut in enumerate(as_completed(futures)):
                cache[futures[fut]] = fut.result()
                if progress is not None:
                    progress(k + 1, len(unique))
    else:
        for k, (fp, r) in enumerate(unique.items()):
            cache[fp] = _execute(r.scenario, bundles, study.by)
            if progress is not None:
                progress(k + 1, len(unique))

    for run in runs:
        grouped = cache[_fingerprint(run)]

        for group, report in grouped.items():
            flat = flatten_report(report)
            if study.report:
                selected = {
                    path: flat[path] if path in flat else _get_report_path(report, path)
                    for path in study.report
                }
            else:
                selected = dict(flat)
            row: dict[str, Any] = {name: run.factors.get(name) for name in factor_names}
            row["seed"] = run.seed
            row["group"] = group
            for path, value in selected.items():
                row[path] = value
                if path not in metric_paths:
                    metric_paths.append(path)
                long_rows.append(
                    TidyRow(
                        factors=dict(run.factors),
                        seed=run.seed,
                        group=group,
                        metric=path,
                        value=value,
                    )
                )
            rows.append(row)

    return StudyResult(
        study=study,
        factor_names=factor_names,
        report_paths=tuple(study.report) if study.report else tuple(metric_paths),
        rows=rows,
        long_rows=long_rows,
        n_executed=len(cache),
    )


def _fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    if isinstance(v, (str, int)):
        return str(v)
    if isinstance(v, dict) and isinstance(v.get("name"), str):
        return str(v["name"])  # a named block (persona, ai) reads as its name
    return json.dumps(v, default=str)  # e.g. an inline-dict factor level


def _write_rows(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_fmt_cell(v) for v in row])


def write_tidy_csv(result: StudyResult, path: Path) -> None:
    """Wide: one row per (run, subgroup), metric paths as columns.

    The ``group`` column appears only when the study conditions on something, so an
    unconditioned study writes byte-for-byte the format it always wrote.
    """
    grouped = any(row.get("group") for row in result.rows)
    header = [
        *result.factor_names,
        "seed",
        *(["group"] if grouped else []),
        *result.report_paths,
    ]
    _write_rows(path, header, [[row.get(c) for c in header] for row in result.rows])


def write_long_csv(result: StudyResult, path: Path) -> None:
    """Long: one row per (cell, seed, subgroup, metric) — the shape to group and plot."""
    header = [*result.factor_names, "seed", "group", "metric", "value"]
    _write_rows(
        path,
        header,
        [
            [
                *[row.factors.get(name) for name in result.factor_names],
                row.seed,
                row.group,
                row.metric,
                row.value,
            ]
            for row in result.long_rows
        ],
    )


def write_summary_csv(result: StudyResult, path: Path) -> None:
    """Per-cell statistics over the replication seeds, plus the paired contrasts."""
    contrast_of = {
        (json.dumps(c.factors, sort_keys=True, default=str), c.group, c.metric): c
        for c in result.contrasts()
    }
    header = [
        *result.factor_names,
        "group",
        "metric",
        "n",
        "mean",
        "sd",
        "ci_low",
        "ci_high",
        "vs_baseline_delta",
        "vs_baseline_ci_low",
        "vs_baseline_ci_high",
        "vs_baseline_n_pairs",
    ]
    rows: list[list[Any]] = []
    for summary in result.summaries():
        key = (
            json.dumps(summary.factors, sort_keys=True, default=str),
            summary.group,
            summary.metric,
        )
        contrast = contrast_of.get(key)
        rows.append(
            [
                *[summary.factors.get(name) for name in result.factor_names],
                summary.group,
                summary.metric,
                summary.stat.n,
                summary.stat.mean,
                summary.stat.sd,
                summary.stat.ci_low,
                summary.stat.ci_high,
                contrast.delta.mean if contrast else None,
                contrast.delta.ci_low if contrast else None,
                contrast.delta.ci_high if contrast else None,
                contrast.n_pairs if contrast else None,
            ]
        )
    _write_rows(path, header, rows)


def write_tidy_json(result: StudyResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "study": result.study.model_dump(),
        "factor_names": list(result.factor_names),
        "report_paths": list(result.report_paths),
        "n_runs": len(result.rows),
        "n_executed": result.n_executed,
        "rows": result.rows,
        "summaries": [
            {
                "factors": s.factors,
                "group": s.group,
                "metric": s.metric,
                "n": s.stat.n,
                "mean": s.stat.mean,
                "sd": s.stat.sd,
                "ci_low": s.stat.ci_low,
                "ci_high": s.stat.ci_high,
            }
            for s in result.summaries()
        ],
        "contrasts": [
            {
                "factors": c.factors,
                "baseline": c.baseline,
                "group": c.group,
                "metric": c.metric,
                "arm_mean": c.arm_mean,
                "baseline_mean": c.baseline_mean,
                "delta_mean": c.delta.mean,
                "delta_ci_low": c.delta.ci_low,
                "delta_ci_high": c.delta.ci_high,
                "n_pairs": c.n_pairs,
                "excludes_zero": c.delta.excludes_zero,
            }
            for c in result.contrasts()
        ],
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
