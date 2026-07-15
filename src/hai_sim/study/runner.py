"""Study runner: execute the expanded scenarios into one tidy results table.

Each `Run` is built and simulated; the requested KPI scalars are pulled out of its
`compute_report` result by dotted path (e.g. ``team.rair``, ``actors.hybrid.recall_macro``)
into a flat row = {factor levels, seed, KPI scalars}. One row per run — ready to group and
plot.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hai_sim.factory import build_from_scenario
from hai_sim.metrics.kpi import compute_report
from hai_sim.sim.engine import SimulationEngine
from hai_sim.study.spec import StudyConfig, expand_runs


def _bundles_for_report(report_paths: tuple[str, ...]) -> tuple[str, ...]:
    """The KPI bundles a report needs = the first segment of each path (order-preserving)."""
    seen: list[str] = []
    for p in report_paths:
        bundle = p.split(".", 1)[0]
        if bundle not in seen:
            seen.append(bundle)
    return tuple(seen)


def _get_report_path(report: dict[str, Any], path: str) -> Any:
    """Traverse a KPI report by dotted path, via dict keys or dataclass attributes."""
    parts = path.split(".")
    cur: Any = report.get(parts[0])
    for p in parts[1:]:
        if cur is None:
            return None
        cur = cur[p] if isinstance(cur, dict) else getattr(cur, p)
    return cur


@dataclass
class StudyResult:
    study: StudyConfig
    factor_names: tuple[str, ...]
    report_paths: tuple[str, ...]
    rows: list[dict[str, Any]] = field(default_factory=list)


def run_study(
    study: StudyConfig, progress: Callable[[int, int], None] | None = None
) -> StudyResult:
    runs = expand_runs(study)
    bundles = _bundles_for_report(study.report)
    factor_names = tuple(study.factors)
    rows: list[dict[str, Any]] = []
    for i, run in enumerate(runs):
        _schema, generator, ai, doctor, sort, severity, n_ai, n_doc = build_from_scenario(
            run.scenario
        )
        result = SimulationEngine(
            patient_generator=generator,
            ai=ai,
            doctor=doctor,
            sort_strategy=sort,
            severity_model=severity,
            n_ai_workers=n_ai,
            n_doctor_workers=n_doc,
        ).run()
        report = compute_report(result.collector.records, bundles)
        row: dict[str, Any] = {name: run.factors.get(name) for name in factor_names}
        row["seed"] = run.seed
        for path in study.report:
            row[path] = _get_report_path(report, path)
        rows.append(row)
        if progress is not None:
            progress(i + 1, len(runs))
    return StudyResult(
        study=study,
        factor_names=factor_names,
        report_paths=tuple(study.report),
        rows=rows,
    )


def _fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    if isinstance(v, (str, int)):
        return str(v)
    return json.dumps(v, default=str)  # e.g. an inline-dict factor level


def write_tidy_csv(result: StudyResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [*result.factor_names, "seed", *result.report_paths]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in result.rows:
            w.writerow([_fmt_cell(row.get(c)) for c in header])


def write_tidy_json(result: StudyResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "study": result.study.model_dump(),
        "factor_names": list(result.factor_names),
        "report_paths": list(result.report_paths),
        "rows": result.rows,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
