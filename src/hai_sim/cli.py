"""Command-line entrypoint: `hai-sim run <scenario.yaml>`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from hai_sim.config.scenario import load_scenario
from hai_sim.factory import build_from_scenario
from hai_sim.metrics.kpi import compute_report
from hai_sim.sim.engine import SimulationEngine
from hai_sim.study import StudyResult, load_study, run_study, write_tidy_csv, write_tidy_json

console = Console()


def cmd_run(args: argparse.Namespace) -> int:
    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        console.print(f"[red]scenario not found: {scenario_path}[/red]")
        return 2
    scenario = load_scenario(scenario_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema, generator, ai, doctor, sort_strategy, severity_model, n_ai, n_doc = (
        build_from_scenario(scenario)
    )

    console.print(
        f"[bold]scenario[/bold]: {scenario.name}  "
        f"seed={scenario.seed}  schema={scenario.label_schema} (n={schema.n})"
    )
    console.print(
        f"[bold]ai[/bold]={ai.name}  [bold]doctor[/bold]={doctor.name}  "
        f"[bold]sort[/bold]={sort_strategy.name}"
    )
    if severity_model is not None:
        console.print(f"[bold]severity[/bold]={severity_model.name}")

    t0 = time.time()
    engine = SimulationEngine(
        patient_generator=generator,
        ai=ai,
        doctor=doctor,
        sort_strategy=sort_strategy,
        severity_model=severity_model,
        n_ai_workers=n_ai,
        n_doctor_workers=n_doc,
    )
    result = engine.run()
    wall = time.time() - t0

    report = compute_report(result.collector.records, scenario.kpis)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"records_{scenario.name}_{stamp}.csv"
    json_path = out_dir / f"summary_{scenario.name}_{stamp}.json"
    result.collector.write_csv(csv_path, label_names=schema.labels)

    summary = {
        "scenario": scenario.model_dump(),
        "ai": ai.name,
        "doctor": doctor.name,
        "sort": sort_strategy.name,
        "wall_clock_s": wall,
        "metadata": result.metadata,
        "kpis": _serialize_report(report),
    }
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    _print_kpi_table(scenario.name, report)
    console.print(f"[green]records -> {csv_path}[/green]")
    console.print(f"[green]summary -> {json_path}[/green]")
    return 0


def cmd_study(args: argparse.Namespace) -> int:
    study_path = Path(args.study)
    if not study_path.exists():
        console.print(f"[red]study not found: {study_path}[/red]")
        return 2
    study = load_study(study_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from hai_sim.study import expand_runs

    n_runs = len(expand_runs(study))
    console.print(
        f"[bold]study[/bold]: {study.name}  design={study.design}  "
        f"factors={list(study.factors)}  reps={len(study.replications)}  runs={n_runs}"
    )

    def progress(done: int, total: int) -> None:
        step = max(1, total // 20)
        if done % step == 0 or done == total:
            console.print(f"  run {done}/{total}", highlight=False)

    t0 = time.time()
    result = run_study(study, progress)
    wall = time.time() - t0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"study_{study.name}_{stamp}.csv"
    json_path = out_dir / f"study_{study.name}_{stamp}.json"
    write_tidy_csv(result, csv_path)
    write_tidy_json(result, json_path)

    _print_study_preview(result)
    console.print(f"[green]tidy csv  -> {csv_path}[/green]")
    console.print(
        f"[green]tidy json -> {json_path}[/green]  ({wall:.1f}s, {len(result.rows)} runs)"
    )
    return 0


def _print_study_preview(result: StudyResult) -> None:
    table = Table(title=f"study {result.study.name} — {len(result.rows)} runs (preview)")
    header = [*result.factor_names, "seed", *result.report_paths]
    for c in header:
        table.add_column(c, justify="right")
    for row in result.rows[:12]:
        table.add_row(*[_fmt(row.get(c)) for c in header])
    console.print(table)
    if len(result.rows) > 12:
        console.print(f"  … {len(result.rows) - 12} more rows in the CSV/JSON")


def _print_kpi_table(name: str, report: dict[str, Any]) -> None:
    actors = report.get("actors")
    if actors:
        table = Table(title=f"KPI - {name}")
        table.add_column("metric")
        table.add_column("ai", justify="right")
        table.add_column("doctor", justify="right")
        table.add_column("hybrid", justify="right")
        for row, getter in [
            ("F1 macro", lambda a: a.f1_macro),
            ("precision macro", lambda a: a.precision_macro),
            ("recall macro", lambda a: a.recall_macro),
            ("accuracy", lambda a: a.accuracy),
            ("n_FN", lambda a: a.n_fn),
            ("n_FP", lambda a: a.n_fp),
            ("TTFE", lambda a: a.ttfe),
            ("MTTF", lambda a: a.mttf),
            ("mean t-to-FN", lambda a: a.mttfn),
        ]:
            table.add_row(
                row,
                _fmt(getter(actors["ai"])),
                _fmt(getter(actors["doctor"])),
                _fmt(getter(actors["hybrid"])),
            )
        console.print(table)
    sysm = report.get("system")
    if sysm:
        console.print(
            f"n_patients={int(sysm['n_patients'])}  sim_t={sysm['sim_duration']:.2f}  "
            f"throughput={sysm['throughput']:.3f}  "
            f"wait_model={sysm['mean_wait_to_model']:.2f}  "
            f"wait_doctor={sysm['mean_wait_to_doctor']:.2f}  "
            f"time_in_sys={sysm['mean_time_in_system']:.2f}"
        )
    team = report.get("team")
    if team:
        console.print(
            f"[bold]team[/bold]: complementarity gap_f1={team.complementarity_gap_f1:+.4f}  "
            f"RAIR={_fmt(team.rair)} (n={team.n_ai_reliance_opportunities})  "
            f"RSR={_fmt(team.rsr)} (n={team.n_self_reliance_opportunities})  "
            f"over={_fmt(team.over_reliance)}  under={_fmt(team.under_reliance)}"
        )
    jt = report.get("joint_table")
    if jt:
        _print_joint_table(name, jt)
    for key, result in report.items():
        if key not in _SPECIALLY_RENDERED:
            _print_generic_bundle(key, result)


# Bundles with a bespoke rendering above; anything else gets the generic table.
_SPECIALLY_RENDERED = frozenset({"actors", "system", "team", "joint_table"})


def _print_generic_bundle(key: str, result: Any) -> None:
    """Render a bundle the CLI knows nothing about — a custom one, most likely.

    A metric the framework lets you plug in has to be *visible* without opening the
    JSON, so an unrecognized bundle gets a plain field/value table rather than being
    quietly dropped from the console.
    """
    if is_dataclass(result) and not isinstance(result, type):
        fields: dict[str, Any] = asdict(result)
    elif isinstance(result, dict):
        fields = result
    else:
        console.print(f"[bold]{key}[/bold]: {_fmt(result)}")
        return
    table = Table(title=key)
    table.add_column("field")
    table.add_column("value", justify="right")
    for field_name, value in fields.items():
        table.add_row(field_name, _fmt(value))
    console.print(table)


def _print_joint_table(name: str, jt: Any) -> None:
    ai_c, doc_c = jt.ai_confusion, jt.doctor_confusion
    console.print(
        f"[bold]AI x Doctor x GT[/bold] (n={jt.n_entries})  "
        f"AI[tp={ai_c['tp']} fp={ai_c['fp']} fn={ai_c['fn']} tn={ai_c['tn']}]  "
        f"Doctor[tp={doc_c['tp']} fp={doc_c['fp']} fn={doc_c['fn']} tn={doc_c['tn']}]"
    )
    t = Table(title=f"joint (AI, Doctor) by GT sign - {name}")
    t.add_column("GT")
    t.add_column("AI+ Doc+", justify="right")
    t.add_column("AI+ Doc-", justify="right")
    t.add_column("AI- Doc+", justify="right")
    t.add_column("AI- Doc-", justify="right")
    for g in (1, 0):
        t.add_row(
            f"GT={g}",
            str(jt.joint[f"gt{g}_ai1_doc1"]),
            str(jt.joint[f"gt{g}_ai1_doc0"]),
            str(jt.joint[f"gt{g}_ai0_doc1"]),
            str(jt.joint[f"gt{g}_ai0_doc0"]),
        )
    console.print(t)


def _serialize_report(report: dict[str, Any]) -> dict[str, Any]:
    """Convert bundle results to JSON-serializable structures (dataclasses → dicts)."""
    out: dict[str, Any] = {}
    for key, result in report.items():
        if key == "actors" and isinstance(result, dict):
            out[key] = {k: asdict(v) for k, v in result.items()}
        elif is_dataclass(result) and not isinstance(result, type):
            out[key] = asdict(result)
        else:
            out[key] = result
    return out


def _fmt(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _validate_from_cache(args: argparse.Namespace, out_dir: Path) -> int:
    from hai_sim.analysis.localization.cache import LocalizationCache
    from hai_sim.analysis.localization.evaluate import evaluate_from_cache
    from hai_sim.analysis.localization.metrics import Dice, EnergyInMask, IoU, PointingGame
    from hai_sim.core.protocols import LocalizationMetric

    cache = LocalizationCache.load(args.from_cache)
    # --pointing-tolerance is given in 224px (model-input) units; the cache scores at
    # mask_resolution, so scale the Chebyshev radius to match the live validator.
    tol = round(args.pointing_tolerance * cache.mask_resolution / 224)
    metrics: list[LocalizationMetric] = [
        PointingGame(tolerance=tol),
        EnergyInMask(),
        IoU(0.5),
        Dice(0.5),
    ]
    report = evaluate_from_cache(
        cache,
        metrics=metrics,
        threshold=args.threshold,
        member_aggregation=args.aggregation,
        split=args.split,
        model_name="+".join(cache.weights),
    )
    name = Path(args.from_cache).stem + (f"_{args.split}" if args.split else "")
    console.print(
        f"[bold]validation (from cache)[/bold]: {name}  members={cache.n_members}  "
        f"images={report.n_images}  thr={args.threshold}  agg={args.aggregation}  "
        f"split={args.split or 'all'}"
    )
    report_path = out_dir / f"{name}_thr{args.threshold:g}_{args.aggregation}_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    _print_validation(name, report)
    console.print(f"[green]report -> {report_path}[/green]")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_cache:
        return _validate_from_cache(args, out_dir)

    from hai_sim.config.scenario import load_validation_scenario
    from hai_sim.factory import build_validation

    if not args.scenario:
        console.print("[red]provide a scenario YAML or --from-cache <cache-stem>[/red]")
        return 2
    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        console.print(f"[red]scenario not found: {scenario_path}[/red]")
        return 2
    scenario = load_validation_scenario(scenario_path)

    _schema, dataset, validator = build_validation(scenario)
    n_total = len(dataset)
    limit = args.limit if args.limit is not None else scenario.limit
    indices = list(range(n_total))
    if limit and limit > 0:
        indices = indices[:limit]

    console.print(
        f"[bold]validation[/bold]: {scenario.name}  model={validator.actor.name}  "
        f"dataset={dataset.name} (n={len(indices)}/{n_total})  saliency={scenario.saliency}"
    )
    records_path = out_dir / f"{scenario.name}_records.json"

    def progress(done: int, total: int) -> None:
        step = max(1, total // 20)
        if done % step == 0 or done == total:
            console.print(f"  processed {done}/{total}", highlight=False)

    t0 = time.time()
    report = validator.run(
        indices=indices,
        records_path=records_path,
        checkpoint_every=args.checkpoint_every,
        progress=progress,
    )
    wall = time.time() - t0

    report_path = out_dir / f"{scenario.name}_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    _print_validation(scenario.name, report)
    console.print(f"[green]report  -> {report_path}[/green]")
    console.print(f"[green]records -> {records_path}[/green]  ({wall:.1f}s)")
    return 0


def _fmt_agg(agg: object) -> str:
    mean = getattr(agg, "mean", None)
    if mean is None:
        return "-"
    return f"{mean:.3f} (n={getattr(agg, 'n_valid', 0)})"


def _print_validation(name: str, report) -> None:
    pooled = report.localization_pooled
    if pooled and any(m.n_total for m in pooled):
        metric_names = [m.metric for m in pooled]
        pp = {m.metric: m for m in report.localization_pooled_predicted}
        t = Table(title=f"Localization pooled - {name}")
        t.add_column("metric")
        t.add_column("all GT-present", justify="right")
        t.add_column("predicted+ only", justify="right")
        for m in pooled:
            t.add_row(m.metric, _fmt_agg(m), _fmt_agg(pp.get(m.metric)))
        console.print(t)

        per = Table(title="Localization per pathology (mean; n instances)")
        per.add_column("pathology")
        for mn in metric_names:
            per.add_column(mn, justify="right")
        for path, aggs in report.localization_per_label.items():
            by = {a.metric: a for a in aggs}
            per.add_row(path, *[_fmt_agg(by.get(mn)) for mn in metric_names])
        console.print(per)
    else:
        console.print("[yellow]no localization scored (saliency: none or no localized findings)[/yellow]")

    lab = Table(title=f"Labels vs expert GT - {name}")
    lab.add_column("pathology")
    lab.add_column("n_pos", justify="right")
    lab.add_column("n_neg", justify="right")
    lab.add_column("sens", justify="right")
    lab.add_column("spec", justify="right")
    for path, c in report.label_confusion.items():
        if c.n_pos == 0 and c.n_neg == 0:
            continue
        lab.add_row(path, str(c.n_pos), str(c.n_neg), _fmt(c.sensitivity), _fmt(c.specificity))
    console.print(lab)
    console.print(
        f"macro sens={_fmt(report.macro_sensitivity)}  "
        f"macro spec={_fmt(report.macro_specificity)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hai-sim", description="Human-AI sim runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run a scenario YAML")
    run_p.add_argument("scenario", help="path to scenario YAML")
    run_p.add_argument("--out-dir", default="out", help="output directory (default: out/)")
    run_p.set_defaults(func=cmd_run)

    study_p = sub.add_parser("study", help="run a parameter study (factor grid → tidy KPI table)")
    study_p.add_argument("study", help="path to study YAML")
    study_p.add_argument("--out-dir", default="out", help="output directory (default: out/)")
    study_p.set_defaults(func=cmd_study)

    val_p = sub.add_parser("validate", help="validate a model's labels + heatmaps vs expert GT")
    val_p.add_argument("scenario", nargs="?", help="validation scenario YAML (direct mode)")
    val_p.add_argument(
        "--from-cache", help="evaluate offline from a localization cache stem instead"
    )
    val_p.add_argument(
        "--out-dir",
        default="out/analysis/localization",
        help="output directory (default: out/analysis/localization/)",
    )
    val_p.add_argument("--limit", type=int, default=None, help="cap images (direct mode)")
    val_p.add_argument(
        "--checkpoint-every", type=int, default=25, help="persist records every N images"
    )
    # --from-cache options (offline, swappable):
    val_p.add_argument("--threshold", type=float, default=0.5, help="decision threshold")
    val_p.add_argument(
        "--aggregation", choices=["mean", "max"], default="mean", help="member CAM aggregation"
    )
    val_p.add_argument("--split", default=None, help="restrict to a split (train/test)")
    val_p.add_argument(
        "--pointing-tolerance", type=int, default=16, help="pointing-game radius px (from-cache)"
    )
    val_p.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
