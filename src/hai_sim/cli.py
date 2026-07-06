"""Command-line entrypoint: `hai-sim run <scenario.yaml>`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from hai_sim.config.scenario import load_scenario
from hai_sim.factory import build_from_scenario
from hai_sim.metrics.kpi import compute_kpis
from hai_sim.sim.engine import SimulationEngine

console = Console()


def cmd_run(args: argparse.Namespace) -> int:
    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        console.print(f"[red]scenario not found: {scenario_path}[/red]")
        return 2
    scenario = load_scenario(scenario_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema, generator, ai, doctor, sort_strategy, n_ai, n_doc = build_from_scenario(scenario)

    console.print(
        f"[bold]scenario[/bold]: {scenario.name}  "
        f"seed={scenario.seed}  schema={scenario.label_schema} (n={schema.n})"
    )
    console.print(
        f"[bold]ai[/bold]={ai.name}  [bold]doctor[/bold]={doctor.name}  "
        f"[bold]sort[/bold]={sort_strategy.name}"
    )

    t0 = time.time()
    engine = SimulationEngine(
        patient_generator=generator,
        ai=ai,
        doctor=doctor,
        sort_strategy=sort_strategy,
        n_ai_workers=n_ai,
        n_doctor_workers=n_doc,
    )
    result = engine.run()
    wall = time.time() - t0

    kpis = compute_kpis(result.collector.records)

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
        "kpis": {
            "n_patients": kpis.n_patients,
            "sim_duration": kpis.sim_duration,
            "throughput": kpis.throughput,
            "mean_wait_to_model": kpis.mean_wait_to_model,
            "mean_wait_to_doctor": kpis.mean_wait_to_doctor,
            "mean_time_in_system": kpis.mean_time_in_system,
            "ai": asdict(kpis.ai),
            "doctor": asdict(kpis.doctor),
            "hybrid": asdict(kpis.hybrid),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    _print_kpi_table(scenario.name, kpis)
    console.print(f"[green]records -> {csv_path}[/green]")
    console.print(f"[green]summary -> {json_path}[/green]")
    return 0


def _print_kpi_table(name: str, kpis) -> None:
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
            _fmt(getter(kpis.ai)),
            _fmt(getter(kpis.doctor)),
            _fmt(getter(kpis.hybrid)),
        )
    console.print(table)
    console.print(
        f"n_patients={kpis.n_patients}  sim_t={kpis.sim_duration:.2f}  "
        f"throughput={kpis.throughput:.3f}  "
        f"wait_model={kpis.mean_wait_to_model:.2f}  "
        f"wait_doctor={kpis.mean_wait_to_doctor:.2f}  "
        f"time_in_sys={kpis.mean_time_in_system:.2f}"
    )


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

    cache = LocalizationCache.load(args.from_cache)
    metrics = [
        PointingGame(tolerance=args.pointing_tolerance),
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
