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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hai-sim", description="Human-AI sim runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run a scenario YAML")
    run_p.add_argument("scenario", help="path to scenario YAML")
    run_p.add_argument("--out-dir", default="out", help="output directory (default: out/)")
    run_p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
