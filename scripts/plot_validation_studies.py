"""Thesis figures for the three validation simulation studies.

Reads the latest summary CSV of each study in ``out/`` (rho_sweep, nih_persona_sdt,
order_load) plus the realized-coupling probe (``out/analysis/rho_probe.json``) and
renders three publication figures, PNG + PDF:

    out/fig_rho_sweep.{png,pdf}          coupling lever: benefit of AI advice vs slope
    out/fig_persona_pathology.{png,pdf}  per-pathology effect of each persona vs solo senior
    out/fig_order_load.{png,pdf}         severity-conditioned time deltas, per load regime

    uv run python scripts/plot_validation_studies.py [--out-dir out]

Everything is per pathology / per severity band: the macro rows are deliberately not
plotted (pooled numbers hide which label the effect lives on).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK = "#0b0b0b"
MUTED = "#7a7a7a"
DIVERGING = LinearSegmentedColormap.from_list("hai_div", (COLORS[1], "#f2f1ee", COLORS[0]))

HIGHLIGHT = ("Effusion", "Cardiomegaly", "Pneumothorax", "Consolidation")


def _frame(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d9d9d9")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color="#ececec", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _latest_summary(out_dir: Path, study: str) -> list[dict[str, str]]:
    candidates = sorted(out_dir.glob(f"study_{study}_*_summary.csv"))
    if not candidates:
        raise FileNotFoundError(f"no study_{study}_*_summary.csv under {out_dir}")
    with candidates[-1].open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _label_of(group: str) -> str | None:
    return group.split("=", 1)[1] if group.startswith("label=") else None


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=170, facecolor="#ffffff", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="#ffffff", bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem.with_suffix('.png')}  (+.pdf)")


# ── Figure 1: the coupling lever ─────────────────────────────────────────────────


def fig_rho_sweep(out_dir: Path) -> None:
    rows = _latest_summary(out_dir, "rho_sweep")
    probe_path = out_dir / "analysis" / "rho_probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else None

    assisted = [r for r in rows if abs(float(r["persona.trust.weight_ai"]) - 0.39) < 1e-9]
    slopes = sorted({float(r["persona.capability.difficulty_slope"]) for r in assisted})

    def series(metric: str, field: str) -> dict[str, dict[float, float]]:
        got: dict[str, dict[float, float]] = {}
        for r in assisted:
            patho = _label_of(r["group"])
            if patho is None or r["metric"] != metric or not r[field]:
                continue
            got.setdefault(patho, {})[float(r["persona.capability.difficulty_slope"])] = float(r[field])
        return got

    n_panels = 3 if probe else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 5.0))
    fig.patch.set_facecolor("#ffffff")

    panels = (
        ("actors.hybrid.f1_macro", "vs_baseline_delta",
         "Δ F1 (assisted - unaided), per pathology", axes[0]),
        ("team.rair", "mean", "RAIR (relative AI reliance)", axes[1]),
    )
    for metric, field, title, ax in panels:
        _frame(ax)
        data = series(metric, field)
        means = []
        for patho, by_slope in sorted(data.items()):
            xs = [s for s in slopes if s in by_slope]
            ys = [by_slope[s] for s in xs]
            if patho in HIGHLIGHT:
                idx = HIGHLIGHT.index(patho)
                ax.plot(xs, ys, lw=2.0, color=COLORS[idx % len(COLORS)], zorder=3, label=patho)
            else:
                ax.plot(xs, ys, lw=0.8, color="#cfcfcf", zorder=2)
        for s in slopes:
            vals = [by_slope[s] for by_slope in data.values() if s in by_slope]
            means.append(float(np.mean(vals)) if vals else np.nan)
        ax.plot(slopes, means, lw=3.0, color=INK, zorder=4, label="mean (all pathologies)")
        if field == "vs_baseline_delta":
            ax.axhline(0.0, color=MUTED, lw=0.9, ls=":", zorder=1)
        ax.set_xlabel("difficulty_slope (coupling to human difficulty)", fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK)
        ax.legend(fontsize=7.5, frameon=False)

    if probe:
        ax = axes[2]
        _frame(ax)
        items = sorted(probe.items(), key=lambda kv: float(kv[0]))
        ps = [float(k) for k, _ in items]
        mean = [v["mean"] for _, v in items]
        lo = [v["min"] for _, v in items]
        hi = [v["max"] for _, v in items]
        ax.fill_between(ps, lo, hi, color=COLORS[0], alpha=0.15, zorder=2,
                        label="per-pathology min-max")
        ax.plot(ps, mean, marker="o", ms=4, lw=2.2, color=COLORS[0], zorder=3,
                label="per-pathology mean")
        ax.axhline(0.0, color=MUTED, lw=0.9, ls=":", zorder=1)
        ax.set_xlabel("difficulty_slope", fontsize=8.5)
        ax.set_ylabel("doctor-AI error correlation (φ)", fontsize=8.5)
        ax.set_title("realized coupling (probe, unaided reader)", loc="left",
                     fontsize=10, fontweight="bold", color=INK)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    fig.suptitle("rho as an experimental lever: shared human difficulty (VinDr, real ensemble)",
                 x=0.01, y=0.99, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.01, 0.925,
             "SDT reader (sens/spec 0.95) + advice weight 0.39 · paired baseline at weight 0 · "
             "difficulty = disagreement of the 3 radiologists (75% of cases at δ=0)",
             ha="left", va="top", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    _save(fig, out_dir / "fig_rho_sweep")


# ── Figure 2: persona x pathology ────────────────────────────────────────────────


def fig_persona_pathology(out_dir: Path) -> None:
    rows = _latest_summary(out_dir, "nih_persona_sdt")

    def persona_name(cell: str) -> str | None:
        try:
            return json.loads(cell.replace("'", '"').replace("True", "true").replace("False", "false"))["name"]
        except Exception:
            return None

    metrics = (("actors.hybrid.recall_macro", "Δ recall (sensitivity)"),
               ("actors.hybrid.n_fp", "Δ false positives (count)"))
    personas = ["senior_skeptical_sdt", "senior_trusting_sdt",
                "junior_skeptical_sdt", "junior_trusting_sdt"]
    short = {p: p.replace("_sdt", "").replace("_", "\n") for p in personas}

    cells: dict[tuple[str, str, str], tuple[float, bool]] = {}
    pathologies: set[str] = set()
    for r in rows:
        name = persona_name(r["persona"])
        patho = _label_of(r["group"])
        if name not in personas or patho is None or not r["vs_baseline_delta"]:
            continue
        delta = float(r["vs_baseline_delta"])
        lo, hi = float(r["vs_baseline_ci_low"]), float(r["vs_baseline_ci_high"])
        significant = (lo > 0) or (hi < 0)
        cells[(r["metric"], name, patho)] = (delta, significant)
        pathologies.add(patho)

    ordered = sorted(pathologies)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.42 * len(ordered) + 2.4))
    fig.patch.set_facecolor("#ffffff")
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        grid = np.full((len(ordered), len(personas)), np.nan)
        sig = np.zeros_like(grid, dtype=bool)
        for (m, name, patho), (delta, s) in cells.items():
            if m != metric:
                continue
            grid[ordered.index(patho), personas.index(name)] = delta
            sig[ordered.index(patho), personas.index(name)] = s
        span = np.nanmax(np.abs(grid)) or 1.0
        ax.imshow(grid, cmap=DIVERGING if metric.endswith("recall_macro") else DIVERGING.reversed(),
                  vmin=-span, vmax=span, aspect="auto")
        for i in range(len(ordered)):
            for j in range(len(personas)):
                if np.isfinite(grid[i, j]):
                    star = "*" if sig[i, j] else ""
                    ax.text(j, i, f"{grid[i, j]:+.2f}{star}", ha="center", va="center",
                            fontsize=6.8, color=INK,
                            fontweight="bold" if sig[i, j] else "normal")
        ax.set_xticks(range(len(personas)))
        ax.set_xticklabels([short[p] for p in personas], fontsize=7.5)
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels(ordered if ax is axes[0] else [""] * len(ordered), fontsize=7.5)
        ax.tick_params(colors=MUTED)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK)
    fig.suptitle("Effect of AI assistance per persona and per pathology (NIH, real ensemble)",
                 x=0.01, y=0.99, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.01, 0.945,
             "paired contrasts vs solo_senior (5 seeds) · * = 95% interval excludes zero · "
             "blue = improves, orange = worsens",
             ha="left", va="top", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    _save(fig, out_dir / "fig_persona_pathology")


# ── Figure 3: order x load, severity-conditioned ─────────────────────────────────


def fig_order_load(out_dir: Path) -> None:
    rows = _latest_summary(out_dir, "order_load")
    rates = sorted({float(r["patient_generator.arrival_rate"]) for r in rows})
    bands = ("severity=<0.35", "severity=0.35-0.75", "severity=>=0.75")
    nice_bands = ("routine\n(<0.35)", "urgent\n(0.35-0.75)", "critical\n(≥0.75)")
    metric = "system.mean_time_in_system"

    # One scale per panel: the three load regimes differ by orders of magnitude, and a
    # shared axis would flatten the sub-saturation panels to invisibility.
    fig, axes = plt.subplots(1, len(rates), figsize=(4.6 * len(rates), 4.6))
    fig.patch.set_facecolor("#ffffff")
    for ax, rate in zip(axes, rates, strict=True):
        _frame(ax)
        deltas, lows, highs = [], [], []
        for band in bands:
            row = next(
                (r for r in rows
                 if r["sort.kind"] == "uncertainty_desc" and r["group"] == band
                 and r["metric"] == metric and r["vs_baseline_delta"]
                 and float(r["patient_generator.arrival_rate"]) == rate),
                None,
            )
            deltas.append(float(row["vs_baseline_delta"]) if row else np.nan)
            lows.append(float(row["vs_baseline_ci_low"]) if row else np.nan)
            highs.append(float(row["vs_baseline_ci_high"]) if row else np.nan)
        x = np.arange(len(bands), dtype=float)
        colors = [COLORS[2] if d < 0 else COLORS[1] for d in deltas]
        ax.bar(x, deltas, 0.55, color=colors, zorder=3)
        ax.errorbar(x, deltas, yerr=[np.subtract(deltas, lows), np.subtract(highs, deltas)],
                    fmt="none", ecolor=INK, elinewidth=1.1, capsize=3, zorder=4)
        ax.axhline(0.0, color=MUTED, lw=0.9, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(nice_bands, fontsize=8)
        ax.set_title(f"arrival rate {rate:g}  (utilization ≈ {rate * 13.2:.1f})",
                     loc="left", fontsize=10, fontweight="bold", color=INK)
        if ax is axes[0]:
            ax.set_ylabel("Δ time in system (uncertainty triage - random)", fontsize=8.5)
    fig.suptitle("Uncertainty triage redistributes waiting: who gains and who pays",
                 x=0.01, y=0.99, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.01, 0.925,
             "paired contrasts vs random order (5 seeds, same patients) · green = time "
             "gained, orange = time paid · error bars = 95% CI",
             ha="left", va="top", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    _save(fig, out_dir / "fig_order_load")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render the validation-study thesis figures")
    p.add_argument("--out-dir", default="out")
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    print("rendering validation figures:")
    fig_rho_sweep(out_dir)
    fig_persona_pathology(out_dir)
    fig_order_load(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
