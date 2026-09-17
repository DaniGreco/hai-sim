"""Figures for the simulation studies, drawn from a study's CSV. English only.

``hai-sim plot queue_load <summary.csv>``   experiment E3
``hai-sim plot team_effect <long.csv>``     experiment E4

Same look as the capture figures (`hai_sim.analysis.capture_figures`), so the chapter
reads as one system.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hai_sim.analysis.capture_figures import INK, INK_2, RULE, SERIES, SURFACE, _plt, _pretty

NEUTRAL = "#8a8983"


def _read(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _save(fig: Any, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))


# --- E3 -----------------------------------------------------------------------------------


def render_queue_load(summary_csv: str | Path, out: str | Path, metric: str = "system.mean_wait_to_doctor") -> Path:
    """Mean wait of the patients who HAVE each finding, per ordering, at each load."""
    from matplotlib.lines import Line2D

    plt = _plt()
    rows = [r for r in _read(summary_csv) if r["metric"] == metric and r["group"].endswith(":positive")]
    rates = sorted({r["patient_generator.arrival_rate"] for r in rows}, key=float)
    findings = list(dict.fromkeys(r["group"][len("truth="):-len(":positive")] for r in rows))
    # heaviest random-order wait at the top load decides nothing; keep the dataset's order
    sorts = [("random", NEUTRAL, "o", "Random order"),
             ("uncertainty_desc", SERIES[1], "s", "Most uncertain first"),
             ("probability_desc", SERIES[0], "D", "Most suspicious first")]
    cell = {(r["patient_generator.arrival_rate"], r["sort.kind"], r["group"]): r for r in rows}

    fig, axes = plt.subplots(1, len(rates), figsize=(7.2, 0.27 * len(findings) + 1.55), sharey=True)
    load_name = {0: "Half idle", 1: "At capacity", 2: "Overloaded"}
    for j, (ax, rate) in enumerate(zip(np.atleast_1d(axes), rates, strict=True)):
        for i, f in enumerate(findings):
            y = len(findings) - 1 - i
            values = []
            for kind, color, marker, _ in sorts:
                r = cell.get((rate, kind, f"truth={f}:positive"))
                if r is None:
                    continue
                v = float(r["mean"])
                values.append(v)
                clear = kind == "random" or (
                    r["vs_baseline_ci_low"] != ""
                    and float(r["vs_baseline_ci_low"]) * float(r["vs_baseline_ci_high"]) > 0
                )
                ax.plot(v, y, marker=marker, ms=4.6, mew=1.2, color=color,
                        mfc=color if clear else SURFACE, zorder=3, ls="none")
            if values:
                ax.plot([min(values), max(values)], [y, y], color=RULE, lw=1.2, zorder=1)
        ax.set_title(f"{load_name.get(j, '')}\narrival rate {float(rate):g}", fontsize=8, color=INK, loc="left")
        ax.set_xlim(left=0)
        ax.grid(axis="x", color=RULE, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(range(len(findings)))
    first.set_yticklabels([_pretty(f) for f in reversed(findings)], fontsize=7.4, color=INK)
    h = fig.get_figheight()
    fig.suptitle("Do patients with a finding wait less when the worklist is reordered?",
                 x=0.01, y=1 - 0.04 / h, ha="left", va="top", fontsize=10, color=INK, weight="bold")
    fig.supxlabel("mean wait before the reader, patients who have the finding (time units)",
                  fontsize=8, color=INK_2, y=0.42 / h)
    handles = [Line2D([], [], color=c, marker=m, ls="none", ms=4.6, label=lab) for _k, c, m, lab in sorts]
    handles.append(Line2D([], [], color=INK_2, marker="o", mfc=SURFACE, ls="none", ms=4.6,
                          label="Hollow: paired difference from random includes zero"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.2,
               labelcolor=INK_2, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.5 / h, 1, 1 - 0.24 / h), w_pad=1.0)
    out = Path(out)
    _save(fig, out)
    plt.close(fig)
    return out


# --- E4 -----------------------------------------------------------------------------------


def team_effect_table(long_csv: str | Path) -> list[dict[str, Any]]:
    """Per (ai, fusion, persona, pathology): team minus the same reader alone, on the same
    reads, averaged over seeds with a 95% t interval over the per-seed differences."""
    rows = _read(long_csv)
    value: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    rep_key = "replication" if "replication" in rows[0] else "seed"
    for r in rows:
        if not r["group"].startswith("label="):
            continue
        key = (r["ai"], r["fusion.kind"], r["persona"], r["group"][6:], r[rep_key])
        value[key][r["metric"]] = float(r["value"]) if r["value"] not in ("", "nan") else float("nan")
    diffs: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (ai, fusion, persona, label, _rep), m in value.items():
        for stat in ("n_fn", "n_fp", "recall_macro"):
            a, b = m.get(f"actors.hybrid.{stat}"), m.get(f"actors.doctor.{stat}")
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                diffs[(ai, fusion, persona, label)][stat].append(a - b)
    t95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}
    table = []
    for (ai, fusion, persona, label), stats in diffs.items():
        row: dict[str, Any] = {"ai": ai, "fusion": fusion, "persona": persona, "pathology": label}
        for stat, d in stats.items():
            arr = np.asarray(d)
            half = t95.get(arr.size, 1.96) * arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else float("nan")
            row[f"d_{stat}"] = float(arr.mean())
            row[f"d_{stat}_lo"], row[f"d_{stat}_hi"] = float(arr.mean() - half), float(arr.mean() + half)
        table.append(row)
    return table


def render_team_effect(long_csv: str | Path, out: str | Path, persona: str = "senior_trusting") -> Path:
    """Missed findings and false alarms the advice adds or removes, per pathology and AI."""
    from matplotlib.lines import Line2D

    plt = _plt()
    table = [r for r in team_effect_table(long_csv) if r["persona"] == persona and "d_n_fn" in r]
    ais = list(dict.fromkeys(r["ai"] for r in table))[: len(SERIES)]
    fusions = [f for f in ("linear_pool", "log_odds_fusion") if any(r["fusion"] == f for r in table)]
    labels = sorted({r["pathology"] for r in table if abs(r["d_n_fn"]) + abs(r["d_n_fp"]) > 0})
    cell = {(r["ai"], r["fusion"], r["pathology"]): r for r in table}
    titles = {"linear_pool": "Linear pool (average the probabilities)",
              "log_odds_fusion": "Logarithmic pool (add the evidence)"}
    stats = (("d_n_fn", "missed findings"), ("d_n_fp", "false alarms"))

    fig, axes = plt.subplots(len(fusions), 2, figsize=(7.2, (0.25 * len(labels) + 0.85) * len(fusions) + 0.9),
                             sharey=True, squeeze=False)
    for i, fusion in enumerate(fusions):
        for j, (stat, what) in enumerate(stats):
            ax = axes[i][j]
            ax.axvline(0, color=INK_2, lw=0.8, zorder=1)
            for k, lab in enumerate(labels):
                y0 = len(labels) - 1 - k
                for a, (ai, color) in enumerate(zip(ais, SERIES, strict=False)):
                    r = cell.get((ai, fusion, lab))
                    if r is None or stat not in r:
                        continue
                    y = y0 + (a - (len(ais) - 1) / 2) * 0.24
                    clear = r[f"{stat}_lo"] * r[f"{stat}_hi"] > 0
                    ax.plot([r[f"{stat}_lo"], r[f"{stat}_hi"]], [y, y], color=color, lw=1.0, zorder=2)
                    ax.plot(r[stat], y, marker="osD"[a], ms=3.8, mew=1.0, color=color,
                            mfc=color if clear else SURFACE, zorder=3)
            head = titles[fusion] if j == 0 else ""
            ax.set_title(f"{head}\nchange in {what}", fontsize=7.8, color=INK, loc="left")
            ax.grid(axis="x", color=RULE, lw=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
        axes[i][0].set_yticks(range(len(labels)))
        axes[i][0].set_yticklabels([_pretty(x) for x in reversed(labels)], fontsize=7.2, color=INK)
    h = fig.get_figheight()
    fig.suptitle(f"What the advice changes for the reader, per finding ({persona.replace('_', ' ')})",
                 x=0.01, y=1 - 0.04 / h, ha="left", va="top", fontsize=10, color=INK, weight="bold")
    fig.supxlabel("team minus the same reader alone, cases per run (left of zero = fewer)",
                  fontsize=8, color=INK_2, y=0.42 / h)
    handles = [Line2D([], [], color=c, marker="osD"[a], ls="none", ms=4.2, label=ai)
               for a, (ai, c) in enumerate(zip(ais, SERIES, strict=False))]
    handles.append(Line2D([], [], color=INK_2, marker="o", mfc=SURFACE, ls="none", ms=4.2,
                          label="Hollow: 95% interval includes zero"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.2, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.5 / h, 1, 1 - 0.24 / h), w_pad=1.0, h_pad=1.2)
    out = Path(out)
    _save(fig, out)
    plt.close(fig)
    return out
