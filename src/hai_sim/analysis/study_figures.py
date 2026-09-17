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
BAND = "#f4f3ef"  # alternate-row band: separates the rows without drawing lines


def _read(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _save(fig: Any, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))


# --- E3 -----------------------------------------------------------------------------------


def render_queue_load(
    summary_csv: str | Path,
    out: str | Path,
    metric: str = "system.mean_wait_to_doctor",
    mean_read: float = 13.2,
) -> Path:
    """Mean wait of the patients who HAVE each finding, per ordering, at each load.

    The simulator's clock has no physical unit, so waits are shown in *reads*: multiples of
    ``mean_read``, the reader's mean reading time (lognormal mu 2.5, sigma 0.4 -> 13.2).
    ``arrival rate x mean_read`` is the utilisation of the single reader, which is what
    "half idle / at capacity / overloaded" actually means.
    """
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

    fig, axes = plt.subplots(1, len(rates), figsize=(7.2, 0.27 * len(findings) + 1.95), sharey=True)
    load_name = {0: "Half idle", 1: "At capacity", 2: "Overloaded"}
    for j, (ax, rate) in enumerate(zip(np.atleast_1d(axes), rates, strict=True)):
        for i, f in enumerate(findings):
            y = len(findings) - 1 - i
            values = []
            for kind, color, marker, _ in sorts:
                r = cell.get((rate, kind, f"truth={f}:positive"))
                if r is None:
                    continue
                v = float(r["mean"]) / mean_read
                values.append(v)
                clear = kind == "random" or (
                    r["vs_baseline_ci_low"] != ""
                    and float(r["vs_baseline_ci_low"]) * float(r["vs_baseline_ci_high"]) > 0
                )
                ax.plot(v, y, marker=marker, ms=4.6, mew=1.2, color=color,
                        mfc=color if clear else SURFACE, zorder=3, ls="none")
            if values:
                ax.plot([min(values), max(values)], [y, y], color=RULE, lw=1.2, zorder=1)
        ax.set_title(f"{load_name.get(j, '')}\nutilisation {float(rate) * mean_read:.1f}",
                     fontsize=8, color=INK, loc="left")
        for y in range(0, len(findings), 2):
            ax.axhspan(y - 0.5, y + 0.5, color=BAND, zorder=0, lw=0)
        ax.set_ylim(-0.5, len(findings) - 0.5)
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
    fig.text(0.01, 1 - 0.27 / h, "One reader serves the list; patients arrive at random times. Utilisation = work arriving / "
             "work the reader can do,\nso 0.5 is a half-idle reader, 1 a reader exactly at capacity, "
             "2 twice as much work as can be done.",
             ha="left", va="top", fontsize=7.6, color=INK_2, linespacing=1.35)
    fig.supxlabel("mean wait of the patients who have the finding, in reads (1 read = the mean reading time)",
                  fontsize=8, color=INK_2, y=0.42 / h)
    handles = [Line2D([], [], color=c, marker=m, ls="none", ms=4.6, label=lab) for _k, c, m, lab in sorts]
    handles.append(Line2D([], [], color=INK_2, marker="o", mfc=SURFACE, ls="none", ms=4.6,
                          label="Hollow: paired difference from random includes zero"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.2,
               labelcolor=INK_2, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.5 / h, 1, 1 - 0.58 / h), w_pad=1.0)
    out = Path(out)
    _save(fig, out)
    plt.close(fig)
    return out


# --- E4 -----------------------------------------------------------------------------------

RULE_NAME = {"linear_pool": "average the probabilities", "log_odds_fusion": "add the evidence"}
_T95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float(arr.mean()), float("nan"), float("nan")
    half = _T95.get(arr.size, 1.96) * arr.std(ddof=1) / np.sqrt(arr.size)
    return float(arr.mean()), float(arr.mean() - half), float(arr.mean() + half)


def _team_diffs(long_csv: str | Path) -> dict[tuple[str, str, str, str], dict[str, dict[str, float]]]:
    """``(ai, fusion, persona, pathology) -> stat -> run -> team minus the same reader alone``.

    Both calls come from the same read (the reader's own call is kept next to the team's), so
    the difference is the effect of the advice with the reading noise cancelled.
    """
    rows = _read(long_csv)
    rep_key = "replication" if "replication" in rows[0] else "seed"
    value: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["group"].startswith("label=") and r["value"] not in ("", "nan"):
            value[(r["ai"], r["fusion.kind"], r["persona"], r["group"][6:], r[rep_key])][r["metric"]] = float(r["value"])
    out: dict[tuple[str, str, str, str], dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (ai, fusion, persona, label, run), m in value.items():
        for stat in ("n_fn", "n_fp"):
            a, b = m.get(f"actors.hybrid.{stat}"), m.get(f"actors.doctor.{stat}")
            if a is not None and b is not None:
                out[(ai, fusion, persona, label)][stat][run] = a - b
    return out


def team_effect_table(long_csv: str | Path) -> list[dict[str, Any]]:
    """Per (ai, fusion, persona, pathology): mean difference and 95% t interval over runs."""
    table = []
    for (ai, fusion, persona, label), stats in _team_diffs(long_csv).items():
        row: dict[str, Any] = {"ai": ai, "fusion": fusion, "persona": persona, "pathology": label}
        for stat, runs in stats.items():
            row[f"d_{stat}"], row[f"d_{stat}_lo"], row[f"d_{stat}_hi"] = _mean_ci(list(runs.values()))
        table.append(row)
    return table


def _dot_rows(ax: Any, row_names: list[str], series: list[str], get: Any) -> None:
    """One banded row per name, one marker per series with its 95% interval."""
    n = len(row_names)
    for k in range(n):
        y0 = n - 1 - k
        if k % 2 == 0:
            ax.axhspan(y0 - 0.5, y0 + 0.5, color=BAND, zorder=0, lw=0)
        for a, (name, color) in enumerate(zip(series, SERIES, strict=False)):
            got = get(row_names[k], name)
            if got is None:
                continue
            mean, lo, hi = got
            y = y0 + ((len(series) - 1) / 2 - a) * 0.26
            clear = np.isfinite(lo) and lo * hi > 0
            ax.plot([lo, hi], [y, y], color=color, lw=1.1, zorder=2, solid_capstyle="butt")
            ax.plot(mean, y, marker="osD"[a], ms=4.2, mew=1.1, color=color,
                    mfc=color if clear else SURFACE, zorder=3)
    ax.axvline(0, color=INK, lw=0.9, zorder=1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.grid(axis="x", color=RULE, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def _team_figure(plt: Any, row_labels: list[str], series: list[str], getters: list[Any],
                 title: str, subtitle: str, out: Path, row_height: float = 0.40) -> None:
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(7.2, row_height * len(row_labels) + 2.05), sharey=True)
    for ax, get, what in zip(axes, getters, ("Missed findings", "False alarms"), strict=True):
        _dot_rows(ax, row_labels, series, get)
        ax.set_title(f"{what}: team minus reader alone", fontsize=8.2, color=INK, loc="left")
        lo, hi = ax.get_xlim()
        ax.set_xlim(min(lo, -0.08 * (hi - lo)), max(hi, 0.08 * (hi - lo)))
        ax.set_xlabel("fewer errors  <  0  <  more errors", fontsize=7, color=INK_2, labelpad=2)
    axes[0].set_yticks(range(len(row_labels)))
    axes[0].set_yticklabels([_pretty(x) for x in reversed(row_labels)], fontsize=7.6, color=INK)
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.04 / h, ha="left", va="top", fontsize=10, color=INK, weight="bold")
    fig.text(0.01, 1 - 0.27 / h, subtitle, ha="left", va="top", fontsize=7.6, color=INK_2, linespacing=1.35)
    fig.supxlabel("change in the number of cases, per 2000 patients", fontsize=8, color=INK_2, y=0.52 / h)
    handles = [Line2D([], [], color=c, marker="osD"[a], ls="none", ms=4.4, label=f"advice from {name}")
               for a, (name, c) in enumerate(zip(series, SERIES, strict=False))]
    handles.append(Line2D([], [], color=INK_2, lw=1.1, label="95% interval over 5 runs"))
    handles.append(Line2D([], [], color=INK_2, marker="o", mfc=SURFACE, ls="none", ms=4.4,
                          label="hollow: interval includes zero"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.2,
               labelcolor=INK_2, handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.66 / h, 1, 1 - 0.56 / h), w_pad=1.2)
    _save(fig, out)
    plt.close(fig)


def render_team_effect(long_csv: str | Path, out: str | Path, persona: str = "senior_trusting",
                       fusion: str = "log_odds_fusion") -> Path:
    """Per finding: what the advice of each AI changes for ONE reader under ONE rule."""
    plt = _plt()
    table = [r for r in team_effect_table(long_csv) if r["persona"] == persona and r["fusion"] == fusion]
    series = list(dict.fromkeys(r["ai"] for r in table))[: len(SERIES)]
    cell = {(r["pathology"], r["ai"]): r for r in table}
    labels = sorted({r["pathology"] for r in table if "d_n_fn" in r and abs(r["d_n_fn"]) + abs(r["d_n_fp"]) > 0})

    def getter(stat: str) -> Any:
        def get(label: str, ai: str) -> tuple[float, float, float] | None:
            r = cell.get((label, ai))
            return None if r is None or f"d_{stat}" not in r else (r[f"d_{stat}"], r[f"d_{stat}_lo"], r[f"d_{stat}_hi"])
        return get

    out = Path(out)
    _team_figure(
        plt, labels, series, [getter("n_fn"), getter("n_fp")],
        f"What the AI's advice changes, finding by finding ({persona.replace('_', ' ')} reader)",
        "Each row is a finding. Each marker is the same reader helped by a different AI, compared with that reader\n"
        f"working alone on the same images. Left of the line: the team makes fewer of those errors. "
        f"Rule: {RULE_NAME.get(fusion, fusion)}.",
        out,
    )
    return out


def render_team_rules(long_csv: str | Path, out: str | Path) -> Path:
    """All findings together: every persona under both rules, one marker per AI."""
    plt = _plt()
    diffs = _team_diffs(long_csv)
    totals: dict[tuple[str, str, str], dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for (ai, fusion, persona, _label), stats in diffs.items():
        for stat, runs in stats.items():
            for run, d in runs.items():
                totals[(persona, fusion, ai)][stat][run] += d
    series = list(dict.fromkeys(ai for (_p, _f, ai) in totals))[: len(SERIES)]
    personas = [p for p in ("senior_skeptical", "senior_trusting", "junior_skeptical", "junior_trusting")
                if any(k[0] == p for k in totals)]
    rows = [(p, f) for p in personas for f in ("linear_pool", "log_odds_fusion") if any(k[:2] == (p, f) for k in totals)]
    names = [f"{p.replace('_', ' ')}\n{RULE_NAME[f]}" for p, f in rows]
    index = dict(zip(names, rows, strict=True))

    def getter(stat: str) -> Any:
        def get(name: str, ai: str) -> tuple[float, float, float] | None:
            runs = totals.get((*index[name], ai), {}).get(stat)
            return None if not runs else _mean_ci(list(runs.values()))
        return get

    out = Path(out)
    _team_figure(
        plt, names, series, [getter("n_fn"), getter("n_fp")],
        "Two ways of combining the reader and the AI, all findings together",
        "Each row is a reader and a rule. Averaging the probabilities gives the same result whichever AI advises:\n"
        "the AI is not being used. Adding the evidence uses it: no extra misses, and the better AI removes more false alarms.",
        out, row_height=0.46,
    )
    return out
