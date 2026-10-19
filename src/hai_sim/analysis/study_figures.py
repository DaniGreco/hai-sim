"""Figures for the simulation studies, drawn from a study's CSV.

``hai-sim plot queue_load <summary.csv>``   the waiting-time study
``hai-sim plot team_effect <long.csv>``     the human-AI team study

Same look as the capture figures (`hai_sim.analysis.capture_figures`), so all the
figures read as one system.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hai_sim.analysis.capture_figures import (
    DATASET_NAME,
    INK,
    INK_2,
    MODEL_NAME,
    RULE,
    SERIES,
    SPLIT_NAME,
    SURFACE,
    _plt,
    _pretty,
    _wrap,
)
from hai_sim.personas import PERSONA_LIBRARY
from hai_sim.study.stats import _t_critical

NEUTRAL = "#8a8983"
BAND = "#f4f3ef"  # alternate-row band: separates the rows without drawing lines


def _read(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _study_of(csv_path: str | Path) -> dict[str, Any]:
    """The study's own config, read from the JSON written next to the CSV, so a figure can
    state its data, model and sample size without being told."""
    path = Path(csv_path)
    stem = path.name
    for tail in ("_summary.csv", "_long.csv", ".csv"):
        if stem.endswith(tail):
            stem = stem[: -len(tail)]
            break
    found = path.with_name(stem + ".json")
    return json.loads(found.read_text(encoding="utf-8"))["study"] if found.exists() else {}


def _data_line(study: dict[str, Any]) -> str:
    gen = study.get("base", {}).get("patient_generator", {})
    dataset = Path(str(gen.get("cache_stem", ""))).name.split("__")[0]
    n, runs = gen.get("n_patients"), len(study.get("replications", []) or [1])
    once = ", each image read once" if gen.get("sample_mode") == "permutation" else ""
    return (f"Data: {DATASET_NAME.get(dataset, dataset)}, {SPLIT_NAME.get(gen.get('split', 'all'), '')}, "
            f"{n:,} patients per run{once}; {runs} runs per condition.")


def _reader_line(name: str) -> str:
    cfg = PERSONA_LIBRARY.get(name)
    if cfg is None:
        return f"Reader: {name}."
    cap = cfg.capability
    level = "senior" if "senior" in name else "junior" if "junior" in name else name
    text = (f"Reader: simulated {level} radiologist "
            f"(sensitivity {cap.sensitivity:g}, specificity {cap.specificity:g})")
    if cfg.trust is not None and cfg.sees_ai:
        text += f", weight given to the AI {cfg.trust.weight_ai:g}"
    return text + "."


def _save(fig: Any, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))


# --- the waiting-time study --------------------------------------------------------------


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
    study = _study_of(summary_csv)
    base = study.get("base", {})
    ai_key = base.get("ai", {}).get("name") or "ensemble"
    calibrated = ", calibrated per finding" if base.get("ai", {}).get("calibration") else ""
    rows = [r for r in _read(summary_csv) if r["metric"] == metric and r["group"].endswith(":positive")]
    rates = sorted({r["patient_generator.arrival_rate"] for r in rows}, key=float)
    findings = list(dict.fromkeys(r["group"][len("truth="):-len(":positive")] for r in rows))
    counts = _truth_counts(study)
    findings.sort(key=lambda name: -counts.get(name, (0, 0))[0])
    # rows from the most to the least prevalent finding, prevalence shown on the label
    sorts = [("random", NEUTRAL, "o", "Random order"),
             ("uncertainty_desc", SERIES[1], "s", "Most uncertain first"),
             ("probability_desc", SERIES[0], "D", "Most suspicious first")]
    cell = {(r["patient_generator.arrival_rate"], r["sort.kind"], r["group"]): r for r in rows}

    fig, axes = plt.subplots(1, len(rates), figsize=(7.2, 0.27 * len(findings) + 2.3), sharey=True)
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
    first.set_yticklabels([_row_label(f, counts) for f in reversed(findings)], fontsize=7.4, color=INK)
    h = fig.get_figheight()
    fig.suptitle("Waiting time of patients with a finding, by worklist order and reader load",
                 x=0.01, y=1 - 0.04 / h, ha="left", va="top", fontsize=10, color=INK, weight="bold")
    fig.text(0.01, 1 - 0.27 / h,
             _wrap(f"{_data_line(study)} AI ordering the worklist: {MODEL_NAME.get(ai_key, ai_key)}{calibrated}. "
                   f"{_reader_line(str(base.get('persona', '')))[:-1]}, one reader. Patients arrive at random "
                   "times. Utilisation = work arriving / work the reader can do: 0.5 half idle, 1 at capacity, "
                   "2 overloaded."),
             ha="left", va="top", fontsize=7.6, color=INK_2, linespacing=1.35)
    fig.supxlabel("mean wait of the patients who have the finding, in reads (1 read = the mean reading time)",
                  fontsize=8, color=INK_2, y=0.54 / h)
    handles = [Line2D([], [], color=c, marker=m, ls="none", ms=4.6, label=lab) for _k, c, m, lab in sorts]
    handles.append(Line2D([], [], color=INK_2, marker="o", mfc=SURFACE, ls="none", ms=4.6,
                          label="Hollow: not distinguishable from random order"))
    handles.append(Line2D([], [], color=RULE, lw=1.6, label="Grey bar: guide joining the three orders"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.0,
               labelcolor=INK_2, handletextpad=0.4, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.62 / h, 1, 1 - 0.74 / h), w_pad=1.0)
    out = Path(out)
    _save(fig, out)
    plt.close(fig)
    return out


# --- the human-AI team study -------------------------------------------------------------

RULE_NAME = {"linear_pool": "average the probabilities", "log_odds_fusion": "add the evidence"}
def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float(arr.mean()), float("nan"), float("nan")
    half = _t_critical(arr.size - 1) * arr.std(ddof=1) / np.sqrt(arr.size)
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

    fig, axes = plt.subplots(1, 2, figsize=(7.2, row_height * len(row_labels) + 2.25), sharey=True)
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
    fig.text(0.01, 1 - 0.27 / h, _wrap(subtitle), ha="left", va="top", fontsize=7.6, color=INK_2,
             linespacing=1.35)
    fig.supxlabel("change in the number of cases per run", fontsize=8, color=INK_2, y=0.52 / h)
    handles = [Line2D([], [], color=c, marker="osD"[a], ls="none", ms=4.4,
                      label=f"advice from {MODEL_NAME.get(name, name)}")
               for a, (name, c) in enumerate(zip(series, SERIES, strict=False))]
    handles.append(Line2D([], [], color=INK_2, lw=1.1, label="95% interval over the runs"))
    handles.append(Line2D([], [], color=INK_2, marker="o", mfc=SURFACE, ls="none", ms=4.4,
                          label="hollow: interval includes zero"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.2,
               labelcolor=INK_2, handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.66 / h, 1, 1 - 0.74 / h), w_pad=1.2)
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
        "Change in the reader's errors when the AI's advice is shown, by finding",
        f"{_data_line(_study_of(long_csv))} {_reader_line(persona)} Opinions combined by: "
        f"{RULE_NAME.get(fusion, fusion)}. Each marker is the same reader helped by a different AI, compared "
        "with that reader alone on the same images. Horizontal bar: 95% confidence interval of that change "
        "over the runs.",
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
        "Change in the reader's errors with the AI's advice, by reader and combination rule",
        f"{_data_line(_study_of(long_csv))} All findings summed. Each row is a simulated reader and a rule "
        "for combining the two opinions. Each marker is that reader helped by a different AI, compared with the "
        "same reader alone on the same images. Horizontal bar: 95% confidence interval of that change over the runs.",
        out, row_height=0.46,
    )
    return out


# --- the team study: the three actors side by side ------------------------------------------


def _row_label(name: str, counts: dict[str, tuple[int, int]]) -> str:
    """``'11.9 %  Cardiomegaly  (n = 892)'``: prevalence among the patients of a run, then the
    finding, then the count. Rows sorted by prevalence put the calibration story in view."""
    pos, neg = counts[name]
    total = pos + neg
    share = 100.0 * pos / total if total else float("nan")
    return f"{share:4.1f} %   {_pretty(name)}  (n = {pos})"


def _truth_counts(study: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """``pathology -> (patients with it, patients without it)`` among the patients of a run."""
    from hai_sim.analysis.inference_cache import InferenceCache

    gen = study.get("base", {}).get("patient_generator", {})
    cache = InferenceCache.load(gen["cache_stem"])
    rows = np.arange(cache.n_images)
    if gen.get("split", "all") != "all":
        rows = rows[rows % 2 == (0 if gen["split"] == "even" else 1)]
    gt = np.asarray(cache.ground_truth)[rows]
    return {name: (int((gt[:, k] == 1).sum()), int((gt[:, k] == 0).sum()))
            for k, name in enumerate(cache.pathologies)}


def render_actor_errors(long_csv: str | Path, out: str | Path, ai: str = "ensemble",
                        persona: str = "senior_trusting", fusion: str = "log_odds_fusion") -> Path:
    """Per finding: the miss rate and the false-alarm rate of the AI alone, of the reader alone
    and of the two together. The one figure that says what a model is worth."""
    from matplotlib.lines import Line2D

    plt = _plt()
    study = _study_of(long_csv)
    gen = study.get("base", {}).get("patient_generator", {})
    if gen.get("sample_mode") != "permutation":
        raise ValueError("rates need every image read once per run (sample_mode: permutation)")
    counts = _truth_counts(study)
    rows = [r for r in _read(long_csv)
            if r["ai"] == ai and r["persona"] == persona and r["fusion.kind"] == fusion
            and r["group"].startswith("label=") and r["value"] not in ("", "nan")]
    actors = [("ai", "AI alone", SERIES[1], "s"), ("doctor", "Reader alone", NEUTRAL, "o"),
              ("hybrid", "Reader with the AI's advice", SERIES[0], "D")]
    value: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        parts = r["metric"].split(".")
        if len(parts) != 3 or parts[0] != "actors":
            continue
        _, actor, stat = parts
        if stat in ("n_fn", "n_fp"):
            label = r["group"][6:]
            pos, neg = counts[label]
            denom = pos if stat == "n_fn" else neg
            if denom:
                value[(label, actor, stat)].append(100.0 * float(r["value"]) / denom)
    labels = [name for name in counts if counts[name][0] >= 20 and (name, "ai", "n_fn") in value]
    labels.sort(key=lambda name: -counts[name][0])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.40 * len(labels) + 2.55), sharey=True)
    total = int(gen.get("n_patients", 0))
    panels = (("n_fn", "Missed findings\n% of the n patients with the finding"),
              ("n_fp", f"False alarms\n% of the {total:,} - n patients without it"))
    for ax, (stat, title) in zip(axes, panels, strict=True):
        n = len(labels)
        for k, label in enumerate(labels):
            y0 = n - 1 - k
            if k % 2 == 0:
                ax.axhspan(y0 - 0.5, y0 + 0.5, color=BAND, zorder=0, lw=0)
            for a, (actor, _name, color, marker) in enumerate(actors):
                got = value.get((label, actor, stat))
                if not got:
                    continue
                mean, lo, hi = _mean_ci(got)
                y = y0 + (1 - a) * 0.26
                if np.isfinite(lo):
                    ax.plot([lo, hi], [y, y], color=color, lw=1.1, zorder=2, solid_capstyle="butt")
                ax.plot(mean, y, marker=marker, ms=4.4, color=color, zorder=3, ls="none")
        ax.set_title(title, fontsize=8.0, color=INK, loc="left")
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_xlim(left=0)
        ax.grid(axis="x", color=RULE, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels([_row_label(x, counts) for x in reversed(labels)],
                            fontsize=7.4, color=INK)
    h = fig.get_figheight()
    fig.suptitle("Errors of the AI alone, of the reader alone and of the two together, by finding",
                 x=0.01, y=1 - 0.04 / h, ha="left", va="top", fontsize=10, color=INK, weight="bold")
    fig.text(0.01, 1 - 0.27 / h,
             _wrap(f"{_data_line(study)} AI: {MODEL_NAME.get(ai, ai)}, calibrated per finding; on its own it "
                   "calls a finding present above the cut where it calls as many positives as there are. "
                   f"{_reader_line(persona)} Opinions combined by: {RULE_NAME.get(fusion, fusion)}; the team "
                   "calls a finding present at 0.5. n = patients who have the finding. Lower is better in "
                   "both panels. Horizontal bar: 95% confidence interval over the runs."),
             ha="left", va="top", fontsize=7.6, color=INK_2, linespacing=1.35)
    handles = [Line2D([], [], color=c, marker=m, ls="none", ms=4.6, label=name) for _a, name, c, m in actors]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.4, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.32 / h, 1, 1 - 0.95 / h), w_pad=1.2)
    out = Path(out)
    _save(fig, out)
    plt.close(fig)
    return out
