"""Figures, table and numbers of the paper (RQ1 to RQ4), from the caches.

    uv run python scripts/paper_figures.py \
        --queue-summary out/study_queue_load_medgemma27b_<stamp>_summary.csv \
        --team-long out/study_team_effect_medgemma27b_<stamp>_long.csv \
        --out ../overleaf-paper/images

Every figure fits one column of the paper (3.5 in), except the RQ4 one, which spans both.
RQ1 and RQ2 rerun the capture study of configs/studies/capture.yaml (about a minute) and keep
the ten findings annotated on both datasets, three panels per row.
RQ1 draws, for the ensemble, the positive cases met along the order of each criterion, and
tests the criteria against each other with the Wilcoxon signed-rank test, paired by finding.
RQ2 draws the false negatives of each model met along its own order, prints the skill table
as LaTeX rows, how often each model is the best, and the Wilcoxon tests of MedGemma 27B
against the other two models.
RQ3 draws the waiting-time figure of the MedGemma 27B queue study and prints the mean waits
quoted in the text. A wait is shown in mean reading times: when the reader is never idle,
that is the number of patients read before the waiting one.
RQ4 draws the errors of the AI alone, of the reader alone and of the two teams, one quarter
per reader profile, from the MedGemma 27B team study, and prints the numbers quoted in the text.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

from hai_sim.analysis import capture_figures
from hai_sim.analysis.capture_study import load_config, run_capture_study
from hai_sim.analysis.inference_cache import InferenceCache
from hai_sim.analysis.study_figures import BAND, NEUTRAL, _mean_ci, _read, _study_of, _truth_counts

# The ten findings annotated on both datasets (Lung Opacity is VinDr-CXR only), most
# prevalent on VinDr-CXR first. VinDr-CXR annotates Nodule and Mass as one finding.
COMMON = ["Cardiomegaly", "Fibrosis", "Pleural_Thickening", "Effusion", "Nodule", "Mass",
          "Infiltration", "Consolidation", "Atelectasis", "Pneumothorax"]
NAME = {"Fibrosis": "Pulmonary fibrosis", "Pleural_Thickening": "Pleural thickening",
        "Effusion": "Pleural effusion"}
MODELS = ["ensemble", "medgemma-4b", "medgemma-27b"]
DATASETS = ["vindr", "nih"]
KEY = ("fn", "predictive_entropy", "calibrated")  # RQ2: own misses, calibrated entropy order
# RQ1: the three criteria of the curves, as (ordering, probabilities) of the capture study.
# Platt scaling is monotone per finding, so the raw and the calibrated probability give the
# same order; the entropy is the calibrated one, the member entropy needs two members.
PROB, ENTROPY, MEMBER = ("probability", "raw"), ("predictive_entropy", "calibrated"), ("mean_entropy", "raw")
MEAN_READ = 13.2
PERSONAS = [("senior_skeptical", "Senior, skeptical"), ("senior_trusting", "Senior, trusting"),
            ("junior_skeptical", "Junior, skeptical"), ("junior_trusting", "Junior, trusting")]


def _holm(pvalues: dict[Any, float]) -> dict[Any, float]:
    """Holm step-down adjusted p-values."""
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[k]))
        adjusted[k] = running
    return adjusted


def _paired(a: list[float], b: list[float]) -> str:
    """How often a beats b, the median difference and the exact two-sided Wilcoxon test.
    Pairs with no difference carry no sign and are dropped, as in the Wilcoxon test."""
    d = [x - y for x, y in zip(a, b, strict=True)]
    nonzero = [v for v in d if round(v, 6) != 0]
    text = (f"higher on {sum(v > 0 for v in nonzero)}, lower on {sum(v < 0 for v in nonzero)}, "
            f"tied on {len(d) - len(nonzero)} of {len(d)}, median difference {statistics.median(d):+.3f}")
    if len(nonzero) < 2:
        return text + ", too few untied pairs for a test"
    w = wilcoxon(nonzero, method="exact")
    return text + f", W = {w.statistic:.0f}, p = {w.pvalue:.4f} (n = {len(nonzero)})"


def rq1(result: dict[str, Any], split: str, out: Path) -> None:
    for ds in DATASETS:
        capture_figures.render_orderings(
            result["curves"], ds, "ensemble", "positive", out / f"rq1_pos_{ds}", split,
            pathologies=COMMON, cols=3, column=True,
        )
        print(f"figure -> {out / f'rq1_pos_{ds}.pdf'}")

    skill = {(r["dataset"], r["model"], r["pathology"], r["ordering"], r["probabilities"]): r["skill"]
             for r in result["rows"] if r["target"] == "positive"}
    print("\nRQ1 skill on the positive cases: probability | calibrated entropy | member entropy")
    for ds in DATASETS:
        for m in MODELS:
            print(f"  {ds} {m}")
            for p in COMMON:
                cells = [skill.get((ds, m, p, *c)) for c in (PROB, ENTROPY, MEMBER)]
                print(f"    {NAME.get(p, p):20s} " + "  ".join("  n/a" if v is None else f"{v:+.2f}" for v in cells))

    print("\nRQ1 tests, paired by finding (skill of the first criterion minus the second)")
    pvalues = {}
    for ds in DATASETS:
        for m in MODELS:
            a = [skill[(ds, m, p, *PROB)] for p in COMMON]
            b = [skill[(ds, m, p, *ENTROPY)] for p in COMMON]
            print(f"  {ds} {m}: probability vs calibrated entropy: {_paired(a, b)}")
            d = [x - y for x, y in zip(a, b, strict=True) if round(x - y, 6) != 0]
            if len(d) >= 2:
                pvalues[(ds, m)] = wilcoxon(d, method="exact").pvalue
        both = [p for p in COMMON if (ds, "ensemble", p, *MEMBER) in skill]
        a = [skill[(ds, "ensemble", p, *PROB)] for p in both]
        b = [skill[(ds, "ensemble", p, *MEMBER)] for p in both]
        print(f"  {ds} ensemble: probability vs member entropy on {len(both)} findings: {_paired(a, b)}")
        print(f"    (with {len(both)} pairs the smallest possible two-sided p is {2 / 2 ** len(both):.4f})")
    print("  Holm-adjusted p of probability vs calibrated entropy: "
          + ", ".join(f"{ds} {m} {p:.3f}" for (ds, m), p in _holm(pvalues).items()))


def rq2(result: dict[str, Any], split: str, out: Path) -> None:
    for ds in DATASETS:
        capture_figures.render_models(
            result["curves"], MODELS, ds, "fn", out / f"rq2_fn_{ds}", split, None,
            ordering=KEY[1], kind=KEY[2], pathologies=COMMON, cols=3, column=True,
        )
        print(f"figure -> {out / f'rq2_fn_{ds}.pdf'}")

    skill = {(r["dataset"], r["pathology"], r["model"]): r["skill"] for r in result["rows"]
             if (r["target"], r["ordering"], r["probabilities"]) == KEY}
    print("\n% skill table rows: VinDr-CXR DN, MG 4B, MG 27B | ChestX-ray14 DN, MG 4B, MG 27B")
    for p in COMMON:
        cells = []
        for ds in DATASETS:
            values = {m: skill[(ds, p, m)] for m in MODELS}
            best = max(values.values())
            cells += [f"\\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}" for v in values.values()]
        print(f"        {NAME.get(p, p)} & " + " & ".join(cells) + r" \\")

    for ds in DATASETS:
        x = {m: [skill[(ds, p, m)] for p in COMMON] for m in MODELS}
        wins = Counter(max(MODELS, key=lambda m, p=p: skill[(ds, p, m)]) for p in COMMON)
        print(f"\n{ds}: best model per finding {dict(wins)}")
        print(f"  Friedman over the three models: p = {friedmanchisquare(*x.values()).pvalue:.4f}")
        for other in MODELS[:2]:
            d = [a - b for a, b in zip(x["medgemma-27b"], x[other], strict=True)]
            w = wilcoxon(x["medgemma-27b"], x[other], method="exact")
            print(f"  MedGemma 27B vs {other}: higher on {sum(v > 0 for v in d)}/10, "
                  f"median difference {statistics.median(d):+.3f}, W = {w.statistic:.0f}, p = {w.pvalue:.4f}")


def render_wait(summary: Path, out: Path) -> None:
    """Mean wait of the patients with each finding, per order and load, in one paper column."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    plt = capture_figures._plt()
    ink, ink_2, rule, surface = (capture_figures.INK, capture_figures.INK_2, capture_figures.RULE,
                                 capture_figures.SURFACE)
    rows = [r for r in csv.DictReader(summary.open(encoding="utf-8"))
            if r["metric"] == "system.mean_wait_to_doctor" and r["group"].endswith(":positive")]
    cell = {(r["patient_generator.arrival_rate"], r["sort.kind"], r["group"][6:-9]): r for r in rows}
    rates = sorted({r["patient_generator.arrival_rate"] for r in rows}, key=float)
    cache = InferenceCache.load("data/cache/composed/vindr__ens4")  # same ids and truth as every cell
    labels = list(cache.pathologies)
    gt = cache.ground_truth[1::2]  # the odd half the study reads
    share = {f: 100 * gt[:, labels.index(f)].mean() for f in COMMON}
    findings = sorted(COMMON, key=lambda f: -share[f])
    sorts = [("random", NEUTRAL, "o", "Random order"),
             ("uncertainty_desc", capture_figures.SERIES[1], "s", "Most uncertain first"),
             ("probability_desc", capture_figures.SERIES[0], "D", "Most suspicious first")]
    names = ["Half idle", "At capacity", "Overloaded"]

    fig, axes = plt.subplots(1, len(rates), figsize=(3.5, 0.19 * len(findings) + 1.0), sharey=True)
    for j, (ax, rate) in enumerate(zip(axes, rates, strict=True)):
        for i, f in enumerate(findings):
            y = len(findings) - 1 - i
            if i % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color=BAND, zorder=0, lw=0)
            values = []
            for kind, color, marker, _ in sorts:
                r = cell[(rate, kind, f)]
                v = float(r["mean"]) / MEAN_READ
                values.append(v)
                clear = kind == "random" or float(r["vs_baseline_ci_low"]) * float(r["vs_baseline_ci_high"]) > 0
                ax.plot(v, y, marker=marker, ms=3.6, mew=0.9, color=color, mfc=color if clear else surface,
                        ls="none", zorder=3)
            ax.plot([min(values), max(values)], [y, y], color=rule, lw=1.0, zorder=1)
        rho = round(float(rate) * MEAN_READ, 1)
        ax.set_title(f"{names[j]}\n$\\rho$ = {rho:g}", fontsize=6.8, color=ink, loc="left", pad=3,
                     linespacing=1.2)
        ax.set_ylim(-0.5, len(findings) - 0.5)
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
        ax.grid(axis="x", color=rule, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=6, colors=ink_2, pad=1.5, length=2)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    # finding name, then its share of the patients
    axes[0].set_yticks(range(len(findings)))
    axes[0].set_yticklabels([f"{NAME.get(f, f)}  {share[f]:.1f} %" for f in reversed(findings)],
                            fontsize=6.4, color=ink)
    h = fig.get_figheight()
    fig.supxlabel("mean wait, in patients read before them (left is better)", fontsize=6.6,
                  color=ink_2, y=0.2 / h, va="bottom")
    handles = [Line2D([], [], color=c, marker=m, ls="none", ms=3.6, label=lab) for _k, c, m, lab in sorts]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.4, labelcolor=ink_2,
               handletextpad=0.2, columnspacing=0.9, borderaxespad=0.1)
    fig.subplots_adjust(left=1.3 / 3.5, right=0.98, top=1 - 0.3 / h, bottom=0.52 / h, wspace=0.12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def rq3(summary: Path, out: Path) -> None:
    render_wait(summary, out / "rq3_wait")
    print(f"\nfigure -> {out / 'rq3_wait.pdf'}")
    rows = [r for r in csv.DictReader(summary.open(encoding="utf-8"))
            if r["metric"] == "system.mean_wait_to_doctor" and r["group"].startswith("truth=")]
    wait: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        wait[(r["patient_generator.arrival_rate"], r["sort.kind"])][r["group"][6:]] = float(r["mean"]) / MEAN_READ
    for rate in sorted({k[0] for k in wait}, key=float):
        print(f"utilisation {float(rate) * MEAN_READ:.2f}, waits in mean reading times")
        for order in ("random", "uncertainty_desc", "probability_desc"):
            w = {g[:-9]: v for g, v in wait[(rate, order)].items() if g.endswith(":positive")}
            lo, hi = min(w, key=w.get), max(w, key=w.get)
            print(f"  {order:17s} {w[lo]:9.1f} ({lo}) .. {w[hi]:9.1f} ({hi})")
        gap = max(abs(wait[(rate, "uncertainty_desc")][g] - wait[(rate, "probability_desc")][g])
                  for g in wait[(rate, "probability_desc")] if g.endswith(":positive"))
        print(f"  largest gap between the two AI orders: {gap:.1f}")
        neg = "Cardiomegaly:negative"
        print(f"  patients without Cardiomegaly: random {wait[(rate, 'random')][neg]:.0f}, "
              f"most suspicious first {wait[(rate, 'probability_desc')][neg]:.0f}")


def _team_rates(long_csv: Path) -> tuple[dict[tuple[str, ...], list[float]], dict[str, tuple[int, int]]]:
    """``(persona, actor, stat, finding) -> rate per run, in %``, with actor ai, doctor, or the
    team under one rule (``log_odds_fusion``, ``linear_pool``); and the truth counts.

    The AI alone and the reader alone are taken from the runs of the logarithmic pool: the
    rule changes neither of the two."""
    study = _study_of(long_csv)
    counts = _truth_counts(study)
    rates: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for r in _read(long_csv):
        parts = r["metric"].split(".")
        if not r["group"].startswith("label=") or r["value"] in ("", "nan") or parts[0] != "actors":
            continue
        _, actor, stat = parts
        if stat not in ("n_fn", "n_fp"):
            continue
        finding = r["group"][6:]
        pos, neg = counts[finding]
        rate = 100.0 * float(r["value"]) / (pos if stat == "n_fn" else neg)
        rule = r["fusion.kind"]
        if actor == "hybrid":
            rates[(r["persona"], rule, stat, finding)].append(rate)
        elif rule == "log_odds_fusion":
            rates[(r["persona"], actor, stat, finding)].append(rate)
        else:
            rates[(r["persona"], f"{actor}@linear", stat, finding)].append(rate)
    return rates, counts


def render_team(long_csv: Path, out: Path) -> None:
    """One quarter per reader profile, two panels per quarter (missed findings, false alarms),
    one row per finding, four actors per row."""
    from matplotlib.lines import Line2D

    plt = capture_figures._plt()
    ink, ink_2, rule = capture_figures.INK, capture_figures.INK_2, capture_figures.RULE
    rates, counts = _team_rates(long_csv)
    findings = sorted(COMMON, key=lambda f: -counts[f][0])
    share = {f: 100 * counts[f][0] / sum(counts[f]) for f in findings}
    actors = [("ai", "AI alone", capture_figures.SERIES[1], "s"),
              ("doctor", "Reader alone", NEUTRAL, "o"),
              ("linear_pool", "Reader + AI, linear pool", capture_figures.SERIES[2], "^"),
              ("log_odds_fusion", "Reader + AI, log-odds", capture_figures.SERIES[0], "D")]
    n = len(findings)
    fig = plt.figure(figsize=(7.16, 0.17 * n * 2 + 1.2))
    quarters = fig.add_gridspec(2, 2, wspace=0.07, hspace=0.36)
    panels = (("n_fn", "Missed findings, %"), ("n_fp", "False alarms, %"))
    for q, (persona, title) in enumerate(PERSONAS):
        pair = quarters[q // 2, q % 2].subgridspec(1, 2, wspace=0.06)
        for s, (stat, what) in enumerate(panels):
            ax = fig.add_subplot(pair[0, s])
            for k, f in enumerate(findings):
                y0 = n - 1 - k
                if k % 2 == 0:
                    ax.axhspan(y0 - 0.5, y0 + 0.5, color=BAND, zorder=0, lw=0)
                for a, (actor, _name, color, marker) in enumerate(actors):
                    got = rates.get((persona, actor, stat, f))
                    if not got:
                        continue
                    mean, lo, hi = _mean_ci(got)
                    y = y0 + (1.5 - a) * 0.2
                    if np.isfinite(lo) and hi > lo:
                        ax.plot([max(lo, 0.0), hi], [y, y], color=color, lw=0.8, zorder=2, solid_capstyle="butt")
                    ax.plot(mean, y, marker=marker, ms=3.0, mew=0.6, color=color, zorder=3, ls="none")
            # linear up to 1 %, logarithmic above: a rate of zero stays on the axis
            ax.set_xscale("symlog", linthresh=1, linscale=0.35)
            ax.set_xlim(0, 130)
            ax.set_xticks([0, 1, 10, 100])
            ax.set_xticklabels(["0", "1", "10", "100"])
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.set_ylim(-0.5, n - 0.5)
            ax.grid(axis="x", color=rule, lw=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=6, colors=ink_2, pad=1.5, length=2)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{title}\n{what}" if s == 0 else f"\n{what}", fontsize=6.8, color=ink, loc="left",
                         pad=3, linespacing=1.3)
            ax.set_yticks(range(n))
            if q % 2 == 0 and s == 0:
                # finding name, then its share of the patients, as in the RQ3 figure
                ax.set_yticklabels([f"{NAME.get(f, f)}  {share[f]:.1f} %" for f in reversed(findings)],
                                   fontsize=6.4, color=ink)
            else:
                ax.set_yticklabels([])
    h = fig.get_figheight()
    handles = [Line2D([], [], color=c, marker=m, ls="none", ms=3.6, label=lab) for _a, lab, c, m in actors]
    handles.append(Line2D([], [], color=ink_2, lw=0.8, label="95% interval over the 20 runs"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6.6, labelcolor=ink_2,
               handletextpad=0.3, columnspacing=1.4, borderaxespad=0.1)
    fig.supxlabel("rate of the error, in % (linear up to 1, logarithmic above; left is better)", fontsize=6.8,
                  color=ink_2, y=0.2 / h, va="bottom")
    fig.subplots_adjust(left=1.42 / 7.16, right=0.99, top=1 - 0.3 / h, bottom=0.55 / h)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def rq4(long_csv: Path, out: Path) -> None:
    render_team(long_csv, out / "rq4_team")
    print(f"\nfigure -> {out / 'rq4_team.pdf'}")
    rates, counts = _team_rates(long_csv)
    same = all(rates[(p, "doctor", s, f)] == rates[(p, "doctor@linear", s, f)]
               for p, _t in PERSONAS for s in ("n_fn", "n_fp") for f in COMMON)
    print(f"reader alone identical under the two rules: {same}")
    for persona, title in PERSONAS:
        print(f"\n{title}: rate in % (mean over runs) AI | reader | linear pool | log-odds")
        for stat in ("n_fn", "n_fp"):
            print(f"  {'missed findings' if stat == 'n_fn' else 'false alarms'}")
            for f in sorted(COMMON, key=lambda f: -counts[f][0]):
                v = [statistics.mean(rates[(persona, a, stat, f)])
                     for a in ("ai", "doctor", "linear_pool", "log_odds_fusion")]
                print(f"    {NAME.get(f, f):20s} n = {counts[f][0]:4d}  " + "  ".join(f"{x:6.2f}" for x in v))
    # team minus the same reader alone, summed over the findings, per run: counts of cases
    print("\nchange in the number of errors per run, all findings summed (mean, 95% interval)")
    rows = _read(long_csv)
    total: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        if r["group"].startswith("label=") and r["metric"] in (
                "actors.hybrid.n_fn", "actors.hybrid.n_fp", "actors.doctor.n_fn", "actors.doctor.n_fp"):
            sign = 1.0 if ".hybrid." in r["metric"] else -1.0
            total[(r["persona"], r["fusion.kind"], r["metric"][-4:])][r["seed"]] += sign * float(r["value"])
    for persona, title in PERSONAS:
        for fusion in ("linear_pool", "log_odds_fusion"):
            fn = _mean_ci(list(total[(persona, fusion, "n_fn")].values()))
            fp = _mean_ci(list(total[(persona, fusion, "n_fp")].values()))
            print(f"  {title:18s} {fusion:16s} missed {fn[0]:+8.1f} [{fn[1]:+.1f}, {fn[2]:+.1f}]  "
                  f"false alarms {fp[0]:+8.1f} [{fp[1]:+.1f}, {fp[2]:+.1f}]")
    print("\nfindings where the team beats the reader alone (mean over runs), per rule")
    for persona, title in PERSONAS:
        for fusion in ("linear_pool", "log_odds_fusion"):
            better = {s: [f for f in COMMON if statistics.mean(rates[(persona, fusion, s, f)])
                          < statistics.mean(rates[(persona, "doctor", s, f)])] for s in ("n_fn", "n_fp")}
            worse_fn = [NAME.get(f, f) for f in COMMON if f not in better["n_fn"]]
            print(f"  {title:18s} {fusion:16s} fewer misses on {len(better['n_fn'])}/10, fewer false alarms "
                  f"on {len(better['n_fp'])}/10; misses not reduced: {worse_fn}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue-summary", type=Path, required=True, help="*_summary.csv of the RQ3 study")
    ap.add_argument("--team-long", type=Path, required=True, help="*_long.csv of the RQ4 study")
    ap.add_argument("--out", type=Path, default=Path("out/paper"), help="folder of the figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config("configs/studies/capture.yaml")
    result = run_capture_study(cfg)
    capture_figures.PRETTY.update(NAME)  # the paper's names on the panels
    rq1(result, cfg.split, args.out)
    rq2(result, cfg.split, args.out)
    rq3(args.queue_summary, args.out)
    rq4(args.team_long, args.out)


if __name__ == "__main__":
    main()
