"""Figures, table and numbers of the ICSE-SEIS 2027 paper (RQ2 and RQ3), from the caches.

    uv run python scripts/paper_icse_seis.py \
        --queue-summary out/study_queue_load_medgemma27b_<stamp>_summary.csv \
        --out ../overleaf-paper/images

RQ2 reruns the capture study of configs/studies/capture.yaml (about a minute) and keeps the
ten findings annotated on both datasets: two figures (one per dataset, two rows of five
panels, no title), the skill table as LaTeX rows, how often each model is the best, and the
Wilcoxon signed-rank tests of MedGemma 27B against the other two models, paired by finding.
RQ3 draws the waiting-time figure of the MedGemma 27B queue study, in the layout of the
defense slides, and prints the mean waits quoted in the text. A wait is shown in mean reading
times: when the reader is never idle, that is the number of patients read before the waiting one.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

from hai_sim.analysis import capture_figures
from hai_sim.analysis.capture_study import load_config, run_capture_study
from hai_sim.analysis.study_figures import BAND, NEUTRAL

# The ten findings annotated on both datasets (Lung Opacity is VinDr-CXR only), most
# prevalent on VinDr-CXR first. VinDr-CXR annotates Nodule and Mass as one finding.
COMMON = ["Cardiomegaly", "Fibrosis", "Pleural_Thickening", "Effusion", "Nodule", "Mass",
          "Infiltration", "Consolidation", "Atelectasis", "Pneumothorax"]
NAME = {"Fibrosis": "Pulmonary fibrosis", "Pleural_Thickening": "Pleural thickening",
        "Effusion": "Pleural effusion"}
MODELS = ["ensemble", "medgemma-4b", "medgemma-27b"]
DATASETS = ["vindr", "nih"]
KEY = ("fn", "predictive_entropy", "calibrated")  # RQ2: own misses, calibrated entropy order
MEAN_READ = 13.2


def rq2(out: Path) -> None:
    cfg = load_config("configs/studies/capture.yaml")
    result = run_capture_study(cfg)
    capture_figures.PRETTY.update(NAME)  # the paper's names on the panels
    for ds in DATASETS:
        capture_figures.render_models(
            result["curves"], MODELS, ds, "fn", out / f"rq2_fn_{ds}", cfg.split, None,
            ordering=KEY[1], kind=KEY[2], pathologies=COMMON, cols=5, show_title=False,
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
    """Mean wait of the patients with each finding, per order and load (the defense layout)."""
    from matplotlib.lines import Line2D

    plt = capture_figures._plt()
    ink, ink_2, rule, surface = (capture_figures.INK, capture_figures.INK_2, capture_figures.RULE,
                                 capture_figures.SURFACE)
    rows = [r for r in csv.DictReader(summary.open(encoding="utf-8"))
            if r["metric"] == "system.mean_wait_to_doctor" and r["group"].endswith(":positive")]
    cell = {(r["patient_generator.arrival_rate"], r["sort.kind"], r["group"][6:-9]): r for r in rows}
    rates = sorted({r["patient_generator.arrival_rate"] for r in rows}, key=float)
    stem = "data/cache/composed/vindr__ens4"  # same ids and truth as every cell
    labels = json.loads(Path(f"{stem}.json").read_text(encoding="utf-8"))["pathologies"]
    gt = np.load(f"{stem}.npz", allow_pickle=True)["ground_truth"][1::2]  # the odd half the study reads
    share = {f: 100 * gt[:, labels.index(f)].mean() for f in COMMON}
    findings = sorted(COMMON, key=lambda f: -share[f])
    sorts = [("random", NEUTRAL, "o", "Random order"),
             ("uncertainty_desc", capture_figures.SERIES[1], "s", "Most uncertain first"),
             ("probability_desc", capture_figures.SERIES[0], "D", "Most suspicious first")]
    names = ["Half idle", "At capacity", "Overloaded"]

    fig, axes = plt.subplots(1, len(rates), figsize=(7.2, 0.25 * len(findings) + 1.25), sharey=True)
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
                ax.plot(v, y, marker=marker, ms=5.2, mew=1.2, color=color, mfc=color if clear else surface,
                        ls="none", zorder=3)
            ax.plot([min(values), max(values)], [y, y], color=rule, lw=1.4, zorder=1)
        rho = round(float(rate) * MEAN_READ, 1)
        ax.set_title(f"{names[j]} · $\\rho$ = {rho:g}", fontsize=9.5, color=ink, loc="left")
        ax.set_ylim(-0.5, len(findings) - 0.5)
        ax.set_xlim(left=0)
        ax.grid(axis="x", color=rule, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8.5, colors=ink_2)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    # finding name, then its share of the patients at the right edge: the numbers line up
    axes[0].set_yticks(range(len(findings)))
    axes[0].set_yticklabels([f"{NAME.get(f, f)}  {share[f]:.1f} %" for f in reversed(findings)],
                            fontsize=9, color=ink)
    h = fig.get_figheight()
    fig.text(0.5, 0.30 / h, "mean wait of the patients with the finding, in patients read before them: "
             "left is better\none unit = one mean reading time · hollow marker = no clear gain over "
             "the random order", ha="center", va="bottom", fontsize=8.5, color=ink_2, linespacing=1.5)
    handles = [Line2D([], [], color=c, marker=m, ls="none", ms=5.2, label=lab) for _k, c, m, lab in sorts]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=ink_2,
               handletextpad=0.4, columnspacing=2.4)
    fig.tight_layout(rect=(0, 0.72 / h, 1, 1), w_pad=1.2)
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue-summary", type=Path, required=True, help="*_summary.csv of the RQ3 study")
    ap.add_argument("--out", type=Path, default=Path("out/paper"), help="folder of the figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rq2(args.out)
    rq3(args.queue_summary, args.out)


if __name__ == "__main__":
    main()
