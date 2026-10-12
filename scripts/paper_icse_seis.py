"""Figures, table and numbers of the ICSE-SEIS 2027 paper (RQ2 and RQ3), from the caches.

    uv run python scripts/paper_icse_seis.py \
        --queue-summary out/study_queue_load_medgemma27b_<stamp>_summary.csv \
        --out ../overleaf-paper/images

RQ2 reruns the capture study of configs/studies/capture.yaml (about a minute) and keeps the
ten findings annotated on both datasets: two figures (one per dataset, two rows of five
panels, no title), the skill table as LaTeX rows, how often each model is the best, and the
Wilcoxon signed-rank tests of MedGemma 27B against the other two models, paired by finding.
RQ3 draws the waiting-time figure of the MedGemma 27B queue study and prints the ranges of
the mean waits quoted in the text, in reads (one read = the mean reading time).
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import friedmanchisquare, wilcoxon

from hai_sim.analysis import capture_figures
from hai_sim.analysis.capture_study import load_config, run_capture_study
from hai_sim.analysis.study_figures import render_queue_load

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


def rq3(summary: Path, out: Path) -> None:
    path = render_queue_load(summary, out / "rq3_wait", mean_read=MEAN_READ, show_title=False)
    print(f"\nfigure -> {path}.pdf")
    rows = [r for r in csv.DictReader(summary.open(encoding="utf-8"))
            if r["metric"] == "system.mean_wait_to_doctor" and r["group"].endswith(":positive")]
    wait: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        finding = r["group"][len("truth="):-len(":positive")]
        wait[(r["patient_generator.arrival_rate"], r["sort.kind"])][finding] = float(r["mean"]) / MEAN_READ
    for rate in sorted({k[0] for k in wait}, key=float):
        print(f"utilisation {float(rate) * MEAN_READ:.1f}")
        for order in ("random", "uncertainty_desc", "probability_desc"):
            w = wait[(rate, order)]
            lo, hi = min(w, key=w.get), max(w, key=w.get)
            print(f"  {order:17s} {w[lo]:9.1f} ({lo}) .. {w[hi]:9.1f} ({hi}) reads")
        print("    per finding, probability_desc: "
              + ", ".join(f"{f} {v:.1f}" for f, v in sorted(wait[(rate, 'probability_desc')].items(),
                                                            key=lambda kv: kv[1])))


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
