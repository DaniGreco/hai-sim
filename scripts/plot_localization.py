"""Per-pathology localization figures + localization-vs-uncertainty analysis.

Runs entirely offline from a `LocalizationCache`: reconstructs each finding's
Grad-CAM, scores it against the expert mask (per_instance_localization), and joins
the per-instance uncertainty derived from the cached per-member probabilities
(measures_from_cache) — so it answers both "how well does the model localize each
disease?" and "does it localize worse when it is more uncertain?".

Run (from repo root):
  uv run python scripts/plot_localization.py --cache data/cache/chestxdet_localization \
    --out-dir out/analysis/localization/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hai_sim.analysis.inference_cache import measures_from_cache
from hai_sim.analysis.localization.cache import LocalizationCache
from hai_sim.analysis.localization.evaluate import per_instance_localization, report_from_instances
from hai_sim.analysis.localization.metrics import Dice, EnergyInMask, IoU, PointingGame

UNC_LABEL = {"predictive_entropy": "predictive entropy (total)", "mutual_information": "mutual info (epistemic)"}


def _mean_metric(recs: list, metric: str) -> float:
    vals = np.array([r["scores"][metric] for r in recs], dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 5:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="localization + uncertainty figures from a cache")
    parser.add_argument("--cache", required=True, help="cache stem")
    parser.add_argument("--family", default="ensemble", help="UQ source folder (top partition)")
    parser.add_argument(
        "--out-dir", default=None,
        help="override base; default out/analysis/localization/<family>/thr<X>/<agg>",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pointing-tolerance", type=int, default=16, help="224px units")
    parser.add_argument("--aggregation", choices=["mean", "max"], default="mean")
    parser.add_argument("--split", default=None)
    parser.add_argument("--min-instances", type=int, default=15)
    parser.add_argument("--uncertainty", default="predictive_entropy",
                        choices=["predictive_entropy", "mutual_information"])
    args = parser.parse_args()

    cache = LocalizationCache.load(args.cache)
    tol = round(args.pointing_tolerance * cache.mask_resolution / 224)
    metrics = [PointingGame(tolerance=tol), EnergyInMask(), IoU(0.5), Dice(0.5)]
    predicted, records = per_instance_localization(
        cache, metrics=metrics, threshold=args.threshold,
        member_aggregation=args.aggregation, split=args.split,
    )
    if not records:
        print("no localized instances in the cache yet")
        return 1

    # Output path encodes the parameters, mirroring scripts/plot_dashboard.py:
    #   out/analysis/localization/<family>/thr<X>/<agg>/   -> per_pathology + report.json
    #   .../vs_uncertainty/<measure>/                      -> correlation (depends on measure too)
    base = (
        Path(args.out_dir)
        if args.out_dir
        else Path("out/analysis/localization") / args.family / f"thr{args.threshold:g}" / args.aggregation
    )
    base.mkdir(parents=True, exist_ok=True)
    report = report_from_instances(
        cache, predicted, records, metric_names=[m.name for m in metrics],
        split=args.split, member_aggregation=args.aggregation, threshold=args.threshold,
        model_name="+".join(cache.weights),
    )
    (base / "report.json").write_text(
        json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
    )

    unc = measures_from_cache(cache.member_probs)[args.uncertainty]  # (N, L)
    for r in records:
        r["uncertainty"] = float(unc[r["image_row"], r["label"]])

    pg = f"pointing_game@{tol}px" if tol else "pointing_game"
    labels = sorted({r["label_name"] for r in records})
    unc_dir = base / "vs_uncertainty" / args.uncertainty
    unc_dir.mkdir(parents=True, exist_ok=True)

    # ---- per-pathology aggregates -----------------------------------------
    rows = []
    for name in labels:
        recs = [r for r in records if r["label_name"] == name]
        if len(recs) < args.min_instances:
            continue
        u = np.array([r["uncertainty"] for r in recs])
        e = np.array([r["scores"]["energy_in_mask"] for r in recs])
        rows.append({
            "pathology": name, "n": len(recs),
            "hit_rate": _mean_metric(recs, pg), "energy_in_mask": _mean_metric(recs, "energy_in_mask"),
            "iou": _mean_metric(recs, "iou@0.5"), "dice": _mean_metric(recs, "dice@0.5"),
            "mean_uncertainty": float(np.nanmean(u)),
            "spearman_unc_energy": _spearman(u, e),
        })
    rows.sort(key=lambda d: (d["hit_rate"] if np.isfinite(d["hit_rate"]) else -1), reverse=True)

    # ---- Figure 1: per-pathology localization -----------------------------
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(rows) + 1)))
    y = np.arange(len(rows))
    hgt = 0.26
    ax.barh(y + hgt, [d["hit_rate"] for d in rows], hgt, label=pg, color="#2b7bba")
    ax.barh(y, [d["energy_in_mask"] for d in rows], hgt, label="energy_in_mask", color="#f18f01")
    ax.barh(y - hgt, [d["iou"] for d in rows], hgt, label="iou@0.5", color="#8aab3c")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d['pathology']} (n={d['n']})" for d in rows])
    ax.invert_yaxis()
    ax.set_xlabel("score (higher = better spatial agreement with expert mask)")
    ax.set_xlim(0, 1)
    ax.set_title(f"Localization vs expert masks — {cache.dataset} ({len(records)} findings, "
                 f"{cache.n_images} imgs)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(base / "per_pathology.png", dpi=110)
    plt.close(fig)

    # ---- Figure 2: localization vs uncertainty ----------------------------
    fig, (axb, axs) = plt.subplots(1, 2, figsize=(13, 5))
    all_u = np.array([r["uncertainty"] for r in records])
    all_e = np.array([r["scores"]["energy_in_mask"] for r in records])
    all_h = np.array([1.0 if r["scores"][pg] >= 0.5 else 0.0 for r in records])
    # pooled binned curve
    edges = np.quantile(all_u, np.linspace(0, 1, 7))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(all_u, edges) - 1, 0, len(edges) - 2)
    bx, be, bh = [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() < 5:
            continue
        bx.append(float(all_u[sel].mean()))
        be.append(float(np.nanmean(all_e[sel])))
        bh.append(float(all_h[sel].mean()))
    axb.plot(bx, be, "o-", color="#f18f01", label="energy_in_mask")
    axb.plot(bx, bh, "s-", color="#2b7bba", label="hit-rate")
    axb.set_xlabel(UNC_LABEL[args.uncertainty])
    axb.set_ylabel("mean localization score")
    axb.set_title(f"Localization vs uncertainty (pooled)\nSpearman rho(unc, energy) = "
                  f"{_spearman(all_u, all_e):+.2f}")
    axb.legend(fontsize=8)
    axb.grid(alpha=0.3)
    # per-pathology spearman
    sr = sorted([d for d in rows if np.isfinite(d["spearman_unc_energy"])],
                key=lambda d: d["spearman_unc_energy"])
    yy = np.arange(len(sr))
    colors = ["#c0392b" if d["spearman_unc_energy"] < 0 else "#2e7d32" for d in sr]
    axs.barh(yy, [d["spearman_unc_energy"] for d in sr], color=colors)
    axs.set_yticks(yy)
    axs.set_yticklabels([f"{d['pathology']} (n={d['n']})" for d in sr], fontsize=8)
    axs.axvline(0, color="k", lw=0.8)
    axs.set_xlabel("Spearman rho(uncertainty, energy_in_mask)")
    axs.set_title("Per-pathology correlation\n(negative = localizes worse when uncertain)")
    axs.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(unc_dir / "correlation.png", dpi=110)
    plt.close(fig)

    (unc_dir / "summary.json").write_text(
        json.dumps({"n_findings": len(records), "n_images": cache.n_images,
                    "uncertainty": args.uncertainty,
                    "pooled_spearman_unc_energy": _spearman(all_u, all_e),
                    "per_pathology": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"findings={len(records)} imgs={cache.n_images}  "
          f"pooled Spearman(unc,energy)={_spearman(all_u, all_e):+.3f}")
    for d in rows:
        print(f"  {d['pathology']:18s} n={d['n']:4d}  hit={d['hit_rate']:.2f}  "
              f"energy={d['energy_in_mask']:.3f}  iou={d['iou']:.3f}  "
              f"rho(unc,energy)={d['spearman_unc_energy']:+.2f}")
    print(f"report -> {base}/report.json ; figures -> {base}/ (+ vs_uncertainty/{args.uncertainty}/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
