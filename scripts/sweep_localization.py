"""Sweep the IoU/Dice binarization threshold and emit a comparable index + curves.

Demonstrates the parametric-sweep pattern (like scikit-learn's ParameterGrid +
cv_results_): reconstruct each finding's Grad-CAM **once**, then re-score it at
every threshold in the grid (thresholding a heatmap is cheap, so a whole sweep
costs one Grad-CAM pass). Produces:

    out/analysis/localization/<family>/thr<X>/<agg>/sweep_iou/
    |-- iou_sweep.png   IoU-vs-threshold curve per pathology (+ pooled)
    `-- sweep.json      the comparable index: {pathology -> {threshold -> IoU, Dice, n}}

Run (from repo root):
  uv run python scripts/sweep_localization.py --cache data/cache/chestxdet_localization \
    --thresholds 0.1,0.3,0.5,0.7,0.9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hai_sim.analysis.localization.cache import LocalizationCache
from hai_sim.analysis.localization.evaluate import per_instance_localization
from hai_sim.analysis.localization.metrics import Dice, EnergyInMask, IoU, PointingGame


def _mean(recs: list, key: str) -> float:
    v = np.array([r["scores"][key] for r in recs], dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description="sweep the IoU/Dice threshold from a cache")
    p.add_argument("--cache", required=True)
    p.add_argument("--family", default="ensemble")
    p.add_argument("--thresholds", default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--threshold", type=float, default=0.5, help="decision threshold (path)")
    p.add_argument("--pointing-tolerance", type=int, default=16)
    p.add_argument("--aggregation", choices=["mean", "max"], default="mean")
    p.add_argument("--split", default=None)
    p.add_argument("--min-instances", type=int, default=15)
    p.add_argument("--top", type=int, default=6, help="pathologies to draw (by n); all in json")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    cache = LocalizationCache.load(args.cache)
    tvals = [float(t) for t in args.thresholds.split(",")]
    tol = round(args.pointing_tolerance * cache.mask_resolution / 224)
    # One Grad-CAM pass scores every threshold: pass all IoU(t)/Dice(t) at once.
    metrics = (
        [PointingGame(tolerance=tol), EnergyInMask()]
        + [IoU(t) for t in tvals]
        + [Dice(t) for t in tvals]
    )
    _pred, records = per_instance_localization(
        cache, metrics=metrics, threshold=args.threshold,
        member_aggregation=args.aggregation, split=args.split,
    )
    if not records:
        print("no localized instances in the cache yet")
        return 1

    labels = sorted({r["label_name"] for r in records})
    index: dict[str, dict] = {}
    for name in labels:
        recs = [r for r in records if r["label_name"] == name]
        if len(recs) < args.min_instances:
            continue
        per_t = {
            f"{t:g}": {"iou": _mean(recs, IoU(t).name), "dice": _mean(recs, Dice(t).name)}
            for t in tvals
        }
        best = max(tvals, key=lambda t: (per_t[f"{t:g}"]["iou"] if np.isfinite(per_t[f"{t:g}"]["iou"]) else -1))
        index[name] = {"n": len(recs), "by_threshold": per_t, "best_iou_threshold": best}

    # pooled IoU at each threshold (all findings)
    pooled = {f"{t:g}": _mean(records, IoU(t).name) for t in tvals}

    base = (
        Path(args.out_dir)
        if args.out_dir
        else Path("out/analysis/localization") / args.family / f"thr{args.threshold:g}"
        / args.aggregation / "sweep_iou"
    )
    base.mkdir(parents=True, exist_ok=True)

    # ---- figure: IoU vs threshold, per pathology + pooled -----------------
    ranked = sorted(index.items(), key=lambda kv: kv[1]["n"], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, (name, d) in enumerate(ranked[: args.top]):
        ys = [d["by_threshold"][f"{t:g}"]["iou"] for t in tvals]
        ax.plot(tvals, ys, "o-", color=cmap(i % 10), label=f"{name} (n={d['n']})")
    ax.plot(tvals, [pooled[f"{t:g}"] for t in tvals], "k--", lw=2.5, label="pooled")
    ax.set_xlabel("IoU binarization threshold (fraction of heatmap max)")
    ax.set_ylabel("mean IoU vs expert mask")
    ax.set_title(f"IoU-threshold sweep — {cache.dataset} ({len(records)} findings, "
                 f"{cache.n_images} imgs)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(base / "iou_sweep.png", dpi=110)
    plt.close(fig)

    (base / "sweep.json").write_text(
        json.dumps({"thresholds": tvals, "n_findings": len(records), "n_images": cache.n_images,
                    "pooled_iou": pooled, "per_pathology": index}, indent=2),
        encoding="utf-8",
    )

    hdr = "  ".join(f"iou@{t:g}" for t in tvals)
    print(f"findings={len(records)}  imgs={cache.n_images}   pathology            {hdr}   best")
    for name, d in ranked:
        cells = "  ".join(f"{d['by_threshold'][f'{t:g}']['iou']:.3f} " for t in tvals)
        print(f"  {name:18s} n={d['n']:4d}  {cells}  @{d['best_iou_threshold']:g}")
    print(f"  {'POOLED':18s}          " + "  ".join(f"{pooled[f'{t:g}']:.3f} " for t in tvals))
    print(f"sweep -> {base}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
