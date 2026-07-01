"""Calibrate per-pathology decision thresholds from an inference cache.

Fits two operating points per pathology on a calibration split and reports the
held-out test metrics, plus a per-pathology ROC figure:

    uv run python scripts/calibrate.py --cache data/cache/nih_ensemble \
        --target-recall 0.95 --w-fn 1000000 --w-fp 500
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from hai_sim.analysis.calibration import PathologyCalibration, calibrate, render_calibration
from hai_sim.analysis.inference_cache import InferenceCache


def _totals(results: list[PathologyCalibration]) -> dict[str, float]:
    pos = sum(r.n_pos for r in results)
    tp05 = sum(r.recall_05 * r.n_pos for r in results)
    tprt = sum(r.recall_rt * r.n_pos for r in results)
    tpc = sum(r.recall_cost * r.n_pos for r in results)
    return {
        "pos": pos,
        "recall_05": tp05 / pos if pos else 0.0,
        "fp_05": sum(r.fp_05 for r in results),
        "recall_rt": tprt / pos if pos else 0.0,
        "fp_rt": sum(r.fp_rt for r in results),
        "recall_cost": tpc / pos if pos else 0.0,
        "fp_cost": sum(r.fp_cost for r in results),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Per-pathology threshold calibration from a cache")
    p.add_argument("--cache", required=True, help="cache stem, e.g. data/cache/nih_ensemble")
    p.add_argument("--target-recall", type=float, default=0.95)
    p.add_argument("--w-fn", type=float, default=1_000_000.0)
    p.add_argument("--w-fp", type=float, default=500.0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--test-frac", type=float, default=0.5)
    p.add_argument("--out-dir", default="out/analysis/calibration")
    args = p.parse_args(argv)

    cache = InferenceCache.load(args.cache)
    results = calibrate(
        cache,
        split_seed=args.split_seed,
        test_frac=args.test_frac,
        target_recall=args.target_recall,
        w_fn=args.w_fn,
        w_fp=args.w_fp,
    )

    print(
        f"N={cache.n_images} (test split {args.test_frac:.0%})  "
        f"target_recall={args.target_recall}  w_fn={args.w_fn:g} w_fp={args.w_fp:g}\n"
    )
    hdr = (
        f"{'pathology':<19}{'AUC':>5}{'r@.5':>6}{'FP@.5':>7} | "
        f"{'tauR':>6}{'rec':>5}{'FP':>6}{'prec':>6} | "
        f"{'tauC':>6}{'rec':>5}{'FP':>6}{'prec':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda x: -x.auc):
        print(
            f"{r.name:<19}{r.auc:>5.2f}{r.recall_05:>6.2f}{r.fp_05:>7} | "
            f"{r.tau_recall:>6.3f}{r.recall_rt:>5.2f}{r.fp_rt:>6}{r.precision_rt:>6.2f} | "
            f"{r.tau_cost:>6.3f}{r.recall_cost:>5.2f}{r.fp_cost:>6}{r.precision_cost:>6.2f}"
        )
    t = _totals(results)
    print(
        f"\nTOTAL (test): pos={t['pos']:.0f}  |  @0.5 recall={t['recall_05']:.2f} FP={t['fp_05']:.0f}"
        f"  |  recall-target recall={t['recall_rt']:.2f} FP={t['fp_rt']:.0f}"
        f"  |  cost recall={t['recall_cost']:.2f} FP={t['fp_cost']:.0f}"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_calibration(
        cache, out_dir / "roc_per_pathology.png", results,
        split_seed=args.split_seed, test_frac=args.test_frac,
    )
    (out_dir / "calibration.json").write_text(
        json.dumps(
            {
                "n_images": cache.n_images,
                "target_recall": args.target_recall,
                "w_fn": args.w_fn,
                "w_fp": args.w_fp,
                "split_seed": args.split_seed,
                "test_frac": args.test_frac,
                "totals": t,
                "pathologies": [asdict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nROC figure -> {out_dir / 'roc_per_pathology.png'}\nreport    -> {out_dir / 'calibration.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
