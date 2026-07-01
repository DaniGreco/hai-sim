"""Per-image triage aggregator comparison: ``mean`` vs ``max`` of the label vector.

The per-image triage (``UncertaintyDescSort``) collapses the 18-label uncertainty
vector to a single scalar per image before ordering the worklist. Two collapses are
offered: the ``mean`` (an image is uncertain if it is uncertain *on average*) and the
``max`` (an image is uncertain if it is uncertain on its *single most uncertain*
label). This asks which one produces the better error-capture ordering, per error
type (any / FN / FP), for one or more UQ measures — entirely from the cache.

    uv run python scripts/compare_aggregator.py --cache data/cache/nih_ensemble

Nothing is re-inferred; only the per-image row of the dashboard is affected by the
aggregator (the per-entry row keeps every (image, label) separate).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hai_sim.analysis.error_curve import _units_per_image, compute_curve
from hai_sim.analysis.inference_cache import InferenceCache

MEASURES = ["predictive_entropy", "mutual_information", "mean_entropy", "least_confidence", "std"]
ERROR_TYPES = ("any", "fn", "fp")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare mean vs max per-image aggregation")
    p.add_argument("--cache", default="data/cache/nih_ensemble", help="cache stem (no extension)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--measures",
        nargs="+",
        default=["predictive_entropy", "mutual_information"],
        choices=MEASURES,
    )
    p.add_argument("--out", default="out/analysis/experiments/aggregator.json")
    args = p.parse_args(argv)

    cache = InferenceCache.load(args.cache)
    report: dict[str, Any] = {
        "cache": args.cache,
        "n_images": cache.n_images,
        "threshold": args.threshold,
        "view": "per-image (multi-label triage)",
        "measures": {},
    }

    print(f"\nAggregator mean vs max  |  N={cache.n_images}  threshold={args.threshold}")
    print("(per-image triage view; higher AUROC / skill = better error ordering)\n")

    for measure in args.measures:
        print(f"== measure = {measure} ==")
        header = f"  {'error':<6} | {'AUROC mean':>11} {'AUROC max':>10} {'Δ':>7} | {'skill mean':>11} {'skill max':>10} | {'E/N':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        m_report: dict[str, dict[str, Any]] = {}
        for agg in ("mean", "max"):
            unc, errs = _units_per_image(cache, measure, args.threshold, agg)
            for et in ERROR_TYPES:
                cr = compute_curve(
                    unc,
                    errs[et],
                    error_type=et,
                    mode="image",
                    measure=measure,
                    threshold=args.threshold,
                    aggregator=agg,
                )
                m_report.setdefault(et, {})[agg] = {
                    "auroc": cr.auroc,
                    "skill": cr.skill,
                    "n_errors": cr.n_errors,
                    "n_units": cr.n_units,
                }
        for et in ERROR_TYPES:
            a, x = m_report[et]["mean"], m_report[et]["max"]
            d = a["auroc"] - x["auroc"]
            print(
                f"  {et:<6} | {a['auroc']:>11.3f} {x['auroc']:>10.3f} {d:>+7.3f} | "
                f"{a['skill']:>11.3f} {x['skill']:>10.3f} | {a['n_errors']:>4}/{a['n_units']:<5}"
            )
        report["measures"][measure] = m_report
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
