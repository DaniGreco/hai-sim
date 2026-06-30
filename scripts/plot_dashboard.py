"""Render the parametric dashboard from an existing inference cache.

Generate the figure set (``error_curves`` + ``per_pathology_{any,fn,fp}``) for any
UQ measure, on demand, without re-running the model. Defaults the output to a
per-measure subdir so different measures don't clobber each other:

    uv run python scripts/plot_dashboard.py --cache data/cache/nih_ensemble \
        --measure mean_entropy
    # -> out/analysis/mean_entropy/{error_curves,per_pathology_any,_fn,_fp}.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hai_sim.analysis.error_curve import render_dashboard
from hai_sim.analysis.inference_cache import InferenceCache

MEASURES = [
    "mutual_information",
    "predictive_entropy",
    "mean_entropy",
    "std",
    "least_confidence",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render the dashboard for a chosen UQ measure")
    p.add_argument("--cache", required=True, help="cache stem (no extension), e.g. data/cache/nih_ensemble")
    p.add_argument("--measure", default="predictive_entropy", choices=MEASURES)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--aggregator", default="mean", choices=["mean", "max"])
    p.add_argument("--out-dir", default=None, help="default: out/analysis/<measure>")
    args = p.parse_args(argv)

    cache = InferenceCache.load(args.cache)
    out_dir = Path(args.out_dir) if args.out_dir else Path("out/analysis") / args.measure
    render_dashboard(
        cache,
        out_dir,
        measure=args.measure,
        threshold=args.threshold,
        aggregator=args.aggregator,
        title_extra=f"[N={cache.n_images}]",
    )
    print(
        f"dashboard ({args.measure}, threshold={args.threshold}, N={cache.n_images}) "
        f"-> {out_dir}/  (error_curves, per_pathology_any/fn/fp)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
