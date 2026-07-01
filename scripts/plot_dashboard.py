"""Render the parametric dashboard from an existing inference cache.

Generate the figure set on demand, without re-running the model. The output tree
mirrors what each figure actually depends on:

    out/analysis/<family>/<measure>/thr<X>/
    |-- mean/error_curves.png        (+ .json)   depends on measure . threshold . aggregator
    |-- max/error_curves.png         (+ .json)
    |-- per_pathology_any.png        (+ .json)   depends on measure . threshold only
    |-- per_pathology_fn.png                     (kept once, parallel to mean/ and max/)
    `-- per_pathology_fp.png

- ``--family``     = UQ source, the top partition (``ensemble`` now; ``mc_dropout`` /
  ``tta`` later live in sibling folders with their own caches).
- ``--measure``    = **member->label** uncertainty: predictive_entropy / mean_entropy /
  mutual_information / std / least_confidence. The decision uses the members' mean prob.
- ``--aggregator`` = **label->image** grouping criterion for the per-image error_curves
  (mean / max). It does NOT affect the per-pathology figures (each label kept separate),
  which is why those sit outside the ``mean/`` and ``max/`` subdirs.
- ``--threshold``  = decision cut on the mean probability.

    uv run python scripts/plot_dashboard.py --cache data/cache/nih_ensemble \
        --measure mean_entropy --aggregator max --threshold 0.5
    # -> out/analysis/ensemble/mean_entropy/thr0.5/max/error_curves.png
    #    out/analysis/ensemble/mean_entropy/thr0.5/per_pathology_{any,fn,fp}.png
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
    p.add_argument(
        "--cache", required=True, help="cache stem (no extension), e.g. data/cache/nih_ensemble"
    )
    p.add_argument(
        "--family", default="ensemble", help="UQ source folder (ensemble / mc_dropout / tta)"
    )
    p.add_argument("--measure", default="predictive_entropy", choices=MEASURES)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--aggregator", default="mean", choices=["mean", "max"])
    p.add_argument(
        "--out-dir",
        default=None,
        help="override base; default out/analysis/<family>/<measure>/thr<X>",
    )
    args = p.parse_args(argv)

    cache = InferenceCache.load(args.cache)
    if args.out_dir:
        base = Path(args.out_dir)
    else:
        base = Path("out/analysis") / args.family / args.measure / f"thr{args.threshold:g}"
    render_dashboard(
        cache,
        base,
        measure=args.measure,
        threshold=args.threshold,
        aggregator=args.aggregator,
    )
    print(
        f"dashboard rendered  N={cache.n_images}  ->  {base}/\n"
        f"  error_curves (grouping_criterion={args.aggregator})  -> {base / args.aggregator}/error_curves.png\n"
        f"  per_pathology_any/fn/fp (label-split, aggregator-independent) -> {base}/\n"
        f"  member->label uncertainty : {args.measure}  (decision = mean-prob >= {args.threshold})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
