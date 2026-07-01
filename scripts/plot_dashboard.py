"""Render the parametric dashboard from an existing inference cache.

Generate the figure set (``error_curves`` + ``per_pathology_{any,fn,fp}``) on demand,
without re-running the model, choosing both aggregation axes explicitly:

- ``--measure``    = **member→label** uncertainty (how the 5 ensemble members collapse
  to one uncertainty per label): predictive_entropy / mean_entropy / mutual_information
  / std / least_confidence. The decision always uses the members' mean probability.
- ``--aggregator`` = **label→image** collapse for the per-image triage row (mean / max).
- ``--threshold``  = decision cut on the mean probability.

The default output dir encodes all three so nothing clobbers (mean vs max, or two
thresholds, land in sibling dirs):

    uv run python scripts/plot_dashboard.py --cache data/cache/nih_ensemble \
        --measure mean_entropy --aggregator max --threshold 0.5
    # -> out/analysis/mean_entropy/max_thr0.5/{error_curves,per_pathology_*}.png
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
    p.add_argument("--measure", default="predictive_entropy", choices=MEASURES)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--aggregator", default="mean", choices=["mean", "max"])
    p.add_argument("--out-dir", default=None, help="default: out/analysis/<measure>")
    args = p.parse_args(argv)

    cache = InferenceCache.load(args.cache)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("out/analysis") / args.measure / f"{args.aggregator}_thr{args.threshold:g}"
    render_dashboard(
        cache,
        out_dir,
        measure=args.measure,
        threshold=args.threshold,
        aggregator=args.aggregator,
    )
    print(
        f"dashboard rendered  N={cache.n_images}  ->  {out_dir}/\n"
        f"  member→label uncertainty : {args.measure}  (decision = mean-prob ≥ {args.threshold})\n"
        f"  label→image aggregation  : {args.aggregator}  (per-image row only)\n"
        f"  files: error_curves.png, per_pathology_any/fn/fp.png"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
