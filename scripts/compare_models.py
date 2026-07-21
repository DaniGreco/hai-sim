#!/usr/bin/env python
"""Compare several AI readers on exactly the same images.

Reads two or more inference caches, restricts them to the images and pathologies they
share, and renders four figures that a single accuracy number cannot separate:
discrimination, calibration, error overlap, and whether uncertainty finds the errors.

Any backend that produced a cache can take part — a convolutional ensemble and a
generative vision-language model sit in the same figure, which is the point.

Examples
--------
Single member against the ensemble it belongs to (works with one cache on disk)::

    python scripts/compare_models.py \\
        --cache "data/cache/nih_ensemble#0" --name "DenseNet (NIH weights)" \\
        --cache "data/cache/nih_ensemble"   --name "Deep Ensemble (5 members)"

A convolutional ensemble against MedGemma, once its cache exists::

    python scripts/compare_models.py \\
        --cache data/cache/nih_ensemble  --name "Deep Ensemble" \\
        --cache data/cache/medgemma_nih  --name "MedGemma 1.5 4B"

Append ``#i`` (or ``#i,j``) to a stem to use only those ensemble members.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hai_sim.analysis.model_comparison import align_runs, all_overlaps, load_runs, render_all


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--cache", action="append", required=True, metavar="STEM[#MEMBERS]",
        help="cache stem, optionally #-suffixed with member indices; repeat per model",
    )
    p.add_argument(
        "--name", action="append", default=None,
        help="display name for the matching --cache (default: the file name)",
    )
    p.add_argument("--out-dir", default="out/analysis/model_comparison")
    p.add_argument(
        "--measure", default="predictive_entropy",
        choices=["predictive_entropy", "mean_entropy", "mutual_information", "std",
                 "least_confidence"],
        help="which uncertainty the error-capture figure ranks by",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="probability above which a finding counts as called present",
    )
    args = p.parse_args(argv)

    if args.name is not None and len(args.name) != len(args.cache):
        print(f"[error] {len(args.name)} --name values for {len(args.cache)} --cache values")
        return 2
    if len(args.cache) < 2:
        print("[warn] only one model given: the overlap figure needs at least two")

    try:
        runs = load_runs(args.cache, args.name, args.measure)
        aligned = align_runs(runs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}")
        return 2

    print(
        f"comparing {len(aligned)} models over {len(aligned[0].image_ids)} shared images "
        f"x {len(aligned[0].pathologies)} shared pathologies "
        f"(uncertainty={args.measure}, threshold={args.threshold})"
    )
    for run in runs:
        dropped = len(run.image_ids) - len(aligned[0].image_ids)
        note = f"  ({dropped} images not shared)" if dropped else ""
        print(f"  - {run.name}: {len(run.image_ids)} images{note}")

    for overlap in all_overlaps(aligned, args.threshold):
        kappa = "n/a" if overlap.kappa is None else f"{overlap.kappa:+.3f}"
        print(
            f"  overlap {overlap.a} vs {overlap.b}: both wrong {overlap.both}, "
            f"only {overlap.a} {overlap.only_a}, only {overlap.b} {overlap.only_b}, "
            f"kappa {kappa}"
        )

    written = render_all(aligned, Path(args.out_dir), args.threshold)
    for path in written:
        print(f"  figure -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
