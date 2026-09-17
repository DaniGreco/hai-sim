"""Run the RQ1 uncertainty study on the on-disk ensemble cells (no model calls).

Composes the 4-member DenseNet ensemble for each dataset from its per-member cache cells,
then renders the figure set and the master tables described in
``literature/STUDY_uncertainty_sweep.md``.

    uv run python scripts/run_uncertainty_study.py
    uv run python scripts/run_uncertainty_study.py --out-dir out/analysis/uncertainty_study

Output (default ``out/analysis/uncertainty_study/``):
    <dataset>/measure_overlay.png       capture curves, measures overlaid, any/FN/FP
    <dataset>/threshold_policies.png    FN/FP/cost per operating point (held-out 50/50)
    in_vs_ood_mutual_information.png     in-domain vs OOD on epistemic uncertainty
    master_curves.csv · master_policies.csv · report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hai_sim.analysis.uncertainty_study import COST_FN, COST_FP, run_study

# The cross-domain ensemble members (the NIH weight set is dropped — it would leak).
MEMBERS = ("chex", "mimic_ch", "pc", "rsna")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RQ1 uncertainty study over the ensemble cells")
    p.add_argument("--cache-root", default="data/cache", help="where the <dataset>__<member> cells live")
    p.add_argument("--datasets", nargs="+", default=["nih", "vindr"])
    p.add_argument("--out-dir", default="out/analysis/uncertainty_study")
    p.add_argument("--c-fp", type=float, default=COST_FP)
    p.add_argument("--c-fn", type=float, default=COST_FN)
    p.add_argument(
        "--threshold-policy", default="cost_optimal",
        choices=["fixed", "prevalence", "cost_optimal"],
        help="decision threshold used for the capture-curve figures",
    )
    p.add_argument(
        "--difficulty-column", default="mean_delta_active",
        help="column of data/<dataset>/difficulty.csv used as the human difficulty",
    )
    args = p.parse_args(argv)

    root = Path(args.cache_root)
    cells: dict[str, list[str]] = {}
    difficulty_tables: dict[str, tuple[str, str]] = {}
    for ds in args.datasets:
        stems = [str(root / f"{ds}__{m}") for m in MEMBERS]
        missing = [s for s in stems if not Path(s + ".npz").exists()]
        if missing:
            print(f"[skip] {ds}: missing member cells {missing}")
            continue
        cells[ds] = stems
        table = Path("data") / ds / "difficulty.csv"
        if table.exists():
            difficulty_tables[ds] = (str(table), args.difficulty_column)
    if not cells:
        print("[error] no dataset had all member cells present")
        return 2

    report = run_study(
        cells, args.out_dir,
        threshold_policy=args.threshold_policy, c_fp=args.c_fp, c_fn=args.c_fn,
        difficulty_tables=difficulty_tables or None,
    )
    print(f"\nstudy rendered -> {args.out_dir}/")
    for ds, info in report["datasets"].items():
        best = min(info["policies"].items(), key=lambda kv: kv[1]["cost_per_case"])
        print(f"  [{ds}] N={info['n_images']}  best operating point: {best[0]} "
              f"(cost/case {best[1]['cost_per_case']:.2f}, "
              f"FN {best[1]['fn_rate']:.1%} · FP {best[1]['fp_rate']:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
