"""Single pre-trained weight vs the cross-domain Deep Ensemble.

Two questions, both answered offline from the cached per-member probabilities:

**Table 1 - whole-system comparison.** Treat each configuration as a complete
classifier + its own uncertainty. A single weight uses one member's probability and
its single-pass entropy (== ``EntropyUQ``); the ensemble uses the nan-aware mean
probability and its ``predictive_entropy``. Reports decision quality (accuracy,
sensitivity, specificity) and failure-prediction AUROC / skill (any / FN / FP). The
fair single baselines are the members that cover **all 14 NIH-annotated labels**
(``nih`` and ``pc``); ``chex`` / ``mimic`` cover only a subset and are shown
separately on their own covered entries.

**Table 2 - UQ ranking power on a fixed error set.** Freeze the ensemble as the
classifier (same predictions, same errors) and only swap the *score* used to rank
those errors: a single member's entropy vs the ensemble's total uncertainty
(``predictive_entropy``) vs its epistemic disagreement (``mutual_information``),
etc. This isolates whether the ensemble's cross-domain spread adds failure-ranking
power over a single-pass entropy, independent of any accuracy gain.

    uv run python scripts/compare_single_vs_ensemble.py --cache data/cache/nih_ensemble
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hai_sim.analysis.error_curve import _accuracy_decomposition, _auroc, compute_curve
from hai_sim.analysis.inference_cache import InferenceCache, measures_from_cache

ERROR_TYPES = ("any", "fn", "fp")


def _subset_mean_and_scores(cache: InferenceCache, members: list[int]) -> dict[str, Any]:
    """UQ decomposition for a member subset (mean_p + measures, NaN where uncovered)."""
    measures: dict[str, Any] = measures_from_cache(cache.member_probs[:, members, :])
    return measures


def _entry_errors(pred: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "any": pred != gt,
        "fn": (pred == 0) & (gt == 1),
        "fp": (pred == 1) & (gt == 0),
    }


def evaluate_system(
    cache: InferenceCache, members: list[int], threshold: float, uq_measure: str
) -> dict[str, Any]:
    """Whole-system metrics for a member subset over the entries it covers."""
    m = _subset_mean_and_scores(cache, members)
    mean_p_raw = m["mean_p"]  # (N, L), NaN where subset covers no member for that label
    covered = ~np.isnan(mean_p_raw)
    mean_p = np.nan_to_num(mean_p_raw, nan=0.0)
    unc = np.nan_to_num(m[uq_measure], nan=0.0)
    pred = (mean_p >= threshold).astype(np.int8)
    gt = cache.ground_truth.copy()
    gt[~covered] = -1  # exclude labels this subset cannot classify

    # ---- per-entry (pooled over covered, annotated labels) ----
    valid = gt >= 0
    dec = _accuracy_decomposition(pred, gt)
    p_v, g_v, u_v = pred[valid], gt[valid], unc[valid]
    errs = _entry_errors(p_v, g_v)
    entry: dict[str, Any] = {
        "n_entries": int(valid.sum()),
        "acc": dec["acc"],
        "sens": dec["sens"],
        "spec": dec["spec"],
    }
    for et in ERROR_TYPES:
        cr = compute_curve(
            u_v, errs[et], error_type=et, mode="entry", measure=uq_measure, threshold=threshold
        )
        entry[f"auroc_{et}"] = cr.auroc
        entry[f"skill_{et}"] = cr.skill
        entry[f"n_{et}"] = cr.n_errors

    # ---- per-image (mean-aggregated triage view) ----
    unc_img = unc.mean(axis=1)
    img_err = {
        "any": ((pred != gt) & valid).any(axis=1),
        "fn": ((pred == 0) & (gt == 1)).any(axis=1),
        "fp": ((pred == 1) & (gt == 0)).any(axis=1),
    }
    image: dict[str, Any] = {}
    for et in ERROR_TYPES:
        image[f"auroc_{et}"] = _auroc(unc_img, img_err[et])
    return {"entry": entry, "image": image}


def uq_on_fixed_ensemble(cache: InferenceCache, threshold: float) -> dict[str, Any]:
    """Fix ensemble predictions/errors; compare UQ scores at ranking those errors."""
    all_members = list(range(cache.n_members))
    ens = _subset_mean_and_scores(cache, all_members)
    mean_p = np.nan_to_num(ens["mean_p"], nan=0.0)
    pred = (mean_p >= threshold).astype(np.int8)
    gt = cache.ground_truth
    valid = gt >= 0
    p_v, g_v = pred[valid], gt[valid]
    errs = _entry_errors(p_v, g_v)

    nih = _subset_mean_and_scores(cache, [0])  # single-pass NIH entropy
    scores = {
        "nih_single entropy": np.nan_to_num(nih["predictive_entropy"], nan=0.0)[valid],
        "ens predictive_entropy": np.nan_to_num(ens["predictive_entropy"], nan=0.0)[valid],
        "ens mutual_information": np.nan_to_num(ens["mutual_information"], nan=0.0)[valid],
        "ens mean_entropy": np.nan_to_num(ens["mean_entropy"], nan=0.0)[valid],
        "ens std": np.nan_to_num(ens["std"], nan=0.0)[valid],
    }
    out: dict[str, Any] = {}
    for name, s in scores.items():
        out[name] = {et: _auroc(s, errs[et]) for et in ERROR_TYPES}
    return {
        "n_entries": int(valid.sum()),
        "n_errors": {et: int(errs[et].sum()) for et in ERROR_TYPES},
        "scores": out,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Single weight vs cross-domain ensemble")
    p.add_argument("--cache", default="data/cache/nih_ensemble", help="cache stem (no extension)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default="out/analysis/experiments/single_vs_ensemble.json")
    args = p.parse_args(argv)

    cache = InferenceCache.load(args.cache)
    wnames = [w.split("-")[-1] for w in cache.weights]  # nih, chex, mimic_ch, pc, rsna
    systems: list[tuple[str, list[int]]] = [
        (f"{wnames[0]} (single)", [0]),
        (f"{wnames[3]} (single)", [3]),
        (f"{wnames[1]} (single, 8 lbl)", [1]),
        (f"{wnames[2]} (single, 8 lbl)", [2]),
        ("ENSEMBLE (5)", list(range(cache.n_members))),
    ]

    print(f"\nSingle weight vs ensemble  |  N={cache.n_images}  threshold={args.threshold}\n")
    print("TABLE 1 - whole system (own classifier + own UQ), per-entry pooled")
    hdr = (
        f"  {'system':<22} | {'acc':>5} {'sens':>5} {'spec':>5} | "
        f"{'AUROC any':>9} {'AUROC fn':>8} {'AUROC fp':>8} | {'entries':>7}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    report: dict[str, Any] = {
        "cache": args.cache,
        "n_images": cache.n_images,
        "threshold": args.threshold,
        "systems": {},
    }
    for name, members in systems:
        r = evaluate_system(cache, members, args.threshold, "predictive_entropy")
        e = r["entry"]
        print(
            f"  {name:<22} | {e['acc']:>5.2f} {e['sens']:>5.2f} {e['spec']:>5.2f} | "
            f"{e['auroc_any']:>9.3f} {e['auroc_fn']:>8.3f} {e['auroc_fp']:>8.3f} | {e['n_entries']:>7}"
        )
        report["systems"][name] = r

    print("\n  per-image (mean-agg triage view) AUROC:")
    print(f"  {'system':<22} | {'any':>6} {'fn':>6} {'fp':>6}")
    for name, _members in systems:
        img = report["systems"][name]["image"]
        print(
            f"  {name:<22} | {img['auroc_any']:>6.3f} {img['auroc_fn']:>6.3f} {img['auroc_fp']:>6.3f}"
        )

    fixed = uq_on_fixed_ensemble(cache, args.threshold)
    report["uq_on_fixed_ensemble"] = fixed
    print("\nTABLE 2 - UQ score ranking the SAME ensemble errors (per-entry AUROC)")
    ne = fixed["n_errors"]
    print(
        f"  (ensemble errors: any={ne['any']}  fn={ne['fn']}  fp={ne['fp']}  of {fixed['n_entries']} entries)"
    )
    print(f"  {'score':<26} | {'any':>6} {'fn':>6} {'fp':>6}")
    print("  " + "-" * 52)
    for name, d in fixed["scores"].items():
        print(f"  {name:<26} | {d['any']:>6.3f} {d['fn']:>6.3f} {d['fp']:>6.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
