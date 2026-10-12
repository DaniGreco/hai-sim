"""Fit a probability calibrator against an inference cache, and print the config block.

Without calibration a log-odds fusion can be *worse* than the doctor alone (thesis
§2.4), because an over-confident model contributes a huge log-odds term and simply
overrides its partner.
Both backends measured so far are over-confident — the ensemble at an expected
calibration error (ECE) of 0.266, MedGemma at 0.798 — so this is not hypothetical.

Usage:

    # the study path: per-pathology Platt for one (model, dataset), written next to the cache
    python scripts/fit_calibration.py --method platt_per_label --split even \
        --cache data/cache/vindr__chex data/cache/vindr__mimic_ch \
                data/cache/vindr__pc data/cache/vindr__rsna \
        --out data/cache/calibration/vindr__ens4.json
    python scripts/fit_calibration.py --method platt_per_label --split even \
        --cache data/cache/vindr__medgemma-27b \
        --out data/cache/calibration/vindr__medgemma-27b.json

    # quick scalar fits, printed as a YAML block to paste under ``ai:``
    python scripts/fit_calibration.py --cache data/cache/vindr__chex --method platt

Several ``--cache`` stems are composed by image id into one ensemble (the cache matrix
stores one file per member); the fit is on the *fused* probability, the number a reader
is shown. ``--split even`` fits on the even-indexed rows and reports on the odd ones, the
same parity ``patient_generator.split: odd`` then studies.

⚠️ **Fit on data the study does not score.** Fitting and reporting on the same images is
circular; ``--holdout 0.5`` splits the cache and reports the error on the unseen half,
which is the number worth quoting. And a fitted calibrator belongs to
*(model + readout + prompt + population)* — a temperature fitted on NIH is not valid for
CheXpert, and for a generative backend not even for a different prompt.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from hai_sim.analysis.inference_cache import InferenceCache
from hai_sim.uq.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureScaling,
    expected_calibration_error,
    fit_isotonic,
    fit_platt,
    fit_platt_per_label,
    fit_temperature,
    fit_temperature_per_label,
    save_platt_file,
)


def _fused(caches: list[InferenceCache]) -> tuple[np.ndarray, np.ndarray]:
    """Mean over every member of every cache, aligned by image id → ``(probs, labels)``.

    The probability a run actually sorts and fuses on. Rows follow the first cache's
    order restricted to the ids all caches share, which is how `CompositeCachedAI`
    composes them at replay time.
    """
    later = [set(c.image_ids) for c in caches[1:]]
    common = [i for i in caches[0].image_ids if all(i in s for s in later)]
    blocks = []
    for c in caches:
        pos = {img: r for r, img in enumerate(c.image_ids)}
        rows = [pos[img] for img in common]
        blocks.append(np.asarray(c.member_probs, dtype=float)[rows])
    stack = np.concatenate(blocks, axis=1)
    pos0 = {img: r for r, img in enumerate(caches[0].image_ids)}
    labels = np.asarray(caches[0].ground_truth, dtype=float)[[pos0[i] for i in common]]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN = uncovered label
        return np.nanmean(stack, axis=1), labels


def _per_label_ece(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.full(probs.shape[1], np.nan)
    for k in range(probs.shape[1]):
        ok = np.isfinite(probs[:, k]) & (labels[:, k] >= 0)
        if ok.sum() and labels[ok, k].sum() > 0:
            out[k] = expected_calibration_error(probs[ok, k], labels[ok, k])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fit a calibrator against an inference cache")
    p.add_argument(
        "--cache", required=True, nargs="+",
        help="cache stem(s), no extension; several are composed into one ensemble",
    )
    p.add_argument(
        "--method",
        default="temperature",
        choices=["temperature", "temperature_per_label", "platt", "platt_per_label", "isotonic"],
    )
    p.add_argument(
        "--split",
        default="random",
        choices=["random", "even"],
        help="'even' fits on even-indexed rows and reports on odd ones (the study split)",
    )
    p.add_argument("--out", help="platt_per_label: calibration file to write")
    p.add_argument(
        "--holdout",
        type=float,
        default=0.5,
        help="fraction held out for the reported error (0 = fit and report on all)",
    )
    p.add_argument("--seed", type=int, default=0, help="split seed")
    args = p.parse_args(argv)

    caches = [InferenceCache.load(stem) for stem in args.cache]
    cache = caches[0]
    probs, labels = _fused(caches)
    n_images = probs.shape[0]
    print(f"cache {' + '.join(args.cache)}: {n_images} images x {len(cache.pathologies)} labels")

    if args.split == "even":
        rows = np.arange(n_images)
        fit_idx, test_idx = rows[rows % 2 == 0], rows[rows % 2 == 1]
    else:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(n_images)
        cut = int(n_images * (1.0 - args.holdout)) if args.holdout > 0 else n_images
        fit_idx, test_idx = order[:cut], (order[cut:] if args.holdout > 0 else order)
    if fit_idx.size == 0 or test_idx.size == 0:
        print("[error] holdout leaves one side empty; lower --holdout")
        return 2

    fit_p, fit_y = probs[fit_idx], labels[fit_idx]
    test_p, test_y = probs[test_idx], labels[test_idx]

    if args.method == "temperature":
        value = fit_temperature(fit_p, fit_y)
        calibrator = TemperatureScaling(value)
        block = f"    kind: temperature\n    temperature: {value:.4f}"
    elif args.method == "temperature_per_label":
        values = fit_temperature_per_label(fit_p, fit_y)
        calibrator = TemperatureScaling(values)
        rendered = ", ".join(f"{v:.4f}" for v in values)
        block = f"    kind: temperature\n    temperature: [{rendered}]"
        identity = int(np.sum(np.isclose(values, 1.0)))
        if identity:
            print(
                f"note: {identity}/{values.size} labels came back at 1.0 (identity) — "
                "those had nothing to fit. Expected on a small cache with rare findings."
            )
    elif args.method == "platt_per_label":
        a, b, n_pos = fit_platt_per_label(fit_p, fit_y)
        calibrator = PlattCalibrator(a, b)
        thr_raw = np.full(fit_p.shape[1], np.nan)
        for k in range(fit_p.shape[1]):
            ok = np.isfinite(fit_p[:, k]) & (fit_y[:, k] >= 0)
            if ok.any() and fit_y[ok, k].sum() > 0:
                thr_raw[k] = np.quantile(fit_p[ok, k], 1.0 - fit_y[ok, k].mean())
        thr_cal = calibrator.calibrate(np.tile(thr_raw, (1, 1)))[0]
        before_k = _per_label_ece(test_p, test_y)
        after_k = _per_label_ece(calibrator.calibrate(test_p), test_y)
        print(f"\n{'pathology':28s} {'n_pos(fit)':>10s} {'a':>7s} {'b':>7s} {'ECE raw':>8s} {'ECE cal':>8s}")
        for k, name in enumerate(cache.pathologies):
            if n_pos[k] == 0:
                continue
            flag = "  low support" if n_pos[k] < 40 else ""
            print(f"{name:28s} {n_pos[k]:10d} {a[k]:7.3f} {b[k]:7.3f} "
                  f"{before_k[k]:8.4f} {after_k[k]:8.4f}{flag}")
        block = "    kind: platt\n    path: <--out>"
        if args.out:
            save_platt_file(
                args.out,
                pathologies=cache.pathologies,
                a=a,
                b=b,
                n_positive=n_pos,
                prevalence=np.array(
                    [
                        np.mean(fit_y[fit_y[:, k] >= 0, k]) if np.any(fit_y[:, k] >= 0) else np.nan
                        for k in range(fit_y.shape[1])
                    ]
                ),
                threshold_raw=thr_raw,
                threshold_calibrated=thr_cal,
                provenance={
                    "caches": list(args.cache),
                    "model_fingerprints": [c.model_fingerprint for c in caches],
                    "dataset_fingerprint": cache.dataset_fingerprint,
                    "fitted_on": f"{args.split} split, {fit_idx.size} images",
                    "calibrated_quantity": "fused probability (mean over members)",
                    "ece_held_out": {
                        name: {"raw": float(before_k[k]), "calibrated": float(after_k[k])}
                        for k, name in enumerate(cache.pathologies)
                        if np.isfinite(before_k[k])
                    },
                },
            )
            block = f"    kind: platt\n    path: {args.out}"
            print(f"\nwrote {args.out}")
    elif args.method == "platt":
        a, b = fit_platt(fit_p, fit_y)
        calibrator = PlattCalibrator(a, b)
        block = f"    kind: platt\n    a: {a:.4f}\n    b: {b:.4f}"
    else:
        x, y = fit_isotonic(fit_p, fit_y)
        # Thin to a manageable knot set: a config should be readable, and an isotonic
        # map with thousands of knots is memorising the fit sample anyway.
        keep = np.linspace(0, x.size - 1, min(21, x.size)).astype(int)
        calibrator = IsotonicCalibrator(x[keep], y[keep])
        block = (
            f"    kind: isotonic\n"
            f"    knots_x: [{', '.join(f'{v:.4f}' for v in x[keep])}]\n"
            f"    knots_y: [{', '.join(f'{v:.4f}' for v in y[keep])}]"
        )

    before = expected_calibration_error(test_p, test_y)
    after = expected_calibration_error(calibrator.calibrate(test_p), test_y)
    where = "held-out" if args.holdout > 0 else "in-sample (NOT a fair estimate)"
    print(f"\nECE on {test_idx.size} {where} images: {before:.4f} -> {after:.4f}")
    if after >= before:
        print(
            "WARNING: calibration did not help. Either the model is already calibrated on this "
            "population, or the family is the wrong shape — try --method platt if the "
            "model systematically over- or under-calls findings."
        )
    print("\nPaste under `ai:` in the scenario:\n")
    print("  calibration:")
    print(block)
    print(
        "\nWARNING: valid only for this model + readout + prompt + population. Refit for "
        "any other, and never report on the images you fitted on."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
