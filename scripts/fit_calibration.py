"""Fit a probability calibrator against an inference cache, and print the config block.

``../TESI.md`` §8.2 calls temperature scaling the most significant technical debt in the
code: without it a log-odds fusion can be *worse* than the doctor alone, because an
over-confident model contributes a huge log-odds term and simply overrides its partner.
Both backends measured so far are over-confident — the ensemble at an expected
calibration error (ECE) of 0.266, MedGemma at 0.798 — so this is not hypothetical.

Usage:

    python scripts/fit_calibration.py --cache data/cache/nih_ensemble
    python scripts/fit_calibration.py --cache data/cache/medgemma_nih --method platt

The output is a YAML block to paste under ``ai:`` in a scenario.

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
    fit_temperature,
    fit_temperature_per_label,
)


def _fused(cache: InferenceCache) -> np.ndarray:
    """Mean over ensemble members — the probability a run actually sorts and fuses on."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.asarray(cache.member_probs, dtype=float), axis=1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fit a calibrator against an inference cache")
    p.add_argument("--cache", required=True, help="cache stem (no extension)")
    p.add_argument(
        "--method",
        default="temperature",
        choices=["temperature", "temperature_per_label", "platt", "isotonic"],
    )
    p.add_argument(
        "--holdout",
        type=float,
        default=0.5,
        help="fraction held out for the reported error (0 = fit and report on all)",
    )
    p.add_argument("--seed", type=int, default=0, help="split seed")
    args = p.parse_args(argv)

    cache = InferenceCache.load(args.cache)
    probs = _fused(cache)
    labels = np.asarray(cache.ground_truth, dtype=float)
    print(f"cache {args.cache}: {cache.n_images} images x {len(cache.pathologies)} labels")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(cache.n_images)
    cut = int(cache.n_images * (1.0 - args.holdout)) if args.holdout > 0 else cache.n_images
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
