"""Cumulative error-capture curves: does the model err when it is uncertain?

Sort the units (images, or per-(image,label) entries) by **decreasing** model
uncertainty and walk from the most uncertain to the least, accumulating how many
the model got wrong. Three reference curves frame the result:

- **oracle** (red): the best possible ordering — all errors first — so the curve
  rises at slope 1 up to the total error count, then plateaus.
- **random** (gray): an uninformative ordering — a straight line to the same end
  point. Uncertainty carries signal only insofar as the actual curve sits above it.
- **actual** (green): the curve produced by the uncertainty ordering.

Two scalar summaries per panel:

- ``skill`` = (area_actual - area_random) / (area_oracle - area_random), <= 1.
  1 = oracle, 0 = random, < 0 = worse than random (anti-correlated). This is the
  cumulative-curve analogue of an AUSE / Gini concentration score.
- ``auroc`` = probability a random error-unit is ranked more uncertain than a
  random correct one (threshold-free failure-prediction AUC).

Everything is derived from an :class:`InferenceCache` for any UQ measure, decision
threshold, aggregation and error type — no model calls. The per-image view
aggregates the uncertainty vector exactly as ``UncertaintyDescSort`` does (so it
mirrors the triage), while the per-entry view keeps each (image, label) prediction
separate (the cleaner statistical object, decomposable per pathology).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from hai_sim.analysis.inference_cache import InferenceCache, measures_from_cache

ErrorType = Literal["any", "fn", "fp"]
Mode = Literal["image", "entry"]
Aggregator = Literal["mean", "max"]

_MEASURE_ALIASES = {"entropy": "predictive_entropy"}


@dataclass(frozen=True)
class CurveResult:
    """One cumulative error-capture curve plus its references and scalar scores."""

    x: np.ndarray
    """``1 .. K`` — number of units reviewed in decreasing-uncertainty order."""
    actual: np.ndarray
    """Cumulative error count under the uncertainty ordering."""
    oracle: np.ndarray
    """Best-possible cumulative count = ``min(x, n_errors)``."""
    random: np.ndarray
    """Expected cumulative count under random ordering = ``x * n_errors / K``."""
    n_units: int
    n_errors: int
    skill: float
    auroc: float
    error_type: ErrorType
    mode: Mode
    measure: str
    aggregator: str | None
    threshold: float


def _resolve_measure(measure: str) -> str:
    return _MEASURE_ALIASES.get(measure, measure)


def _rankdata_avg(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties sharing the mean of their rank span."""
    order = np.argsort(a, kind="stable")
    sa = a[order]
    ranks = np.empty(a.shape[0], dtype=float)
    i = 0
    n = a.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUROC of ``score`` predicting the boolean ``positive``."""
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_avg(np.asarray(score, dtype=float))
    sum_pos = float(ranks[positive].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def compute_curve(
    uncertainty: np.ndarray,
    is_error: np.ndarray,
    *,
    error_type: ErrorType,
    mode: Mode,
    measure: str,
    threshold: float,
    aggregator: str | None = None,
) -> CurveResult:
    """Build the cumulative error-capture curve for one set of units."""
    uncertainty = np.asarray(uncertainty, dtype=float)
    is_error = np.asarray(is_error, dtype=bool)
    k = uncertainty.shape[0]
    x = np.arange(1, k + 1, dtype=float)
    # Descending uncertainty; stable so equal-uncertainty ties keep input order.
    order = np.argsort(-uncertainty, kind="stable")
    actual = np.cumsum(is_error[order].astype(float))
    n_err = int(is_error.sum())
    oracle = np.minimum(x, float(n_err))
    random = x * (n_err / k) if k > 0 else x.copy()

    area_actual = float(actual.sum())
    area_oracle = float(oracle.sum())
    area_random = float(random.sum())
    denom = area_oracle - area_random
    skill = (area_actual - area_random) / denom if denom > 0 else 0.0

    return CurveResult(
        x=x,
        actual=actual,
        oracle=oracle,
        random=random,
        n_units=k,
        n_errors=n_err,
        skill=skill,
        auroc=_auroc(uncertainty, is_error),
        error_type=error_type,
        mode=mode,
        measure=measure,
        aggregator=aggregator,
        threshold=threshold,
    )


def _units_per_image(
    cache: InferenceCache, measure: str, threshold: float, aggregator: Aggregator
) -> tuple[np.ndarray, dict[ErrorType, np.ndarray]]:
    """Per-image unit: uncertainty aggregated over all labels (triage view)."""
    m = measures_from_cache(cache.member_probs)
    unc_vec = np.nan_to_num(m[_resolve_measure(measure)], nan=0.0)  # (N, L)
    mean_p = np.nan_to_num(m["mean_p"], nan=0.0)  # (N, L)
    pred = (mean_p >= threshold).astype(np.int8)
    gt = cache.ground_truth
    valid = gt >= 0
    if aggregator == "mean":
        unc = unc_vec.mean(axis=1)
    elif aggregator == "max":
        unc = unc_vec.max(axis=1)
    else:  # pragma: no cover - guarded by Literal
        raise ValueError(f"unknown aggregator: {aggregator}")
    errs: dict[ErrorType, np.ndarray] = {
        "any": ((pred != gt) & valid).any(axis=1),
        "fn": ((pred == 0) & (gt == 1)).any(axis=1),
        "fp": ((pred == 1) & (gt == 0)).any(axis=1),
    }
    return unc, errs


def _units_per_entry(
    cache: InferenceCache, measure: str, threshold: float
) -> tuple[np.ndarray, dict[ErrorType, np.ndarray]]:
    """Per-(image, label) unit over annotated entries only (gt in {0, 1})."""
    m = measures_from_cache(cache.member_probs)
    unc_vec = np.nan_to_num(m[_resolve_measure(measure)], nan=0.0)
    mean_p = np.nan_to_num(m["mean_p"], nan=0.0)
    pred = (mean_p >= threshold).astype(np.int8)
    gt = cache.ground_truth
    valid = gt >= 0
    p = pred[valid]
    g = gt[valid]
    errs: dict[ErrorType, np.ndarray] = {
        "any": p != g,
        "fn": (p == 0) & (g == 1),
        "fp": (p == 1) & (g == 0),
    }
    return unc_vec[valid], errs


def render_error_curves(
    cache: InferenceCache,
    out_path: str | Path,
    *,
    measure: str = "mutual_information",
    threshold: float = 0.5,
    aggregator: Aggregator = "mean",
    title_extra: str = "",
) -> dict[str, dict[str, float]]:
    """Render the 2x3 figure (image/entry by any/fn/fp) and a scalar-summary JSON."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unc_img, errs_img = _units_per_image(cache, measure, threshold, aggregator)
    unc_ent, errs_ent = _units_per_entry(cache, measure, threshold)
    rows: list[tuple[Mode, np.ndarray, dict[ErrorType, np.ndarray], str]] = [
        ("image", unc_img, errs_img, f"per-image (multi-label, agg={aggregator})"),
        ("entry", unc_ent, errs_ent, "per-entry (all labels pooled)"),
    ]
    cols: list[tuple[ErrorType, str]] = [
        ("any", "any error"),
        ("fn", "false negatives"),
        ("fp", "false positives"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    summary: dict[str, dict[str, float]] = {}
    for r, (mode, unc, errs, row_title) in enumerate(rows):
        for c, (etype, etitle) in enumerate(cols):
            ax = axes[r][c]
            cr = compute_curve(
                unc,
                errs[etype],
                error_type=etype,
                mode=mode,
                measure=measure,
                threshold=threshold,
                aggregator=aggregator if mode == "image" else None,
            )
            ax.plot(cr.x, cr.oracle, color="red", ls="--", lw=1.2, label="ideal (oracle)")
            ax.plot(cr.x, cr.random, color="gray", ls=":", lw=1.4, label="random")
            ax.plot(cr.x, cr.actual, color="green", lw=1.9, label="actual")
            ax.set_title(
                f"{row_title} — {etitle}\n"
                f"skill={cr.skill:.2f}  AUROC={cr.auroc:.2f}  (E={cr.n_errors}/{cr.n_units})",
                fontsize=9,
            )
            unit = "images" if mode == "image" else "entries"
            ax.set_xlabel(f"{unit} reviewed (decreasing uncertainty →)", fontsize=8)
            ax.set_ylabel("cumulative errors", fontsize=8)
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(alpha=0.25)
            summary[f"{mode}/{etype}"] = {
                "skill": cr.skill,
                "auroc": cr.auroc,
                "n_errors": float(cr.n_errors),
                "n_units": float(cr.n_units),
            }

    suptitle = (
        f"Uncertainty → error capture  |  measure={measure}  "
        f"threshold={threshold}  N={cache.n_images}"
    )
    if title_extra:
        suptitle += f"  {title_extra}"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    out_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "measure": measure,
                "threshold": threshold,
                "aggregator": aggregator,
                "n_images": cache.n_images,
                "panels": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def per_pathology_curves(
    cache: InferenceCache,
    *,
    measure: str = "mutual_information",
    threshold: float = 0.5,
    error_type: ErrorType = "any",
) -> dict[str, CurveResult]:
    """One error-capture curve per annotated pathology (label kept separate).

    Unlike the pooled per-entry view, each label is scored on its own entries, so
    pathologies with very different prevalence / uncertainty scale don't mix.
    Labels no member annotates (all ground truth -1, e.g. the non-NIH columns) are
    skipped. Pooling all labels would let the most frequent / highest-uncertainty
    pathology dominate the ranking — this separates where the UQ works from where
    it doesn't.
    """
    m = measures_from_cache(cache.member_probs)
    unc = np.nan_to_num(m[_resolve_measure(measure)], nan=0.0)
    mean_p = np.nan_to_num(m["mean_p"], nan=0.0)
    pred = (mean_p >= threshold).astype(np.int8)
    gt = cache.ground_truth

    results: dict[str, CurveResult] = {}
    for k, name in enumerate(cache.pathologies):
        valid = gt[:, k] >= 0
        if not valid.any():
            continue  # label not annotated by this dataset
        p = pred[valid, k]
        g = gt[valid, k]
        if error_type == "fn":
            err = (p == 0) & (g == 1)
        elif error_type == "fp":
            err = (p == 1) & (g == 0)
        else:
            err = p != g
        results[name] = compute_curve(
            unc[valid, k],
            err,
            error_type=error_type,
            mode="entry",
            measure=measure,
            threshold=threshold,
        )
    return results


def render_per_pathology(
    cache: InferenceCache,
    out_path: str | Path,
    *,
    measure: str = "mutual_information",
    threshold: float = 0.5,
    error_type: ErrorType = "any",
    ncols: int = 4,
) -> dict[str, dict[str, float]]:
    """Render one small error-capture panel per annotated pathology + summary JSON."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = per_pathology_curves(
        cache, measure=measure, threshold=threshold, error_type=error_type
    )
    names = list(results)
    n = len(names)
    nrows = max(1, -(-n // ncols))  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    summary: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        cr = results[name]
        ax = axes[i // ncols][i % ncols]
        ax.plot(cr.x, cr.oracle, color="red", ls="--", lw=1.0)
        ax.plot(cr.x, cr.random, color="gray", ls=":", lw=1.2)
        ax.plot(cr.x, cr.actual, color="green", lw=1.7)
        ax.set_title(
            f"{name}\nAUROC={cr.auroc:.2f} skill={cr.skill:.2f} (E={cr.n_errors}/{cr.n_units})",
            fontsize=8,
        )
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
        summary[name] = {
            "skill": cr.skill,
            "auroc": cr.auroc,
            "n_errors": float(cr.n_errors),
            "n_units": float(cr.n_units),
        }
    for j in range(n, nrows * ncols):  # hide unused cells
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        f"Per-pathology error capture ({error_type})  |  measure={measure}  "
        f"threshold={threshold}  N={cache.n_images}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    out_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "measure": measure,
                "threshold": threshold,
                "error_type": error_type,
                "n_images": cache.n_images,
                "pathologies": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def render_dashboard(
    cache: InferenceCache,
    out_dir: str | Path,
    *,
    measure: str = "predictive_entropy",
    threshold: float = 0.5,
    aggregator: Aggregator = "mean",
    title_extra: str = "",
    comparison_measure: str | None = "mutual_information",
) -> None:
    """Render the standard incremental figure set into ``out_dir``.

    Writes (overwriting each time, so a watcher / the cache builder can refresh
    them as the cache grows):

    - ``error_curves.png``               — main 2x3 with ``measure``
    - ``error_curves_<comparison>.png``  — main 2x3 with ``comparison_measure``
    - ``per_pathology_fp.png``           — per-pathology FP curves with ``measure``
    - ``per_pathology_fn.png``           — per-pathology FN curves with ``measure``
    """
    out_dir = Path(out_dir)
    render_error_curves(
        cache,
        out_dir / "error_curves.png",
        measure=measure,
        threshold=threshold,
        aggregator=aggregator,
        title_extra=title_extra,
    )
    if comparison_measure and comparison_measure != measure:
        render_error_curves(
            cache,
            out_dir / f"error_curves_{comparison_measure}.png",
            measure=comparison_measure,
            threshold=threshold,
            aggregator=aggregator,
            title_extra=title_extra,
        )
    render_per_pathology(
        cache, out_dir / "per_pathology_fp.png", measure=measure, threshold=threshold, error_type="fp"
    )
    render_per_pathology(
        cache, out_dir / "per_pathology_fn.png", measure=measure, threshold=threshold, error_type="fn"
    )
