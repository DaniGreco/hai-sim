"""RQ1 at scale: does ordering by uncertainty surface the AI's errors first?

The harness for `docs/STUDY_uncertainty_sweep.md`. Everything here is derived from
the on-disk inference caches — no model calls. Three pieces the standard dashboard
(`error_curve.py`) does not provide:

1. **`compose_member_caches`** — the DenseNet ensemble lives on disk as one cell *per
   member* (`<dataset>__chex`, `__mimic_ch`, ...). This stacks them into a single
   `(N, M, L)` `InferenceCache` so the whole analysis stack sees "the ensemble" as one
   model. (Composition otherwise lives only inside the simulator's `CompositeCachedAI`.)

2. **Per-pathology decision thresholds** — the operating point. `error_curve.py` uses one
   scalar threshold for every label; a chest X-ray model needs one *per pathology* because
   prevalences span ~1% (Hernia) to ~40% (Infiltration). Three policies (§6 of the spec):
   ``fixed`` (0.5, the naive baseline), ``prevalence`` (predict-positive rate = observed
   prevalence → false alarms = misses), and ``cost_optimal`` (minimise
   ``c_fp·FP + c_fn·FN`` — the clinically honest one, cast wide when a miss costs more).
   The last two are **rank-based**, hence invariant to any monotone calibration.

3. **Measure-overlay and in-vs-OOD figures** — the standard dashboard draws one measure at
   a time; the whole point of RQ1 is *which* uncertainty concentrates the errors, so these
   overlay the epistemic / total / confidence measures (and the two domains) on one axes.

Locked study parameters (with Daniele, 2026-08-07): calibration = Platt per-pathology;
error costs false-negative : false-positive = 10 : 1, uniform across the 18 pathologies.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from hai_sim.analysis.error_curve import ErrorType, compute_curve
from hai_sim.analysis.inference_cache import InferenceCache, measures_from_cache

if TYPE_CHECKING:
    from collections.abc import Sequence

# Default error costs: a missed finding (false negative) is as bad as ten false alarms.
COST_FP = 10.0
COST_FN = 100.0

# The uncertainty keys `measures_from_cache` publishes, in the order we present them.
# For a single-member model (MedGemma) mutual_information/std are identically 0 and
# mean_entropy == predictive_entropy, so only the confidence ones carry signal there.
ENSEMBLE_MEASURES = (
    "mutual_information",  # epistemic — members disagree (ensemble only)
    "predictive_entropy",  # total
    "mean_entropy",  # aleatoric (ensemble only, else == predictive_entropy)
    "least_confidence",  # confidence-based
)

# A colour-blind-safe categorical palette, fixed slot order so a series keeps its colour.
_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
_INK = "#0b0b0b"
_MUTED = "#7a7a7a"
_ORACLE = "#c0392b"
_RANDOM = "#9aa0a6"


# ── 1. Compose member cells into one ensemble cache ──────────────────────────────


def compose_member_caches(
    stems: Sequence[str | Path], *, dataset_name: str | None = None
) -> InferenceCache:
    """Stack several single-member cells (aligned by image id) into one `(N, M, L)` cache.

    Members are matched on ``image_id`` — only images present in *every* member survive,
    in the first member's order — so a cell built over a slightly different image set still
    composes to a clean rectangle. Ground truth is a property of the dataset, not the
    member, so it is taken from the first cell (they are identical where they overlap).
    """
    caches = [InferenceCache.load(Path(s)) for s in stems]
    if not caches:
        raise ValueError("compose_member_caches needs at least one cell stem")
    ref = caches[0]
    id_sets = [set(c.image_ids) for c in caches]
    common = [img for img in ref.image_ids if all(img in s for s in id_sets[1:])]
    if not common:
        raise ValueError("member cells share no common image id to compose")

    rows = [{img: k for k, img in enumerate(c.image_ids)} for c in caches]
    n, m, ell = len(common), len(caches), ref.n_labels
    member_probs = np.full((n, m, ell), np.nan, dtype=np.float32)
    for j, cache in enumerate(caches):
        idx = np.fromiter((rows[j][img] for img in common), dtype=np.int64, count=n)
        # each member cell carries a single member on axis 1
        member_probs[:, j, :] = cache.member_probs[idx, 0, :]

    ref_idx = np.fromiter((rows[0][img] for img in common), dtype=np.int64, count=n)
    weights = tuple(
        (c.weights[0] if c.weights else Path(str(s)).name)
        for c, s in zip(caches, stems, strict=True)
    )
    return InferenceCache(
        member_probs=member_probs,
        ground_truth=ref.ground_truth[ref_idx],
        inference_seconds=ref.inference_seconds[ref_idx],
        image_ids=tuple(common),
        dataset_indices=ref.dataset_indices[ref_idx],
        weights=weights,
        pathologies=ref.pathologies,
        dataset=dataset_name or ref.dataset,
        batch_size=ref.batch_size,
        dataset_fingerprint=ref.dataset_fingerprint,
        model_fingerprint="ensemble:" + ",".join(weights),
    )


# ── 2. Per-pathology decision thresholds (the operating point) ───────────────────

ThresholdPolicy = Literal["fixed", "prevalence", "cost_optimal"]


def _cost_optimal_threshold(scores: np.ndarray, y: np.ndarray, c_fp: float, c_fn: float) -> float:
    """The cut on ``scores`` that minimises ``c_fp·FP + c_fn·FN`` for one pathology.

    Sort by descending score; calling the top-``j`` positive gives
    ``FP = j - (positives in top j)`` and ``FN = P - (positives in top j)``. The optimum is
    at one of the ``n+1`` cut positions; the returned threshold sits at the midpoint of the
    boundary scores so ``prob >= threshold`` reproduces that cut.
    """
    n = int(scores.size)
    if n == 0:
        return 0.5
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    cum_pos = np.concatenate([[0.0], np.cumsum(y[order].astype(float))])  # (n+1,)
    total_pos = cum_pos[-1]
    j = np.arange(n + 1, dtype=float)
    cost = c_fp * (j - cum_pos) + c_fn * (total_pos - cum_pos)
    jstar = int(np.argmin(cost))
    if jstar == 0:  # predict none positive
        return float(s[0]) + 1e-6
    if jstar == n:  # predict all positive
        return float(s[-1]) - 1e-6
    return float((s[jstar - 1] + s[jstar]) / 2.0)


def _prevalence_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """The cut whose predict-positive rate equals the observed prevalence (⇒ FP == FN)."""
    if scores.size == 0:
        return 0.5
    prev = float((y == 1).mean())
    if prev <= 0.0:
        return float(scores.max()) + 1e-6 if scores.size else 1.0
    if prev >= 1.0:
        return float(scores.min()) - 1e-6
    return float(np.quantile(scores, 1.0 - prev))


def per_pathology_thresholds(
    mean_p: np.ndarray,
    gt: np.ndarray,
    *,
    policy: ThresholdPolicy,
    value: float = 0.5,
    c_fp: float = COST_FP,
    c_fn: float = COST_FN,
) -> np.ndarray:
    """One decision threshold per label under the chosen policy, fitted on annotated entries.

    ``fixed`` returns ``value`` for every label; ``prevalence`` and ``cost_optimal`` fit
    each label on its own annotated entries (``gt`` in {0, 1}), leaving ``value`` where a
    label has nothing to fit.
    """
    ell = mean_p.shape[1]
    tau = np.full(ell, float(value), dtype=float)
    if policy == "fixed":
        return tau
    for k in range(ell):
        valid = gt[:, k] >= 0
        if not valid.any():
            continue
        s = mean_p[valid, k]
        y = gt[valid, k]
        if policy == "prevalence":
            tau[k] = _prevalence_threshold(s, y)
        elif policy == "cost_optimal":
            tau[k] = _cost_optimal_threshold(s, y, c_fp, c_fn)
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"unknown threshold policy: {policy!r}")
    return tau


# ── 3. Confusion + cost at a set of thresholds ───────────────────────────────────


@dataclass(frozen=True)
class Confusion:
    """Decision quality over annotated entries, with the weighted error cost."""

    fn: int
    fp: int
    tp: int
    tn: int
    cost: float

    @property
    def n(self) -> int:
        return self.fn + self.fp + self.tp + self.tn

    @property
    def fn_rate(self) -> float:
        return self.fn / self.n if self.n else 0.0

    @property
    def fp_rate(self) -> float:
        return self.fp / self.n if self.n else 0.0

    @property
    def cost_per_case(self) -> float:
        return self.cost / self.n if self.n else 0.0


def confusion_at(
    mean_p: np.ndarray,
    gt: np.ndarray,
    thresholds: np.ndarray,
    *,
    c_fp: float = COST_FP,
    c_fn: float = COST_FN,
    rows: np.ndarray | None = None,
) -> Confusion:
    """Confusion counts over annotated entries; ``rows`` restricts to a held-out slice."""
    if rows is not None:
        mean_p = mean_p[rows]
        gt = gt[rows]
    pred = (mean_p >= thresholds[None, :]).astype(np.int8)
    valid = gt >= 0
    p = pred[valid]
    g = gt[valid]
    tp = int(((p == 1) & (g == 1)).sum())
    tn = int(((p == 0) & (g == 0)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    return Confusion(fn=fn, fp=fp, tp=tp, tn=tn, cost=c_fp * fp + c_fn * fn)


# ── 4. Per-entry uncertainty + error, and the ranking curve ──────────────────────


def _entry_arrays(
    cache: InferenceCache, measure: str, thresholds: np.ndarray, error_type: ErrorType
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten to annotated (image, label) entries: the uncertainty and the error flag."""
    m = measures_from_cache(cache.member_probs)
    unc = np.nan_to_num(m[measure], nan=0.0)
    mean_p = np.nan_to_num(m["mean_p"], nan=0.0)
    pred = (mean_p >= thresholds[None, :]).astype(np.int8)
    gt = cache.ground_truth
    valid = gt >= 0
    p = pred[valid]
    g = gt[valid]
    if error_type == "fn":
        err = (p == 0) & (g == 1)
    elif error_type == "fp":
        err = (p == 1) & (g == 0)
    else:
        err = p != g
    return unc[valid], err


# ── 5. Figures ───────────────────────────────────────────────────────────────────

_ERR_COLS: tuple[tuple[ErrorType, str], ...] = (
    ("any", "errors (any)"),
    ("fn", "false negatives (misses)"),
    ("fp", "false positives (false alarms)"),
)


def _new_ax(fig: Any, ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d9d9d9")
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.grid(True, color="#ececec", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _capture_curve(unc: np.ndarray, err: np.ndarray, measure: str, etype: ErrorType) -> Any:
    return compute_curve(unc, err, error_type=etype, mode="entry", measure=measure, threshold=0.0)


def render_measure_overlay(
    cache: InferenceCache,
    out_path: str | Path,
    *,
    measures: Sequence[str],
    thresholds: np.ndarray,
    title: str,
    subtitle: str,
) -> dict[str, Any]:
    """One row, three panels (any / FN / FP); each overlays the measures' capture curves.

    x = fraction of entries reviewed (most uncertain first), y = fraction of that error
    type caught by then. The oracle and random references are shared across measures (same
    error set), so they are drawn once per panel. ``skill`` and failure-AUROC go in the
    legend — the two scalars that say whether the ordering carries signal.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), squeeze=False)
    fig.patch.set_facecolor("#ffffff")
    summary: dict[str, Any] = {}
    for c, (etype, etitle) in enumerate(_ERR_COLS):
        ax = axes[0][c]
        _new_ax(fig, ax)
        last = None
        for i, measure in enumerate(measures):
            unc, err = _entry_arrays(cache, measure, thresholds, etype)
            cr = _capture_curve(unc, err, measure, etype)
            last = cr
            if cr.n_errors == 0:
                continue
            x = cr.x / cr.n_units
            y = cr.actual / cr.n_errors
            ax.plot(
                x, y, color=_COLORS[i % len(_COLORS)], lw=2.0, zorder=3,
                label=f"{measure}  (skill {cr.skill:+.2f} · AUROC {cr.auroc:.2f})",
            )
            summary[f"{etype}/{measure}"] = {
                "skill": cr.skill, "auroc": cr.auroc,
                "n_errors": float(cr.n_errors), "n_units": float(cr.n_units),
            }
        if last is not None and last.n_errors > 0:
            xo = last.x / last.n_units
            ax.plot(xo, last.oracle / last.n_errors, color=_ORACLE, ls="--", lw=1.3,
                    zorder=2, label="ideal (oracle)")
            ax.plot(xo, last.random / last.n_errors, color=_RANDOM, ls=":", lw=1.5,
                    zorder=2, label="random")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(etitle, loc="left", fontsize=10, fontweight="bold", color=_INK)
        ax.set_xlabel("fraction of entries reviewed (most uncertain first →)", fontsize=8.5)
        if c == 0:
            ax.set_ylabel("fraction of errors captured", fontsize=8.5)
        ax.legend(fontsize=7.3, loc="lower right", frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.945, subtitle, ha="left", va="top", fontsize=9, color=_MUTED)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    _save(fig, Path(out_path))
    return summary


def render_in_vs_ood(
    caches: dict[str, InferenceCache],
    out_path: str | Path,
    *,
    measure: str,
    thresholds: dict[str, np.ndarray],
    title: str,
) -> dict[str, Any]:
    """Same measure, one curve per dataset — does uncertainty still find errors OOD?"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), squeeze=False)
    fig.patch.set_facecolor("#ffffff")
    summary: dict[str, Any] = {}
    for c, (etype, etitle) in enumerate(_ERR_COLS):
        ax = axes[0][c]
        _new_ax(fig, ax)
        last = None
        for i, (ds, cache) in enumerate(caches.items()):
            unc, err = _entry_arrays(cache, measure, thresholds[ds], etype)
            cr = _capture_curve(unc, err, measure, etype)
            last = cr
            if cr.n_errors == 0:
                continue
            ax.plot(cr.x / cr.n_units, cr.actual / cr.n_errors, color=_COLORS[i % len(_COLORS)],
                    lw=2.0, zorder=3, label=f"{ds}  (skill {cr.skill:+.2f} · AUROC {cr.auroc:.2f})")
            summary[f"{etype}/{ds}"] = {"skill": cr.skill, "auroc": cr.auroc}
        if last is not None and last.n_errors > 0:
            ax.plot(last.x / last.n_units, last.random / last.n_errors, color=_RANDOM,
                    ls=":", lw=1.5, zorder=2, label="random")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(etitle, loc="left", fontsize=10, fontweight="bold", color=_INK)
        ax.set_xlabel("fraction of entries reviewed (most uncertain first →)", fontsize=8.5)
        if c == 0:
            ax.set_ylabel("fraction of errors captured", fontsize=8.5)
        ax.legend(fontsize=7.6, loc="lower right", frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.945, f"measure: {measure}", ha="left", va="top", fontsize=9, color=_MUTED)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    _save(fig, Path(out_path))
    return summary


_POLICIES: tuple[tuple[ThresholdPolicy, str], ...] = (
    ("fixed", "fixed 0.5"),
    ("prevalence", "prevalence (FP=FN)"),
    ("cost_optimal", "cost-optimal 10:1"),
)


def render_threshold_policies(
    cache: InferenceCache,
    out_path: str | Path,
    *,
    c_fp: float = COST_FP,
    c_fn: float = COST_FN,
    seed: int = 0,
    title: str,
) -> dict[str, Any]:
    """Two panels: FN%/FP% per policy, and weighted cost per case — fit/report split 50/50.

    Thresholds for the fitted policies are learned on one half and scored on the other, so
    the numbers are honest (the fixed 0.5 policy is scored on the same held-out half for a
    fair comparison).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = measures_from_cache(cache.member_probs)
    mean_p = np.nan_to_num(m["mean_p"], nan=0.0)
    gt = cache.ground_truth
    rng = np.random.default_rng(seed)
    order = rng.permutation(cache.n_images)
    cut = cache.n_images // 2
    fit_rows, test_rows = order[:cut], order[cut:]

    labels, fn_rates, fp_rates, costs = [], [], [], []
    detail: dict[str, Any] = {}
    for policy, nice in _POLICIES:
        tau = per_pathology_thresholds(
            mean_p[fit_rows], gt[fit_rows], policy=policy, c_fp=c_fp, c_fn=c_fn
        )
        conf = confusion_at(mean_p, gt, tau, c_fp=c_fp, c_fn=c_fn, rows=test_rows)
        labels.append(nice)
        fn_rates.append(100 * conf.fn_rate)
        fp_rates.append(100 * conf.fp_rate)
        costs.append(conf.cost_per_case)
        detail[policy] = {
            "fn": conf.fn, "fp": conf.fp, "tp": conf.tp, "tn": conf.tn,
            "fn_rate": conf.fn_rate, "fp_rate": conf.fp_rate,
            "cost_per_case": conf.cost_per_case, "n_test": conf.n,
        }

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    fig.patch.set_facecolor("#ffffff")
    x = np.arange(len(labels), dtype=float)
    w = 0.38
    _new_ax(fig, axL)
    axL.bar(x - w / 2, fn_rates, w, color=_COLORS[0], label="false negatives %", zorder=3)
    axL.bar(x + w / 2, fp_rates, w, color=_COLORS[1], label="false positives %", zorder=3)
    for xi, v in zip(x - w / 2, fn_rates, strict=True):
        axL.text(xi, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=_INK)
    for xi, v in zip(x + w / 2, fp_rates, strict=True):
        axL.text(xi, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=_INK)
    axL.set_xticks(x)
    axL.set_xticklabels(labels, fontsize=8.5)
    axL.set_ylabel("% of annotated entries", fontsize=9)
    axL.set_title("error rate per policy", loc="left", fontsize=10, fontweight="bold")
    axL.legend(fontsize=8, frameon=False, loc="upper center")

    _new_ax(fig, axR)
    bars = axR.bar(x, costs, 0.55, color=_COLORS[2], zorder=3)
    best = int(np.argmin(costs))
    bars[best].set_color("#127a52")
    for xi, v in zip(x, costs, strict=True):
        axR.text(xi, v + max(costs) * 0.01, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=8, color=_INK)
    axR.set_xticks(x)
    axR.set_xticklabels(labels, fontsize=8.5)
    axR.set_ylabel(f"weighted cost per case  ({c_fn:g}·FN + {c_fp:g}·FP)/N", fontsize=9)
    axR.set_title("cost per policy  (lower is better)", loc="left", fontsize=10,
                  fontweight="bold")

    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.94, "per-pathology thresholds · 50/50 fit/report split, held-out",
             ha="left", va="top", fontsize=9, color=_MUTED)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    _save(fig, Path(out_path))
    return detail


def _save(fig: Any, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=125, facecolor="#ffffff")
    plt.close(fig)


# ── 6. Orchestration + master table ──────────────────────────────────────────────


def run_study(
    ensemble_cells: dict[str, Sequence[str]],
    out_dir: str | Path,
    *,
    measures: Sequence[str] = ENSEMBLE_MEASURES,
    threshold_policy: ThresholdPolicy = "cost_optimal",
    c_fp: float = COST_FP,
    c_fn: float = COST_FN,
    difficulty_tables: dict[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Compose each dataset's ensemble, render the figures, write the master tables.

    ``ensemble_cells`` maps ``dataset_id -> [member cell stems]``. The curve figures use
    the ``threshold_policy`` decision (default cost-optimal 10:1, in-sample — the ranking
    metric does not overfit); the policy figure does its own held-out split.
    ``difficulty_tables`` maps ``dataset_id -> (csv_path, value_column)`` for the
    human-difficulty alignment view (inter-reader disagreement).
    """
    out = Path(out_dir)
    caches = {ds: compose_member_caches(stems, dataset_name=ds) for ds, stems in ensemble_cells.items()}
    thresholds = {
        ds: per_pathology_thresholds(
            np.nan_to_num(measures_from_cache(c.member_probs)["mean_p"], nan=0.0),
            c.ground_truth, policy=threshold_policy, c_fp=c_fp, c_fn=c_fn,
        )
        for ds, c in caches.items()
    }

    curve_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {"datasets": {}, "params": {
        "threshold_policy": threshold_policy, "c_fp": c_fp, "c_fn": c_fn,
        "measures": list(measures)}}

    for ds, cache in caches.items():
        overlay = render_measure_overlay(
            cache, out / ds / "measure_overlay.png",
            measures=measures, thresholds=thresholds[ds],
            title=f"Does uncertainty find the errors? — ensemble · {ds}",
            subtitle=f"threshold: {threshold_policy} (FN:FP={c_fn:g}:{c_fp:g}) · "
                     f"{cache.n_images} images · per-entry view",
        )
        for key, vals in overlay.items():
            etype, measure = key.split("/", 1)
            curve_rows.append({"dataset": ds, "measure": measure, "error_type": etype, **vals})

        policy_detail = render_threshold_policies(
            cache, out / ds / "threshold_policies.png",
            c_fp=c_fp, c_fn=c_fn,
            title=f"Operating point: FN and FP cost per threshold policy — {ds}",
        )
        for policy, vals in policy_detail.items():
            policy_rows.append({"dataset": ds, "policy": policy, **vals})
        report["datasets"][ds] = {
            "n_images": cache.n_images, "overlay": overlay, "policies": policy_detail}

    # ── Per-pathology validation views (the citable layer) ──────────────────────
    pp_tables: dict[str, list[dict[str, Any]]] = {}
    pp_rows: list[dict[str, Any]] = []
    cal_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []
    for ds, cache in caches.items():
        table = per_pathology_table(cache, thresholds=thresholds[ds], measures=measures)
        pp_tables[ds] = table
        pp_rows.extend(table)
        render_per_pathology_measures(
            table, out / ds / "per_pathology_measures.png", measures=measures,
            title=f"Which errors uncertainty finds, pathology by pathology — {ds}",
        )
        rows_cal, pooled = calibration_panel(cache)
        cal_rows.extend(rows_cal)
        render_calibration_panel(
            rows_cal, pooled, out / ds / "calibration_panel.png",
            title=f"Per-pathology calibration: raw vs Platt (held-out) — {ds}",
        )
        rows_cost = cost_ratio_sweep(cache, c_fp=c_fp)
        cost_rows.extend(rows_cost)
        render_cost_ratio_sweep(
            rows_cost, out / ds / "cost_ratio_sweep.png",
            title=f"Sensitivity to the FN:FP cost ratio — {ds}",
        )
        rows_overlap = member_error_overlap(cache, c_fp=c_fp, c_fn=c_fn)
        overlap_rows.extend(rows_overlap)
        render_member_overlap(
            rows_overlap, out / ds / "member_overlap.png",
            title=f"Error overlap between ensemble members — {ds}",
        )
        if difficulty_tables and ds in difficulty_tables:
            path, column = difficulty_tables[ds]
            rows_diff, extras = difficulty_alignment(
                cache, path, value_column=column, thresholds=thresholds[ds],
            )
            diff_rows.extend(rows_diff)
            render_difficulty_alignment(
                rows_diff, extras, out / ds / "difficulty_alignment.png",
                title=f"Does model uncertainty track human difficulty? — {ds}",
            )
            report["datasets"][ds]["difficulty_coverage"] = extras["coverage"]

    # in-domain vs OOD, if both present: epistemic pooled curves + per-pathology dumbbell
    if len(caches) >= 2 and "mutual_information" in measures:
        report["in_vs_ood"] = render_in_vs_ood(
            caches, out / "in_vs_ood_mutual_information.png",
            measure="mutual_information", thresholds=thresholds,
            title="In-domain vs out-of-domain (OOD) — epistemic uncertainty (mutual information)",
        )
    if len(caches) == 2:
        capture = "mean_entropy" if "mean_entropy" in measures else measures[0]
        dumbbell_rows = render_cross_dataset_pathology(
            pp_tables, out / "pathology_in_vs_ood.png", capture_measure=capture
        )
        if dumbbell_rows:
            _write_csv(out / "pathology_in_vs_ood.csv", dumbbell_rows,
                       list(dumbbell_rows[0].keys()))

    _write_csv(out / "per_pathology_measures.csv", pp_rows,
               ["dataset", "pathology", "measure", "error_type", "skill", "auroc_unc",
                "auroc_disc", "prevalence", "n_members", "n_annotated", "n_errors",
                "low_support"])
    _write_csv(out / "calibration_panel.csv", cal_rows,
               ["dataset", "pathology", "platt_a", "platt_b", "ece_raw", "ece_platt",
                "n_test", "n_pos_test"])
    _write_csv(out / "cost_ratio_sweep.csv", cost_rows,
               ["dataset", "ratio", "pathology", "threshold", "fn_rate", "fp_rate",
                "sensitivity", "specificity", "cost_per_case"])
    _write_csv(out / "member_overlap.csv", overlap_rows,
               ["dataset", "pathology", "pair", "kappa", "err_rate_a", "err_rate_b", "n"])
    if diff_rows:
        _write_csv(out / "difficulty_alignment.csv", diff_rows,
                   ["dataset", "pathology", "n_images", "spearman_predictive_entropy",
                    "spearman_mean_entropy", "spearman_mutual_information"])

    _write_csv(out / "master_curves.csv", curve_rows,
               ["dataset", "measure", "error_type", "skill", "auroc", "n_errors", "n_units"])
    _write_csv(out / "master_policies.csv", policy_rows,
               ["dataset", "policy", "fn", "fp", "tp", "tn", "fn_rate", "fp_rate",
                "cost_per_case", "n_test"])
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ── 7. Per-pathology validation views ────────────────────────────────────────────
#
# The pooled tables above answer "does uncertainty carry signal at all"; every claim
# the thesis actually defends is per pathology, because prevalences span 1%..40% and a
# pooled number is dominated by whichever label is most common (Simpson). Everything
# below decomposes the same cache into one row per (pathology, ...) and renders the
# matching figure. Values with a polarity (AUROC around 0.5, kappa around 0) use one
# shared diverging map with a neutral midpoint; identities keep the fixed palette.

# Below this many errors a skill/AUROC estimate is noise; rows are kept but flagged.
MIN_ERRORS_PER_PATHOLOGY = 10

COST_RATIOS = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0)


def _diverging_cmap() -> Any:
    """Orange (below midpoint) → neutral → blue (above midpoint), from the fixed palette."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("hai_div", [_COLORS[1], "#f2f1ee", _COLORS[0]])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (average ranks), NaN on degenerate input."""
    from hai_sim.analysis.error_curve import _rankdata_avg

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    rx = _rankdata_avg(x[keep])
    ry = _rankdata_avg(y[keep])
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _split_rows(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The 50/50 fit/report split every data-dependent quantity here shares."""
    order = np.random.default_rng(seed).permutation(n)
    cut = n // 2
    return order[:cut], order[cut:]


def per_pathology_table(
    cache: InferenceCache,
    *,
    thresholds: np.ndarray,
    measures: Sequence[str] = ENSEMBLE_MEASURES,
) -> list[dict[str, Any]]:
    """One row per (pathology, measure, error_type): capture skill/AUROC + shared stats.

    ``auroc_disc`` (discrimination) and ``prevalence`` are constant across the measure
    and error-type axes and repeated on every row so the CSV is self-contained.
    """
    from hai_sim.analysis.error_curve import _auroc

    m = measures_from_cache(cache.member_probs)
    mean_p = np.nan_to_num(m["mean_p"], nan=0.0)
    gt = cache.ground_truth
    # Effective ensemble size per label: members whose weight set covers it. With
    # M=1 the epistemic measure is identically zero and the aleatoric one equals the
    # predictive entropy, so those rows carry no ensemble-specific signal.
    n_members = np.isfinite(cache.member_probs).any(axis=0).sum(axis=0)
    rows: list[dict[str, Any]] = []
    for k, patho in enumerate(cache.pathologies):
        valid = gt[:, k] >= 0
        if not valid.any():
            continue
        y = gt[valid, k]
        p = mean_p[valid, k]
        prevalence = float((y == 1).mean())
        auroc_disc = _auroc(p, y == 1)
        pred = (p >= thresholds[k]).astype(np.int8)
        flags: dict[ErrorType, np.ndarray] = {
            "any": pred != y,
            "fn": (pred == 0) & (y == 1),
            "fp": (pred == 1) & (y == 0),
        }
        for measure in measures:
            unc = np.nan_to_num(m[measure], nan=0.0)[valid, k]
            for etype, err in flags.items():
                cr = _capture_curve(unc, err, measure, etype)
                rows.append({
                    "dataset": cache.dataset,
                    "pathology": patho,
                    "measure": measure,
                    "error_type": etype,
                    "skill": cr.skill,
                    "auroc_unc": cr.auroc,
                    "auroc_disc": auroc_disc,
                    "prevalence": prevalence,
                    "n_members": int(n_members[k]),
                    "n_annotated": int(valid.sum()),
                    "n_errors": int(err.sum()),
                    "low_support": int(err.sum()) < MIN_ERRORS_PER_PATHOLOGY,
                })
    return rows


def render_per_pathology_measures(
    rows: Sequence[dict[str, Any]],
    out_path: str | Path,
    *,
    measures: Sequence[str] = ENSEMBLE_MEASURES,
    title: str,
) -> None:
    """Three heatmap panels (any/FN/FP): pathologies x measures, cell = failure AUROC.

    Diverging around 0.5 — blue above (uncertainty finds this label's errors), orange
    below (anti-predictive). Low-support cells are annotated in muted ink.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pathologies = sorted(
        {r["pathology"] for r in rows},
        key=lambda p: -next(r["prevalence"] for r in rows if r["pathology"] == p),
    )
    cmap = _diverging_cmap()
    members_of = {r["pathology"]: r.get("n_members") for r in rows}
    row_labels = [
        f"{p}  (M={members_of[p]})" if members_of.get(p) else p for p in pathologies
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(14.5, 0.42 * len(pathologies) + 2.2), squeeze=False
    )
    fig.patch.set_facecolor("#ffffff")
    for c, (etype, etitle) in enumerate(_ERR_COLS):
        ax = axes[0][c]
        grid = np.full((len(pathologies), len(measures)), np.nan)
        support = np.ones_like(grid, dtype=bool)
        degenerate = np.zeros_like(grid, dtype=bool)
        for r in rows:
            if r["error_type"] != etype:
                continue
            i = pathologies.index(r["pathology"])
            j = list(measures).index(r["measure"])
            # A single covering member makes the epistemic measure identically zero:
            # the cell would read 0.50 and look "anti-predictive". Leave it blank.
            if r["measure"] == "mutual_information" and (r.get("n_members") or 2) < 2:
                degenerate[i, j] = True
                continue
            grid[i, j] = r["auroc_unc"]
            support[i, j] = not r["low_support"]
        ax.imshow(grid, cmap=cmap, vmin=0.2, vmax=0.8, aspect="auto")
        for i in range(len(pathologies)):
            for j in range(len(measures)):
                if degenerate[i, j]:
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.8,
                            color=_MUTED, style="italic")
                elif np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                            fontsize=6.8, color=_INK if support[i, j] else _MUTED)
        ax.set_xticks(range(len(measures)))
        ax.set_xticklabels([m.replace("_", "\n") for m in measures], fontsize=7)
        if c == 0:
            ax.set_yticks(range(len(pathologies)))
            ax.set_yticklabels(row_labels, fontsize=7.5)
        else:
            ax.set_yticks([])
        ax.set_title(etitle, loc="left", fontsize=10, fontweight="bold", color=_INK)
        ax.tick_params(colors=_MUTED)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.955,
             "failure-prediction AUROC of the uncertainty, per pathology · blue > 0.5 = "
             "predictive, orange < 0.5 = anti-predictive · grey = < "
             f"{MIN_ERRORS_PER_PATHOLOGY} errors (do not cite) · M = members covering the "
             "label; n/a = epistemic measure identically zero (M=1)",
             ha="left", va="top", fontsize=8.5, color=_MUTED)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    _save(fig, Path(out_path))


def render_cross_dataset_pathology(
    tables: dict[str, list[dict[str, Any]]],
    out_path: str | Path,
    *,
    capture_measure: str = "mean_entropy",
) -> list[dict[str, Any]]:
    """Dumbbell panels over the common scorable labels: in-domain vs OOD, per pathology.

    Left: discrimination AUROC. Right: failure AUROC of ``capture_measure`` (any error).
    Returns the tidy rows behind the figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = list(tables)
    if len(datasets) != 2:
        return []
    a, b = datasets

    def _lookup(ds: str) -> dict[str, dict[str, Any]]:
        return {
            r["pathology"]: r
            for r in tables[ds]
            if r["measure"] == capture_measure and r["error_type"] == "any"
        }
    la, lb = _lookup(a), _lookup(b)
    common = [p for p in la if p in lb]
    common.sort(key=lambda p: la[p]["auroc_disc"] if np.isfinite(la[p]["auroc_disc"]) else -1)
    if not common:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.42 * len(common) + 2.0), squeeze=False)
    fig.patch.set_facecolor("#ffffff")
    panels = (("auroc_disc", "discrimination (AUROC vs ground truth)"),
              ("auroc_unc", f"error capture ({capture_measure}, any)"))
    ypos = np.arange(len(common), dtype=float)
    for c, (field, ptitle) in enumerate(panels):
        ax = axes[0][c]
        _new_ax(fig, ax)
        for i, patho in enumerate(common):
            va, vb = la[patho][field], lb[patho][field]
            if np.isfinite(va) and np.isfinite(vb):
                ax.plot([va, vb], [ypos[i]] * 2, color="#d0d0d0", lw=1.4, zorder=2)
        ax.scatter([la[p][field] for p in common], ypos, s=42, color=_COLORS[0],
                   zorder=3, label=a)
        ax.scatter([lb[p][field] for p in common], ypos, s=42, color=_COLORS[1],
                   zorder=3, label=b)
        ax.axvline(0.5, color=_MUTED, lw=0.9, ls=":", zorder=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(common if c == 0 else [""] * len(common), fontsize=7.5)
        ax.set_xlim(0.2, 1.0)
        ax.set_title(ptitle, loc="left", fontsize=10, fontweight="bold", color=_INK)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("In-domain vs out-of-domain, per pathology", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save(fig, Path(out_path))
    return [
        {"pathology": p,
         f"auroc_disc_{a}": la[p]["auroc_disc"], f"auroc_disc_{b}": lb[p]["auroc_disc"],
         f"auroc_unc_{a}": la[p]["auroc_unc"], f"auroc_unc_{b}": lb[p]["auroc_unc"],
         f"prevalence_{a}": la[p]["prevalence"], f"prevalence_{b}": lb[p]["prevalence"]}
        for p in common
    ]


def calibration_panel(
    cache: InferenceCache, *, seed: int = 0, n_bins: int = 10
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-pathology Platt fit on one half, ECE raw vs calibrated on the other.

    Returns (per-pathology rows, pooled reliability curves for the figure).
    """
    from hai_sim.uq.calibration import expected_calibration_error, fit_platt

    m = measures_from_cache(cache.member_probs)
    mean_p = np.nan_to_num(m["mean_p"], nan=0.0)
    gt = cache.ground_truth
    fit_rows, test_rows = _split_rows(cache.n_images, seed)

    def _platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
        q = np.clip(p, 1e-6, 1 - 1e-6)
        return np.asarray(1.0 / (1.0 + np.exp(-(a * np.log(q / (1 - q)) + b))))

    rows: list[dict[str, Any]] = []
    cal_p = np.empty_like(mean_p)
    for k, patho in enumerate(cache.pathologies):
        a, b = fit_platt(mean_p[fit_rows, k], gt[fit_rows, k])
        cal_p[:, k] = _platt(mean_p[:, k], a, b)
        ece_raw = expected_calibration_error(mean_p[test_rows, k], gt[test_rows, k])
        ece_cal = expected_calibration_error(cal_p[test_rows, k], gt[test_rows, k])
        valid = gt[test_rows, k] >= 0
        rows.append({
            "dataset": cache.dataset, "pathology": patho, "platt_a": a, "platt_b": b,
            "ece_raw": ece_raw, "ece_platt": ece_cal,
            "n_test": int(valid.sum()),
            "n_pos_test": int((gt[test_rows, k] == 1).sum()),
        })

    def _reliability(p: np.ndarray) -> dict[str, list[float]]:
        pv = p[test_rows].ravel()
        yv = gt[test_rows].ravel()
        keep = yv >= 0
        pv, yv = pv[keep], yv[keep]
        edges = np.linspace(0, 1, n_bins + 1)
        mid, obs, wt = [], [], []
        for i in range(n_bins):
            mask = (pv >= edges[i]) & (pv <= edges[i + 1] if i == n_bins - 1 else pv < edges[i + 1])
            if mask.sum() == 0:
                continue
            mid.append(float(pv[mask].mean()))
            obs.append(float((yv[mask] == 1).mean()))
            wt.append(float(mask.mean()))
        return {"confidence": mid, "observed": obs, "weight": wt}

    pooled = {"raw": _reliability(mean_p), "platt": _reliability(cal_p)}
    return rows, pooled


def render_calibration_panel(
    rows: Sequence[dict[str, Any]],
    pooled: dict[str, Any],
    out_path: str | Path,
    *,
    title: str,
) -> None:
    """Left: per-pathology ECE dumbbell raw → Platt. Right: pooled reliability curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda r: r["ece_raw"])
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.5, 0.4 * len(ordered) + 2.0),
        gridspec_kw={"width_ratios": (1.35, 1.0)},
    )
    fig.patch.set_facecolor("#ffffff")
    _new_ax(fig, axL)
    ypos = np.arange(len(ordered), dtype=float)
    for i, r in enumerate(ordered):
        axL.plot([r["ece_raw"], r["ece_platt"]], [i, i], color="#d0d0d0", lw=1.4, zorder=2)
    axL.scatter([r["ece_raw"] for r in ordered], ypos, s=40, color=_COLORS[1],
                zorder=3, label="raw")
    axL.scatter([r["ece_platt"] for r in ordered], ypos, s=40, color=_COLORS[0],
                zorder=3, label="Platt (held-out)")
    axL.set_yticks(ypos)
    axL.set_yticklabels([r["pathology"] for r in ordered], fontsize=7.5)
    axL.set_xlabel("expected calibration error (ECE)", fontsize=8.5)
    axL.set_title("ECE per pathology", loc="left", fontsize=10, fontweight="bold")
    axL.legend(fontsize=8, frameon=False, loc="lower right")

    _new_ax(fig, axR)
    axR.plot([0, 1], [0, 1], color=_MUTED, lw=0.9, ls=":", zorder=1, label="perfect")
    for name, color in (("raw", _COLORS[1]), ("platt", _COLORS[0])):
        rel = pooled[name]
        axR.plot(rel["confidence"], rel["observed"], marker="o", ms=4, lw=1.8,
                 color=color, zorder=3, label=name)
    axR.set_xlim(0, 1)
    axR.set_ylim(0, 1)
    axR.set_xlabel("predicted probability", fontsize=8.5)
    axR.set_ylabel("observed frequency", fontsize=8.5)
    axR.set_title("reliability (pooled, held-out)", loc="left", fontsize=10,
                  fontweight="bold")
    axR.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save(fig, Path(out_path))


def cost_ratio_sweep(
    cache: InferenceCache,
    *,
    ratios: Sequence[float] = COST_RATIOS,
    c_fp: float = COST_FP,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Cost-optimal operating point as the FN:FP price ratio varies, per pathology.

    For each ratio, thresholds are refit on the fit half and scored on the other half.
    Pooled rows carry ``pathology = "__pooled__"``; the per-pathology rows carry each
    label's threshold and sensitivity/specificity at that ratio.
    """
    m = measures_from_cache(cache.member_probs)
    mean_p = np.nan_to_num(m["mean_p"], nan=0.0)
    gt = cache.ground_truth
    fit_rows, test_rows = _split_rows(cache.n_images, seed)
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        c_fn = ratio * c_fp
        tau = per_pathology_thresholds(
            mean_p[fit_rows], gt[fit_rows], policy="cost_optimal", c_fp=c_fp, c_fn=c_fn
        )
        conf = confusion_at(mean_p, gt, tau, c_fp=c_fp, c_fn=c_fn, rows=test_rows)
        rows.append({
            "dataset": cache.dataset, "ratio": ratio, "pathology": "__pooled__",
            "threshold": float("nan"),
            "fn_rate": conf.fn_rate, "fp_rate": conf.fp_rate,
            "sensitivity": conf.tp / (conf.tp + conf.fn) if conf.tp + conf.fn else 0.0,
            "specificity": conf.tn / (conf.tn + conf.fp) if conf.tn + conf.fp else 0.0,
            "cost_per_case": conf.cost_per_case,
        })
        for k, patho in enumerate(cache.pathologies):
            y = gt[test_rows, k]
            valid = y >= 0
            if not valid.any():
                continue
            p = mean_p[test_rows, k][valid]
            yy = y[valid]
            pred = (p >= tau[k]).astype(np.int8)
            tp = int(((pred == 1) & (yy == 1)).sum())
            tn = int(((pred == 0) & (yy == 0)).sum())
            fp = int(((pred == 1) & (yy == 0)).sum())
            fn = int(((pred == 0) & (yy == 1)).sum())
            rows.append({
                "dataset": cache.dataset, "ratio": ratio, "pathology": patho,
                "threshold": float(tau[k]),
                "fn_rate": fn / max(tp + tn + fp + fn, 1),
                "fp_rate": fp / max(tp + tn + fp + fn, 1),
                "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
                "specificity": tn / (tn + fp) if tn + fp else 0.0,
                "cost_per_case": (c_fp * fp + c_fn * fn) / max(tp + tn + fp + fn, 1),
            })
    return rows


def render_cost_ratio_sweep(
    rows: Sequence[dict[str, Any]],
    out_path: str | Path,
    *,
    title: str,
    highlight: Sequence[str] = ("Effusion", "Pneumothorax", "Cardiomegaly", "Infiltration"),
) -> None:
    """Left: pooled FN%/FP% vs the price ratio. Right: per-pathology threshold curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = [r for r in rows if r["pathology"] == "__pooled__"]
    pooled.sort(key=lambda r: r["ratio"])
    ratios = [r["ratio"] for r in pooled]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    fig.patch.set_facecolor("#ffffff")
    _new_ax(fig, axL)
    axL.plot(ratios, [100 * r["fn_rate"] for r in pooled], marker="o", ms=4, lw=2.0,
             color=_COLORS[0], label="false negatives %")
    axL.plot(ratios, [100 * r["fp_rate"] for r in pooled], marker="o", ms=4, lw=2.0,
             color=_COLORS[1], label="false positives %")
    axL.set_xscale("log")
    axL.set_xticks(ratios)
    axL.set_xticklabels([f"{r:g}:1" for r in ratios], fontsize=8)
    axL.set_xlabel("FN : FP cost ratio", fontsize=8.5)
    axL.set_ylabel("% of annotated entries (held-out)", fontsize=8.5)
    axL.set_title("cost-optimal operating point vs price of a miss", loc="left",
                  fontsize=10, fontweight="bold")
    axL.legend(fontsize=8, frameon=False)

    _new_ax(axR.figure, axR)
    per_patho: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        if r["pathology"] != "__pooled__" and np.isfinite(r["threshold"]):
            per_patho.setdefault(r["pathology"], []).append((r["ratio"], r["threshold"]))
    for patho, pts in per_patho.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if patho in highlight:
            idx = list(highlight).index(patho)
            axR.plot(xs, ys, lw=2.0, color=_COLORS[idx % len(_COLORS)], zorder=3,
                     label=patho)
        else:
            axR.plot(xs, ys, lw=0.8, color="#cfcfcf", zorder=2)
    axR.set_xscale("log")
    axR.set_xticks(ratios)
    axR.set_xticklabels([f"{r:g}:1" for r in ratios], fontsize=8)
    axR.set_xlabel("FN : FP cost ratio", fontsize=8.5)
    axR.set_ylabel("cost-optimal threshold per pathology", fontsize=8.5)
    axR.set_title("the threshold falls as misses get pricier", loc="left", fontsize=10,
                  fontweight="bold")
    axR.legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    _save(fig, Path(out_path))


def member_error_overlap(
    cache: InferenceCache, *, c_fp: float = COST_FP, c_fn: float = COST_FN, seed: int = 0
) -> list[dict[str, Any]]:
    """Cohen's kappa on error overlap between every member pair, per pathology.

    Each member gets its own per-pathology cost-optimal thresholds (fit half), and
    kappa is computed on the report half, only where both members cover the label.
    Low kappa = the members err on different cases — the raw material of ensembling.
    """
    gt = cache.ground_truth
    fit_rows, test_rows = _split_rows(cache.n_images, seed)
    n_members = cache.n_members
    member_tau = []
    for j in range(n_members):
        probs = cache.member_probs[:, j, :].astype(float)
        tau = np.full(cache.n_labels, 0.5)
        for k in range(cache.n_labels):
            col = probs[fit_rows, k]
            y = gt[fit_rows, k]
            keep = np.isfinite(col) & (y >= 0)
            if keep.any():
                tau[k] = _cost_optimal_threshold(col[keep], y[keep], c_fp, c_fn)
        member_tau.append(tau)

    rows: list[dict[str, Any]] = []
    for j in range(n_members):
        for jj in range(j + 1, n_members):
            pa = cache.member_probs[test_rows, j, :].astype(float)
            pb = cache.member_probs[test_rows, jj, :].astype(float)
            y = gt[test_rows]
            for k, patho in enumerate(cache.pathologies):
                keep = np.isfinite(pa[:, k]) & np.isfinite(pb[:, k]) & (y[:, k] >= 0)
                if keep.sum() < 30:
                    continue
                yy = y[keep, k]
                ea = (pa[keep, k] >= member_tau[j][k]).astype(np.int8) != yy
                eb = (pb[keep, k] >= member_tau[jj][k]).astype(np.int8) != yy
                po = float((ea == eb).mean())
                ra, rb = float(ea.mean()), float(eb.mean())
                pe = ra * rb + (1 - ra) * (1 - rb)
                kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else float("nan")
                rows.append({
                    "dataset": cache.dataset, "pathology": patho,
                    "pair": f"{cache.weights[j]}x{cache.weights[jj]}",
                    "kappa": kappa, "err_rate_a": ra, "err_rate_b": rb,
                    "n": int(keep.sum()),
                })
    return rows


def render_member_overlap(
    rows: Sequence[dict[str, Any]], out_path: str | Path, *, title: str
) -> None:
    """Heatmap pathologies x member pairs, cell = error-overlap kappa (diverging at 0)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pathologies = sorted({r["pathology"] for r in rows})
    pairs = sorted({r["pair"] for r in rows})
    grid = np.full((len(pathologies), len(pairs)), np.nan)
    for r in rows:
        grid[pathologies.index(r["pathology"]), pairs.index(r["pair"])] = r["kappa"]
    fig, ax = plt.subplots(figsize=(1.6 * len(pairs) + 3.2, 0.42 * len(pathologies) + 2.0))
    fig.patch.set_facecolor("#ffffff")
    ax.imshow(grid, cmap=_diverging_cmap(), vmin=-0.6, vmax=0.6, aspect="auto")
    for i in range(len(pathologies)):
        for j in range(len(pairs)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:+.2f}", ha="center", va="center",
                        fontsize=6.8, color=_INK)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p.replace("x", "\nx") for p in pairs], fontsize=7)
    ax.set_yticks(range(len(pathologies)))
    ax.set_yticklabels(pathologies, fontsize=7.5)
    ax.tick_params(colors=_MUTED)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=_INK, pad=14)
    fig.text(0.01, 0.955,
             "Cohen's κ on errors (per-member cost-optimal thresholds, held-out) · "
             "0 = independent errors (exploitable complementarity) · 1 = identical errors",
             ha="left", va="top", fontsize=8.5, color=_MUTED)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save(fig, Path(out_path))


def difficulty_alignment(
    cache: InferenceCache,
    table_path: str | Path,
    *,
    value_column: str = "mean_delta_active",
    id_column: str = "example_id",
    thresholds: np.ndarray | None = None,
    n_bins: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Does the model's uncertainty track the *human* per-case difficulty?

    Joins the inter-reader disagreement table (one row per image) onto the cache by
    image id. Returns per-pathology Spearman correlations (uncertainty vs difficulty)
    and, for the figure, binned profiles: mean uncertainty and model error rate per
    difficulty decile.
    """
    table: dict[str, float] = {}
    with Path(table_path).open("r", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            table[str(rec[id_column])] = float(rec[value_column])
    delta = np.asarray([table.get(img, np.nan) for img in cache.image_ids])
    covered = np.isfinite(delta)

    m = measures_from_cache(cache.member_probs)
    mean_p = np.nan_to_num(m["mean_p"], nan=0.0)
    gt = cache.ground_truth
    if thresholds is None:
        thresholds = per_pathology_thresholds(mean_p, gt, policy="cost_optimal")

    rows: list[dict[str, Any]] = []
    for k, patho in enumerate(cache.pathologies):
        row: dict[str, Any] = {"dataset": cache.dataset, "pathology": patho,
                               "n_images": int(covered.sum())}
        for measure in ("predictive_entropy", "mean_entropy", "mutual_information"):
            unc = np.nan_to_num(m[measure], nan=0.0)[:, k]
            row[f"spearman_{measure}"] = _spearman(unc[covered], delta[covered])
        rows.append(row)

    # Binned profiles (per-entry, over covered images): uncertainty and error rate per
    # difficulty decile — the secondary, pooled view; the per-pathology numbers above
    # are the citable ones.
    edges = np.nanquantile(delta[covered], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    prof: dict[str, list[float]] = {"delta_mid": [], "entropy": [], "mi": [], "err": []}
    pred = (mean_p >= thresholds[None, :]).astype(np.int8)
    ent = np.nan_to_num(m["predictive_entropy"], nan=0.0)
    mi = np.nan_to_num(m["mutual_information"], nan=0.0)
    for i in range(n_bins):
        in_bin = covered & (delta >= edges[i]) & (delta < edges[i + 1])
        if not in_bin.any():
            continue
        g = gt[in_bin]
        valid = g >= 0
        prof["delta_mid"].append(float(np.nanmedian(delta[in_bin])))
        prof["entropy"].append(float(ent[in_bin].mean()))
        prof["mi"].append(float(mi[in_bin].mean()))
        prof["err"].append(float((pred[in_bin][valid] != g[valid]).mean()))
    coverage = {"n_images": cache.n_images, "n_covered": int(covered.sum())}
    return rows, {"profile": prof, "coverage": coverage}


def render_difficulty_alignment(
    rows: Sequence[dict[str, Any]],
    extras: dict[str, Any],
    out_path: str | Path,
    *,
    title: str,
) -> None:
    """Left: per-pathology Spearman bars. Middle/right: profiles per difficulty decile."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda r: r["spearman_predictive_entropy"])
    fig, (axL, axM, axR) = plt.subplots(
        1, 3, figsize=(15.5, max(5.2, 0.36 * len(ordered) + 1.8)),
        gridspec_kw={"width_ratios": (1.5, 1.0, 1.0)},
    )
    fig.patch.set_facecolor("#ffffff")
    _new_ax(fig, axL)
    ypos = np.arange(len(ordered), dtype=float)
    h = 0.38
    axL.barh(ypos + h / 2, [r["spearman_predictive_entropy"] for r in ordered], h,
             color=_COLORS[0], zorder=3, label="predictive_entropy")
    axL.barh(ypos - h / 2, [r["spearman_mutual_information"] for r in ordered], h,
             color=_COLORS[1], zorder=3, label="mutual_information")
    axL.axvline(0.0, color=_MUTED, lw=0.9, zorder=2)
    axL.set_yticks(ypos)
    axL.set_yticklabels([r["pathology"] for r in ordered], fontsize=7.5)
    axL.set_xlabel("Spearman rho with the disagreement of the 3 radiologists", fontsize=8.5)
    axL.set_title("model uncertainty vs human difficulty, per pathology",
                  loc="left", fontsize=10, fontweight="bold")
    axL.legend(fontsize=8, frameon=False, loc="lower right")

    prof = extras["profile"]
    _new_ax(fig, axM)
    axM.plot(prof["delta_mid"], prof["entropy"], marker="o", ms=4, lw=2.0,
             color=_COLORS[0], label="predictive_entropy")
    axM.plot(prof["delta_mid"], prof["mi"], marker="o", ms=4, lw=2.0,
             color=_COLORS[1], label="mutual_information")
    axM.set_xlabel("human difficulty (disagreement deciles)", fontsize=8.5)
    axM.set_ylabel("mean uncertainty", fontsize=8.5)
    axM.set_title("uncertainty profile", loc="left", fontsize=10, fontweight="bold")
    axM.legend(fontsize=8, frameon=False, loc="upper left")

    _new_ax(fig, axR)
    axR.plot(prof["delta_mid"], [100 * e for e in prof["err"]], marker="o", ms=4,
             lw=2.0, color=_COLORS[2])
    axR.set_xlabel("human difficulty (disagreement deciles)", fontsize=8.5)
    axR.set_ylabel("model error % (per-entry)", fontsize=8.5)
    axR.set_title("error profile", loc="left", fontsize=10, fontweight="bold")

    cov = extras["coverage"]
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.955,
             f"join by image id: {cov['n_covered']}/{cov['n_images']} images with "
             "known disagreement · cost-optimal thresholds 10:1",
             ha="left", va="top", fontsize=8.5, color=_MUTED)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    _save(fig, Path(out_path))
