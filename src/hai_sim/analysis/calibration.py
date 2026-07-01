"""Per-pathology decision-threshold calibration from the inference cache.

The decision label is ``mean_p >= tau`` per pathology. A flat ``tau = 0.5`` is the
wrong operating point (xrv's own per-pathology op_threshs are ~0.01-0.1), and the
recall/FP trade-off is bounded by each pathology's separability (ROC AUC), not by
the threshold. This module makes that trade-off explicit and picks per-pathology
thresholds two ways:

- **recall-target (FN-safe)**: the highest ``tau`` whose recall >= ``target_recall``
  on the calibration split — keeps misses low, minimizes FP at that recall.
- **cost-weighted (Borsani-style)**: ``argmin_tau w_fn * FN + w_fp * FP``.

Thresholds are fit on a calibration split and *evaluated on a held-out test split*
so the reported FP/recall are not optimistic. Everything is derived from cached
ensemble mean probabilities — no model is run here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hai_sim.analysis.error_curve import _auroc
from hai_sim.analysis.inference_cache import InferenceCache, measures_from_cache


@dataclass(frozen=True)
class PathologyCalibration:
    """Per-pathology calibration result, metrics measured on the test split."""

    name: str
    n_pos: int
    n_neg: int
    auc: float
    # baseline operating point tau = 0.5
    recall_05: float
    fp_05: int
    precision_05: float
    # recall-target operating point
    tau_recall: float
    recall_rt: float
    fp_rt: int
    precision_rt: float
    # cost-weighted operating point
    tau_cost: float
    recall_cost: float
    fp_cost: int
    fn_cost: int
    precision_cost: float


def threshold_for_recall(pos_scores: np.ndarray, target_recall: float) -> float:
    """Highest threshold whose recall on ``pos_scores`` is >= ``target_recall``."""
    pos = np.asarray(pos_scores, dtype=float)
    n = pos.shape[0]
    if n == 0:
        return 0.0
    m = int(np.ceil(target_recall * n))
    m = max(1, min(n, m))  # need at least m positives at/above tau
    return float(np.sort(pos)[::-1][m - 1])


def cost_optimal_threshold(
    pos_scores: np.ndarray, neg_scores: np.ndarray, w_fn: float, w_fp: float
) -> float:
    """Threshold minimizing ``w_fn * FN + w_fp * FP`` over candidate cut points."""
    pos = np.sort(np.asarray(pos_scores, dtype=float))
    neg = np.sort(np.asarray(neg_scores, dtype=float))
    if pos.size == 0 and neg.size == 0:
        return 0.5
    # Candidates: every observed score, plus one above the max (predict-none option).
    cands = np.unique(np.concatenate([pos, neg, [1.0 + 1e-6]]))
    fn = np.searchsorted(pos, cands, side="left")  # #positives below tau
    fp = neg.size - np.searchsorted(neg, cands, side="left")  # #negatives at/above tau
    cost = w_fn * fn + w_fp * fp
    return float(cands[int(np.argmin(cost))])


def _metrics_at(
    pos: np.ndarray, neg: np.ndarray, tau: float
) -> tuple[float, int, float, int]:
    """Return (recall, FP, precision, FN) at threshold tau on test pos/neg scores."""
    tp = int((pos >= tau).sum())
    fn = int((pos < tau).sum())
    fp = int((neg >= tau).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return recall, fp, precision, fn


def roc_points(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(FPR, TPR) sweep over thresholds, sorted by FPR — for plotting the ROC."""
    scores = np.unique(np.concatenate([pos, neg]))
    taus = np.concatenate([[-np.inf], scores, [np.inf]])
    tpr = np.array([(pos >= t).mean() if pos.size else 0.0 for t in taus])
    fpr = np.array([(neg >= t).mean() if neg.size else 0.0 for t in taus])
    idx = np.argsort(fpr, kind="stable")
    return fpr[idx], tpr[idx]


def calibrate(
    cache: InferenceCache,
    *,
    split_seed: int = 0,
    test_frac: float = 0.5,
    target_recall: float = 0.95,
    w_fn: float = 1_000_000.0,
    w_fp: float = 500.0,
) -> list[PathologyCalibration]:
    """Fit per-pathology thresholds on a calibration split, evaluate on the test split."""
    mean_p = np.nan_to_num(measures_from_cache(cache.member_probs)["mean_p"], nan=0.0)
    gt = cache.ground_truth
    n = cache.n_images

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_test = round(test_frac * n)
    test_idx = perm[:n_test]
    cal_idx = perm[n_test:]

    results: list[PathologyCalibration] = []
    for k, name in enumerate(cache.pathologies):
        cal_pos = mean_p[cal_idx, k][gt[cal_idx, k] == 1]
        cal_neg = mean_p[cal_idx, k][gt[cal_idx, k] == 0]
        test_pos = mean_p[test_idx, k][gt[test_idx, k] == 1]
        test_neg = mean_p[test_idx, k][gt[test_idx, k] == 0]
        if cal_pos.size == 0 or test_pos.size == 0:
            continue  # not enough signal to calibrate / evaluate this pathology

        tau_recall = threshold_for_recall(cal_pos, target_recall)
        tau_cost = cost_optimal_threshold(cal_pos, cal_neg, w_fn, w_fp)

        labels = np.concatenate(
            [np.ones(test_pos.size, dtype=bool), np.zeros(test_neg.size, dtype=bool)]
        )
        auc = _auroc(np.concatenate([test_pos, test_neg]), labels)

        r05, fp05, p05, _ = _metrics_at(test_pos, test_neg, 0.5)
        rrt, fprt, prt, _ = _metrics_at(test_pos, test_neg, tau_recall)
        rc, fpc, pc, fnc = _metrics_at(test_pos, test_neg, tau_cost)

        results.append(
            PathologyCalibration(
                name=name,
                n_pos=int(test_pos.size),
                n_neg=int(test_neg.size),
                auc=auc,
                recall_05=r05,
                fp_05=fp05,
                precision_05=p05,
                tau_recall=tau_recall,
                recall_rt=rrt,
                fp_rt=fprt,
                precision_rt=prt,
                tau_cost=tau_cost,
                recall_cost=rc,
                fp_cost=fpc,
                fn_cost=fnc,
                precision_cost=pc,
            )
        )
    return results


def render_calibration(
    cache: InferenceCache,
    out_path: str | Path,
    results: list[PathologyCalibration],
    *,
    split_seed: int = 0,
    test_frac: float = 0.5,
) -> None:
    """Per-pathology ROC small multiples with the three operating points marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_p = np.nan_to_num(measures_from_cache(cache.member_probs)["mean_p"], nan=0.0)
    gt = cache.ground_truth
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(cache.n_images)
    test_idx = perm[: round(test_frac * cache.n_images)]
    path_index = {name: k for k, name in enumerate(cache.pathologies)}

    n = len(results)
    ncols = 4
    nrows = max(1, -(-n // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for i, r in enumerate(results):
        k = path_index[r.name]
        pos = mean_p[test_idx, k][gt[test_idx, k] == 1]
        neg = mean_p[test_idx, k][gt[test_idx, k] == 0]
        fpr, tpr = roc_points(pos, neg)
        ax = axes[i // ncols][i % ncols]
        ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1.0)
        ax.plot(fpr, tpr, color="green", lw=1.7)
        # operating points: FPR/TPR at each tau (TPR=recall, FPR=fp/n_neg)
        for tau, col, lab in (
            (0.5, "black", "0.5"),
            (r.tau_recall, "red", "recall"),
            (r.tau_cost, "blue", "cost"),
        ):
            fx = (neg >= tau).mean() if neg.size else 0.0
            fy = (pos >= tau).mean() if pos.size else 0.0
            ax.scatter([fx], [fy], color=col, s=22, zorder=5, label=lab)
        ax.set_title(f"{r.name}\nAUC={r.auc:.2f} (P={r.n_pos})", fontsize=8)
        ax.set_xlabel("FPR", fontsize=7)
        ax.set_ylabel("TPR (recall)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="lower right")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(f"Per-pathology ROC + operating points  |  N={cache.n_images}", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
