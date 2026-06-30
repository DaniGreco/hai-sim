"""Tests for per-pathology threshold calibration."""

from __future__ import annotations

import numpy as np

from hai_sim.analysis.calibration import (
    _metrics_at,
    calibrate,
    cost_optimal_threshold,
    threshold_for_recall,
)
from hai_sim.analysis.inference_cache import InferenceCache
from hai_sim.core.types import LabelSchema


def test_threshold_for_recall() -> None:
    pos = np.array([0.1, 0.2, 0.3, 0.9])
    assert threshold_for_recall(pos, 1.0) == 0.1  # catch all
    assert threshold_for_recall(pos, 0.5) == 0.3  # top 2 of 4
    assert threshold_for_recall(pos, 0.75) == 0.2  # top 3 of 4


def test_cost_optimal_direction() -> None:
    pos = np.array([0.4, 0.6])
    neg = np.array([0.5, 0.55])
    # FN heavily penalized → low threshold → high recall
    tau_fn = cost_optimal_threshold(pos, neg, w_fn=1e6, w_fp=1.0)
    # FP heavily penalized → high threshold → low recall
    tau_fp = cost_optimal_threshold(pos, neg, w_fn=1.0, w_fp=1e6)
    recall_fn = _metrics_at(pos, neg, tau_fn)[0]
    recall_fp = _metrics_at(pos, neg, tau_fp)[0]
    assert recall_fn >= recall_fp


def _cache_with_signal(n: int = 200) -> InferenceCache:
    rng = np.random.default_rng(0)
    schema = LabelSchema.xrv()
    gt = rng.choice([1, 0], size=(n, 18)).astype(np.int8)
    gt[:, 14:] = -1  # unannotated (NIH-like)
    # member probs correlated with gt so AUC > 0.5
    base = np.where(gt == 1, 0.55, 0.45).astype(np.float32)
    mp = np.clip(base[:, None, :] + rng.normal(0, 0.1, (n, 4, 18)), 0, 1).astype(np.float32)
    return InferenceCache(
        member_probs=mp,
        ground_truth=gt,
        inference_seconds=np.zeros(n),
        image_ids=tuple(str(i) for i in range(n)),
        dataset_indices=np.arange(n, dtype=np.int64),
        weights=("a", "b", "c", "d"),
        pathologies=tuple(schema.labels),
        dataset="nih",
        batch_size=1,
    )


def test_calibrate_runs_on_split() -> None:
    cache = _cache_with_signal()
    results = calibrate(cache, target_recall=0.9, w_fn=1e6, w_fp=500.0)
    assert results  # annotated pathologies present
    names = LabelSchema.xrv().labels
    for r in results:
        assert r.name in names[:14]  # only annotated labels
        assert 0.0 <= r.auc <= 1.0
        assert 0.0 <= r.recall_rt <= 1.0
        assert 0.0 <= r.tau_recall <= 1.0
        # cost with w_fn >> w_fp lowers tau → recall at least as high as baseline 0.5
        assert r.recall_cost >= r.recall_05 - 1e-9
