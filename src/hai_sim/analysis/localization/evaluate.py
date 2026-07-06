"""Offline evaluation of a `LocalizationCache` — the cheap, swappable step.

This is the modular half of the split: a **pure function of the cache** (plus
metric choices), with no model and no torch. Change the metric set, the decision
threshold, the member aggregation, or the split, and the whole
`ComponentValidationReport` recomputes in seconds — the same way `measures_from_cache`
re-derives every uncertainty measure from cached probabilities.

- **labels**: the ensemble prediction is the nan-mean of the per-member
  probabilities thresholded at ``threshold``; confusion is scored vs the expert
  ``ground_truth`` (``-1`` entries skipped).
- **localization**: each instance's per-member 7x7 Grad-CAMs are aggregated
  (``mean``/``max``), bilinearly upsampled to the mask resolution, and scored by
  each metric against the co-registered expert mask.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from hai_sim.analysis.localization.metrics import default_metrics
from hai_sim.analysis.localization.report import ComponentValidationReport, assemble_report

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hai_sim.analysis.localization.cache import LocalizationCache
    from hai_sim.core.protocols import LocalizationMetric


def cam_from_features(feature_map: np.ndarray, weight_row: np.ndarray) -> np.ndarray:
    """Reconstruct one label's Grad-CAM from a stored feature map + head weights.

    For the xrv DenseNet head (``relu -> global-avg-pool -> linear``) the Grad-CAM
    channel weights are analytic: ``alpha_c ∝ W[label, c] * (fraction of channel c
    that is positive)``. So ``CAM = ReLU(sum_c alpha_c * A_c)`` needs no autograd —
    it matches the backprop Grad-CAM bit for bit after min-max normalization. This
    is what lets the cache store raw feature maps and derive *any* label's map
    offline. ``feature_map`` is ``(C, h, w)``, ``weight_row`` is ``(C,)``.
    """
    A = np.asarray(feature_map, dtype=np.float32)
    pos_frac = (A > 0).mean(axis=(1, 2))  # (C,)
    alpha = np.asarray(weight_row, dtype=np.float32) * pos_frac  # (C,)
    cam = np.maximum(np.tensordot(alpha, A, axes=([0], [0])), 0.0)  # (h, w)
    lo = float(cam.min())
    hi = float(cam.max())
    if hi <= lo:
        return np.zeros_like(cam)
    normed: np.ndarray = (cam - lo) / (hi - lo)
    return normed


def _bilinear_upsample(src: np.ndarray, size: int) -> np.ndarray:
    """Upsample a small 2D array to ``(size, size)``, torch-free (align_corners=False)."""
    h, w = src.shape
    ys = np.clip((np.arange(size) + 0.5) * h / size - 0.5, 0.0, h - 1)
    xs = np.clip((np.arange(size) + 0.5) * w / size - 0.5, 0.0, w - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = src[np.ix_(y0, x0)] * (1 - wx) + src[np.ix_(y0, x1)] * wx
    bot = src[np.ix_(y1, x0)] * (1 - wx) + src[np.ix_(y1, x1)] * wx
    out: np.ndarray = top * (1 - wy) + bot * wy
    return out


def evaluate_from_cache(
    cache: LocalizationCache,
    *,
    metrics: Sequence[LocalizationMetric] | None = None,
    threshold: float = 0.5,
    member_aggregation: str = "mean",
    split: str | None = None,
    model_name: str = "cached",
) -> ComponentValidationReport:
    """Derive a `ComponentValidationReport` from a cache without touching the model."""
    metric_list = tuple(metrics) if metrics is not None else default_metrics()
    metric_names = [m.name for m in metric_list]
    labels = cache.pathologies
    resolution = cache.mask_resolution

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN member columns
        probs = np.nanmean(cache.member_probs, axis=1)  # (N, L) ensemble mean prob
    predicted = (np.nan_to_num(probs, nan=0.0) >= threshold).astype(int)  # (N, L)

    split_arr = np.asarray(cache.split)
    selected = np.ones(cache.n_images, dtype=bool) if split is None else (split_arr == split)
    aggregate = np.max if member_aggregation == "max" else np.mean

    loc_by_image: dict[int, list[dict[str, Any]]] = {}
    for k in range(cache.n_instances):
        image_row = int(cache.inst_image[k])
        if not selected[image_row]:
            continue
        label = int(cache.inst_label[k])
        # Reconstruct each covering member's Grad-CAM offline from its feature map
        # and head weights (analytic, no autograd), then aggregate over members.
        member_cams = [
            cam_from_features(cache.feature_maps[image_row, m], cache.classifier_weights[m, label])
            for m in range(cache.n_members)
            if bool(cache.member_covers[m, label])
        ]
        cam_small = aggregate(member_cams, axis=0) if member_cams else np.zeros((7, 7), np.float32)
        cam = _bilinear_upsample(cam_small, resolution)
        mask = np.asarray(cache.inst_mask[k], dtype=bool)
        scores = {m.name: float(m.score(cam, mask)) for m in metric_list}
        loc_by_image.setdefault(image_row, []).append(
            {"label": label, "predicted": int(predicted[image_row, label]), "scores": scores}
        )

    records: list[dict[str, Any]] = []
    for image_row in range(cache.n_images):
        if not selected[image_row]:
            continue
        records.append(
            {
                "gt": [int(x) for x in cache.ground_truth[image_row]],
                "pred": [int(x) for x in predicted[image_row]],
                "loc": loc_by_image.get(image_row, []),
            }
        )

    return assemble_report(
        records,
        labels,
        metric_names,
        model=model_name,
        dataset=cache.dataset,
        saliency=f"grad_cam({member_aggregation})",
        decision_threshold=threshold,
    )
