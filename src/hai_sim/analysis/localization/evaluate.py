"""Offline evaluation of a `LocalizationCache` — the cheap, swappable step.

This is the modular half of the split: a **pure function of the cache** (plus
metric choices), with no model and no torch. Change the metric set, the decision
threshold, the member aggregation, or the split, and the whole
`ComponentValidationReport` recomputes in seconds — the same way `measures_from_cache`
re-derives every uncertainty measure from cached probabilities.

- **labels**: the ensemble prediction is the nan-mean of the per-member
  probabilities thresholded at ``threshold``; confusion is scored vs the expert
  ``ground_truth`` (``-1`` entries skipped).
- **localization**: each instance's per-member Grad-CAM is reconstructed from the
  cached feature maps + head weights (`cam.cam_from_features`), aggregated
  (``mean``/``max``), upsampled to the mask resolution, and scored by each metric.
  Instances whose label no ensemble member covers are skipped entirely (every
  metric, not just the overlap ones), so the pointing-game denominator never
  silently includes findings the model cannot localize.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from hai_sim.analysis.localization.cam import bilinear_upsample, cam_from_features
from hai_sim.analysis.localization.metrics import default_metrics
from hai_sim.analysis.localization.report import ComponentValidationReport, assemble_report

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hai_sim.analysis.localization.cache import LocalizationCache
    from hai_sim.core.protocols import LocalizationMetric


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

    # Group instances by image so each image's per-member feature map + positivity
    # fraction (both label-independent) are cast/computed once, not per finding.
    instances_by_image: dict[int, list[int]] = defaultdict(list)
    for k in range(cache.n_instances):
        instances_by_image[int(cache.inst_image[k])].append(k)

    loc_by_image: dict[int, list[dict[str, Any]]] = {}
    for image_row, instance_ks in instances_by_image.items():
        if not selected[image_row]:
            continue
        member_a: list[np.ndarray] = []
        member_pos: list[np.ndarray] = []
        for m in range(cache.n_members):
            a = np.asarray(cache.feature_maps[image_row, m], dtype=np.float32)
            member_a.append(a)
            member_pos.append((a > 0).mean(axis=(1, 2)))

        for k in instance_ks:
            label = int(cache.inst_label[k])
            member_cams = [
                cam_from_features(member_a[m], cache.classifier_weights[m, label], member_pos[m])
                for m in range(cache.n_members)
                if bool(cache.member_covers[m, label])
            ]
            if not member_cams:
                continue  # no member covers this label -> not localizable, skip entirely
            cam = bilinear_upsample(aggregate(member_cams, axis=0), resolution)
            mask = np.asarray(cache.inst_mask[k], dtype=bool)
            scores = {mtr.name: float(mtr.score(cam, mask)) for mtr in metric_list}
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
