"""Offline evaluation of a `LocalizationCache` — the cheap, swappable step.

This is the modular half of the split: a **pure function of the cache** (plus
metric choices), with no model and no torch. Change the metric set, the decision
threshold, the member aggregation, or the split, and the whole
`ComponentValidationReport` recomputes in seconds — the same way `measures_from_cache`
re-derives every uncertainty measure from cached probabilities.

- `per_instance_localization` reconstructs each localized finding's Grad-CAM from the
  cached feature maps + head weights (`cam.cam_from_features`), aggregates the members
  (``mean``/``max``), upsamples to the mask resolution and scores it against the expert
  mask — returning **one record per finding**, the raw material for both aggregation and
  cross-analyses such as localization-versus-uncertainty.
- `evaluate_from_cache` groups those into the per-pathology + pooled
  `ComponentValidationReport`, and adds the label block: the ensemble prediction is the
  nan-mean of the per-member probabilities thresholded at ``threshold``, scored against
  the expert ``ground_truth`` (``-1`` entries skipped).

Instances whose label no ensemble member covers are skipped entirely (every metric, not
just the overlap ones), so the pointing-game denominator never silently includes findings
the model cannot localize — matching what the live `LocalizationValidator` does (it skips
the all-NaN saliency map of a non-localizable label).

Fidelity note: by default this path matches the live validator exactly — it scores at
``cache.mask_resolution`` (default 224) from float32 feature maps, the same geometry and
precision the live validator uses (224, float32), with the same closed-form CAM math.
The two diverge only if a cache is deliberately built at a reduced ``mask_resolution``
(or an older float16 cache is reused), scoring the CAM at that lower fidelity.
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


def _ensemble_predicted(cache: LocalizationCache, threshold: float) -> np.ndarray:
    """``(N, L)`` int: ensemble mean probability thresholded (uncovered -> 0)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN member columns
        probs = np.nanmean(cache.member_probs, axis=1)  # (N, L)
    predicted: np.ndarray = (np.nan_to_num(probs, nan=0.0) >= threshold).astype(int)
    return predicted


def per_instance_localization(
    cache: LocalizationCache,
    *,
    metrics: Sequence[LocalizationMetric] | None = None,
    threshold: float = 0.5,
    member_aggregation: str = "mean",
    split: str | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """One record per localized finding: its Grad-CAM scored against the expert mask.

    Returns ``(predicted, records)`` where ``predicted`` is the ``(N, L)`` label
    matrix and each record is ``{image_row, label, label_name, predicted, scores}``.
    """
    metric_list = tuple(metrics) if metrics is not None else default_metrics()
    resolution = cache.mask_resolution
    predicted = _ensemble_predicted(cache, threshold)
    split_arr = np.asarray(cache.split)
    selected = np.ones(cache.n_images, dtype=bool) if split is None else (split_arr == split)
    aggregate = np.max if member_aggregation == "max" else np.mean

    # Group instances by image so each image's per-member feature map + positivity
    # fraction (both label-independent) are cast/computed once, not per finding.
    instances_by_image: dict[int, list[int]] = defaultdict(list)
    for k in range(cache.n_instances):
        instances_by_image[int(cache.inst_image[k])].append(k)

    records: list[dict[str, Any]] = []
    for image_row, instance_ks in instances_by_image.items():
        if not selected[image_row]:
            continue
        member_a = [np.asarray(cache.feature_maps[image_row, m], dtype=np.float32) for m in range(cache.n_members)]
        member_pos = [(a > 0).mean(axis=(1, 2)) for a in member_a]
        for k in instance_ks:
            label = int(cache.inst_label[k])
            member_cams = [
                cam_from_features(member_a[m], cache.classifier_weights[m, label], member_pos[m])
                for m in range(cache.n_members)
                if bool(cache.member_covers[m, label])
            ]
            if not member_cams:
                continue  # no member covers this label -> not localizable, skip
            cam = bilinear_upsample(aggregate(member_cams, axis=0), resolution)
            mask = np.asarray(cache.inst_mask[k], dtype=bool)
            records.append(
                {
                    "image_row": image_row,
                    "label": label,
                    "label_name": cache.pathologies[label],
                    "predicted": int(predicted[image_row, label]),
                    "scores": {m.name: float(m.score(cam, mask)) for m in metric_list},
                }
            )
    return predicted, records


def report_from_instances(
    cache: LocalizationCache,
    predicted: np.ndarray,
    instances: list[dict[str, Any]],
    *,
    metric_names: list[str],
    split: str | None = None,
    model_name: str = "cached",
    member_aggregation: str = "mean",
    threshold: float = 0.5,
) -> ComponentValidationReport:
    """Group per-instance records into a `ComponentValidationReport` (shared by the
    CLI evaluator and the dashboard, so both compute the Grad-CAMs only once)."""
    loc_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in instances:
        loc_by_image[rec["image_row"]].append(
            {"label": rec["label"], "predicted": rec["predicted"], "scores": rec["scores"]}
        )
    split_arr = np.asarray(cache.split)
    selected = np.ones(cache.n_images, dtype=bool) if split is None else (split_arr == split)
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
        cache.pathologies,
        metric_names,
        model=model_name,
        dataset=cache.dataset,
        saliency=f"grad_cam({member_aggregation})",
        decision_threshold=threshold,
    )


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
    predicted, instances = per_instance_localization(
        cache,
        metrics=metric_list,
        threshold=threshold,
        member_aggregation=member_aggregation,
        split=split,
    )
    return report_from_instances(
        cache,
        predicted,
        instances,
        metric_names=[m.name for m in metric_list],
        split=split,
        model_name=model_name,
        member_aggregation=member_aggregation,
        threshold=threshold,
    )
