"""Typed results of a component validation run: the label block + the localization block.

Two independent views of one model against expert ground truth, kept side by side
so a single run answers both "does it call the right label?" and "does it look at
the right place?" — decomposed per pathology, in the spirit of `metrics.kpi`.

- `LabelConfusion`  : per-pathology TP/FN/FP/TN vs expert image-level labels.
- `MetricAggregate` : one localization metric's nan-aware mean over instances,
  with the count of instances that actually contributed (undefined scores, e.g. a
  blank heatmap under an overlap metric, are dropped, not counted as zero).
- `ComponentValidationReport` : the whole thing, JSON-serializable via `to_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LabelConfusion:
    """Per-pathology confusion counts vs expert labels (``-1`` GT entries excluded)."""

    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0

    def update(self, gt: int, pred: int) -> None:
        if gt == 1:
            self.tp += int(pred == 1)
            self.fn += int(pred == 0)
        elif gt == 0:
            self.fp += int(pred == 1)
            self.tn += int(pred == 0)
        # gt == -1: not annotated -> ignored

    @property
    def n_pos(self) -> int:
        return self.tp + self.fn

    @property
    def n_neg(self) -> int:
        return self.tn + self.fp

    @property
    def sensitivity(self) -> float | None:
        return self.tp / self.n_pos if self.n_pos else None

    @property
    def specificity(self) -> float | None:
        return self.tn / self.n_neg if self.n_neg else None

    def to_dict(self) -> dict[str, object]:
        return {
            "tp": self.tp,
            "fn": self.fn,
            "fp": self.fp,
            "tn": self.tn,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
        }


@dataclass(frozen=True)
class MetricAggregate:
    """A localization metric's nan-aware mean and how many instances it covered."""

    metric: str
    mean: float | None
    n_valid: int
    n_total: int

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "mean": self.mean,
            "n_valid": self.n_valid,
            "n_total": self.n_total,
        }


def aggregate_scores(metric: str, scores: list[float]) -> MetricAggregate:
    """Nan-aware mean of one metric's per-instance scores (``None`` if all invalid)."""
    arr = np.asarray(scores, dtype=float)
    valid = arr[~np.isnan(arr)] if arr.size else arr
    mean = float(valid.mean()) if valid.size else None
    return MetricAggregate(metric=metric, mean=mean, n_valid=int(valid.size), n_total=int(arr.size))


@dataclass(frozen=True)
class ComponentValidationReport:
    """Dual-ground-truth validation of one model on one localization dataset."""

    model: str
    dataset: str
    saliency: str | None
    decision_threshold: float | None
    schema_labels: tuple[str, ...]
    n_images: int
    label_confusion: dict[str, LabelConfusion]
    localization_per_label: dict[str, list[MetricAggregate]] = field(default_factory=dict)
    localization_pooled: list[MetricAggregate] = field(default_factory=list)
    localization_pooled_predicted: list[MetricAggregate] = field(default_factory=list)

    @property
    def macro_sensitivity(self) -> float | None:
        vals = [c.sensitivity for c in self.label_confusion.values() if c.sensitivity is not None]
        return float(np.mean(vals)) if vals else None

    @property
    def macro_specificity(self) -> float | None:
        vals = [c.specificity for c in self.label_confusion.values() if c.specificity is not None]
        return float(np.mean(vals)) if vals else None

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "saliency": self.saliency,
            "decision_threshold": self.decision_threshold,
            "n_images": self.n_images,
            "labels": {
                "macro_sensitivity": self.macro_sensitivity,
                "macro_specificity": self.macro_specificity,
                "per_pathology": {k: v.to_dict() for k, v in self.label_confusion.items()},
            },
            "localization": {
                "pooled": [m.to_dict() for m in self.localization_pooled],
                "pooled_predicted_positive": [
                    m.to_dict() for m in self.localization_pooled_predicted
                ],
                "per_pathology": {
                    k: [m.to_dict() for m in v] for k, v in self.localization_per_label.items()
                },
            },
        }
