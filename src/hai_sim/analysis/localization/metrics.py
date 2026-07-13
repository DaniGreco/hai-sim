"""Localization metrics: agreement between a saliency heatmap and an expert mask.

Each metric scores a single ``(heatmap, gt_mask)`` pair, both in the *same*
geometry (the model input, e.g. ``224x224``). Higher = better spatial agreement;
the built-ins return a scalar in ``[0, 1]``. Aggregation across instances and per
pathology is the validator's job, not the metric's.

Undefined cases return ``NaN`` so the validator can average them out with a
nan-aware reduction instead of silently biasing the score:

- empty ``gt_mask`` (the label is not localized in this image) -> ``NaN``;
- flat / all-zero ``heatmap`` where the metric needs contrast -> ``NaN``.

Two families, deliberately kept together:

- **Threshold-free**: ``PointingGame`` (does the single hottest pixel land inside
  the mask?) and ``EnergyInMask`` (what fraction of the heatmap's positive energy
  falls inside the mask?). Robust, no knob to tune.
- **Threshold-based**: ``IoU`` and ``Dice`` binarize the min-max-normalized
  heatmap at ``threshold`` before the set overlap. The threshold is explicit so a
  run records exactly how the heatmap was cut.
"""

from __future__ import annotations

import numpy as np


def _minmax(heatmap: np.ndarray) -> np.ndarray | None:
    """Min-max normalize to ``[0, 1]``; ``None`` if the heatmap has no contrast."""
    h = np.asarray(heatmap, dtype=float)
    lo = float(np.min(h))
    hi = float(np.max(h))
    if hi <= lo:
        return None
    return (h - lo) / (hi - lo)


class PointingGame:
    """Pointing-game hit: ``1.0`` if the heatmap's argmax lands on (or within
    ``tolerance`` pixels of) the mask.

    The classic weakly-supervised-localization check (Zhang et al.): threshold-free
    and forgiving of a diffuse map, it asks only whether the single most-attributed
    point falls on the finding. Averaged over instances it becomes the hit-rate.

    ``tolerance`` is a Chebyshev radius in pixels. ``0`` is the strict exact-pixel
    protocol (CheXlocalize hit-rate). A small tolerance matters for coarse CAMs: a
    Grad-CAM upsampled from a 7x7 grid quantizes its peak to cell centers (~32 px
    apart on a 224 image), so exact-pixel hits on tight polygon masks are nearly a
    lottery; Zhang's original protocol uses ~15 px. Set ``tolerance`` to about half
    the CAM cell size to score the metric at the resolution the map actually has.
    """

    def __init__(self, tolerance: int = 0) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        self.tolerance = int(tolerance)
        self.name = "pointing_game" if tolerance == 0 else f"pointing_game@{tolerance}px"

    def score(self, heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
        gt = np.asarray(gt_mask, dtype=bool)
        if not gt.any():
            return float("nan")
        h = np.asarray(heatmap, dtype=float)
        # A flat / all-zero / all-NaN heatmap has no distinguishable peak: np.argmax
        # would fall back to the top-left pixel, scoring a spurious hit (if the mask
        # happens to cover that corner) or miss. Undefined, exactly like the overlap
        # metrics on a no-contrast map -> NaN (see module docstring). This also keeps
        # the pointing-game denominator aligned with the offline evaluator, which drops
        # a label no model member can localize instead of counting it as a miss.
        if not np.isfinite(h).any() or float(np.nanmax(h)) <= float(np.nanmin(h)):
            return float("nan")
        r, c = (int(v) for v in np.unravel_index(int(np.nanargmax(h)), h.shape))
        if self.tolerance == 0:
            return 1.0 if bool(gt[r, c]) else 0.0
        t = self.tolerance
        r0, r1 = max(r - t, 0), min(r + t + 1, gt.shape[0])
        c0, c1 = max(c - t, 0), min(c + t + 1, gt.shape[1])
        return 1.0 if bool(gt[r0:r1, c0:c1].any()) else 0.0


class EnergyInMask:
    """Fraction of the heatmap's positive energy that falls inside the mask.

    Threshold-free and continuous: ``1.0`` = all attribution lands on the finding,
    ``0.0`` = none. Negative attribution is clipped away first. A random/flat map
    scores ~ the mask's area fraction, which is the natural chance baseline.
    """

    name = "energy_in_mask"

    def score(self, heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
        gt = np.asarray(gt_mask, dtype=bool)
        if not gt.any():
            return float("nan")
        h = np.clip(np.asarray(heatmap, dtype=float), 0.0, None)
        total = float(h.sum())
        if total <= 0.0:
            return float("nan")
        return float(h[gt].sum() / total)


class IoU:
    """Intersection-over-Union after binarizing the normalized heatmap at ``threshold``."""

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold
        self.name = f"iou@{threshold:g}"

    def score(self, heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
        gt = np.asarray(gt_mask, dtype=bool)
        if not gt.any():
            return float("nan")
        norm = _minmax(heatmap)
        if norm is None:
            return float("nan")
        pred = norm >= self.threshold
        inter = int(np.logical_and(pred, gt).sum())
        union = int(np.logical_or(pred, gt).sum())
        return float(inter / union) if union else float("nan")


class Dice:
    """Dice / F1 overlap after binarizing the normalized heatmap at ``threshold``."""

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold
        self.name = f"dice@{threshold:g}"

    def score(self, heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
        gt = np.asarray(gt_mask, dtype=bool)
        if not gt.any():
            return float("nan")
        norm = _minmax(heatmap)
        if norm is None:
            return float("nan")
        pred = norm >= self.threshold
        inter = int(np.logical_and(pred, gt).sum())
        denom = int(pred.sum()) + int(gt.sum())
        return float(2.0 * inter / denom) if denom else float("nan")


def default_metrics() -> tuple[PointingGame, EnergyInMask, IoU, Dice]:
    """The standard localization-metric suite used by the validator by default."""
    return (PointingGame(), EnergyInMask(), IoU(0.5), Dice(0.5))


__all__ = ["Dice", "EnergyInMask", "IoU", "PointingGame", "default_metrics"]
