"""Shared Grad-CAM math — the single implementation used by both the live
validator (`GradCAMSaliency`) and the offline cache evaluator (`evaluate_from_cache`).

Keeping this in one place is deliberate: the two paths must produce numerically
identical heatmaps, and having one function (rather than an autograd copy and an
analytic copy) removes the risk of silent divergence.

- `cam_from_features` reconstructs one label's Grad-CAM from a stored feature map
  and the linear head's weight row. For the xrv DenseNet head
  (``relu -> global-avg-pool -> linear``) the Grad-CAM channel weights are analytic
  (``alpha_c ∝ W[label, c] * fraction-of-channel-c-that-is-positive``), so
  ``CAM = ReLU(sum_c alpha_c * A_c)`` needs no autograd and equals backprop
  Grad-CAM bit for bit after min-max normalization (see `test_cam_math`).
- `bilinear_upsample` resizes a small CAM to the scoring resolution (numpy,
  align_corners=False).
"""

from __future__ import annotations

import numpy as np


def cam_from_features(
    feature_map: np.ndarray, weight_row: np.ndarray, pos_frac: np.ndarray | None = None
) -> np.ndarray:
    """One label's normalized Grad-CAM from a ``(C, h, w)`` feature map + ``(C,)`` head row.

    ``pos_frac`` (the per-channel positive fraction) is label-independent; pass a
    precomputed one to avoid recomputing it for every label of the same image.
    """
    a = np.asarray(feature_map, dtype=np.float32)
    if pos_frac is None:
        pos_frac = (a > 0).mean(axis=(1, 2))  # (C,) fraction of each channel that is positive
    alpha = np.asarray(weight_row, dtype=np.float32) * pos_frac  # (C,)
    cam = np.maximum(np.tensordot(alpha, a, axes=([0], [0])), 0.0)  # (h, w)
    lo = float(cam.min())
    hi = float(cam.max())
    if hi <= lo:
        return np.zeros_like(cam)
    normed: np.ndarray = (cam - lo) / (hi - lo)
    return normed


def bilinear_upsample(src: np.ndarray, size: int) -> np.ndarray:
    """Upsample a small 2D array to ``(size, size)``, torch-free (align_corners=False)."""
    src = np.asarray(src, dtype=np.float32)
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
