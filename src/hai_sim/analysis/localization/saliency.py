"""Grad-CAM saliency for the TorchXRayVision DenseNet ensemble.

Implements the `SaliencyProvider` protocol: given a model-ready image and a label
index, return a ``(H, W)`` heatmap of where the model looked when calling that
pathology. This is the "where" companion to the classifier's "what", and the
input the localization validator compares against expert masks.

The heatmap is reconstructed **analytically** from the model's real-forward
feature maps and its linear head weights via `cam.cam_from_features` — the same
closed-form CAM math the offline cache evaluator uses, so neither path can drift
from the other's saliency reconstruction. (The xrv head is
``relu -> global-avg-pool -> linear``, which makes the Grad-CAM weights closed-form;
`test_cam_math` proves this equals backprop Grad-CAM bit for bit.) By default both
paths also score at the same geometry/precision — this live path at the model input
(224, float32) and `evaluate_from_cache` at the cache's ``mask_resolution`` (default
224) from float32 feature maps — so their metrics coincide; they differ only for a
cache deliberately built at a lower ``mask_resolution`` (see `evaluate`).

Ensemble aggregation: a cross-domain member not trained on the target pathology
carries no signal for it, so we reconstruct a CAM only for the covering members,
min-max normalize each, and average. If no member covers the label the map is all
NaN (the label is not localizable by this model) and the validator skips the
instance, keeping it out of every metric's denominator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hai_sim.analysis.localization.cam import bilinear_upsample, cam_from_features
from hai_sim.core.types import Image

if TYPE_CHECKING:
    from hai_sim.models.torchxrv import TorchXRayVisionAI


class GradCAMSaliency:
    """Analytic Grad-CAM `SaliencyProvider` over a `TorchXRayVisionAI` ensemble."""

    name = "grad_cam"

    def __init__(self, model: TorchXRayVisionAI) -> None:
        self.model = model

    def member_cams(self, image: Image, label_index: int) -> np.ndarray:
        """Per-member native-resolution Grad-CAMs, shape ``(M, h, w)``.

        Row ``m`` is member ``m``'s min-max-normalized CAM at the DenseNet's native
        spatial resolution (``7x7`` for a 224 input), or all-``NaN`` where member
        ``m`` was not trained on ``label_index``.
        """
        _probs, maps = self.model.member_feature_maps_and_probs(image)  # (M, L), (M, C, h, w)
        weights = self.model.member_classifier_weights()  # (M, L, C)
        rows: list[np.ndarray | None] = []
        native: tuple[int, ...] | None = None
        for m, mask in enumerate(self.model.member_masks):
            if not bool(mask[label_index]):
                rows.append(None)
                continue
            cam = cam_from_features(maps[m], weights[m, label_index])
            native = cam.shape
            rows.append(cam)
        if native is None:  # no member covers this label
            return np.full((len(rows), 1, 1), np.nan, dtype=np.float32)
        filled = [r if r is not None else np.full(native, np.nan, dtype=np.float32) for r in rows]
        return np.stack(filled).astype(np.float32)

    def heatmap(self, image: Image, label_index: int) -> np.ndarray:
        arr = np.asarray(image)
        h, w = int(arr.shape[-2]), int(arr.shape[-1])
        if h != w:
            raise ValueError(f"expected a square model input, got {h}x{w}")
        cams = self.member_cams(image, label_index)  # (M, native, native)
        if bool(np.isnan(cams).all()):
            # No member covers this label -> not localizable. Signal it with an
            # all-NaN map (not zeros): the validator then *skips* the instance, exactly
            # as `evaluate_from_cache` `continue`s on a label no member covers, so both
            # paths share one pointing-game denominator instead of scoring a spurious
            # zero map at pixel (0, 0).
            return np.full((h, w), np.nan, dtype=np.float32)
        agg = np.nanmean(cams, axis=0)  # (native, native)
        return bilinear_upsample(agg, h).astype(np.float32)
