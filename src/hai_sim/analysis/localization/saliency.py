"""Grad-CAM saliency for the TorchXRayVision DenseNet ensemble.

Implements the `SaliencyProvider` protocol: given a model-ready image and a label
index, return a ``(H, W)`` heatmap of where the model looked when calling that
pathology. This is the "where" companion to the classifier's "what", and the
input the localization validator compares against expert masks.

Grad-CAM (Selvaraju et al.) on the DenseNet's last conv block: the xrv forward is
``features -> relu -> global-avg-pool -> classifier -> sigmoid -> op_norm``. We
tap the spatial ``features`` map ``(1, 1024, 7, 7)`` and take the gradient of the
**pre-sigmoid logit** (reconstructed from ``features`` + ``classifier``) w.r.t. it
— using the logit, not the squashed probability, avoids gradient saturation near
0/1. Channel-mean the gradients into weights, weight the activations, ReLU, and
bilinear-upsample to the input geometry.

Ensemble aggregation: a cross-domain member that was not trained on the target
pathology carries no signal for it (xrv blanks that output), so we Grad-CAM only
the members whose coverage mask includes the label, min-max normalize each to make
them commensurable, and average. If no member covers the label the map is all
zeros (the label is not localizable by this model).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from hai_sim.core.types import Image

if TYPE_CHECKING:
    from hai_sim.models.torchxrv import TorchXRayVisionAI


class GradCAMSaliency:
    """Grad-CAM `SaliencyProvider` over a `TorchXRayVisionAI` ensemble."""

    name = "grad_cam"

    def __init__(self, model: TorchXRayVisionAI) -> None:
        self.model = model
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                'GradCAMSaliency needs the [models] extra: uv pip install -e ".[models]"'
            ) from exc
        self._torch = torch

    def heatmap(self, image: Image, label_index: int) -> np.ndarray:
        torch = self._torch
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:  # (H, W) -> (1, H, W)
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"expected a (1, H, W) model-ready image, got shape {arr.shape}")
        h, w = arr.shape[-2], arr.shape[-1]
        x = torch.from_numpy(arr).unsqueeze(0).to(self.model.device)  # (1, 1, H, W)

        cams: list[np.ndarray] = []
        for member, mask in zip(self.model.members, self.model.member_masks, strict=True):
            if not bool(mask[label_index]):
                continue
            cams.append(self._grad_cam(member, x, label_index, (h, w)))

        if not cams:
            return np.zeros((h, w), dtype=np.float32)
        mean_cam: np.ndarray = np.mean(cams, axis=0).astype(np.float32)
        return mean_cam

    def _grad_cam(
        self, module: Any, x: Any, label_index: int, size: tuple[int, int]
    ) -> np.ndarray:
        torch = self._torch
        import torch.nn.functional as F

        with torch.enable_grad():
            feats = module.features(x)  # (1, C, h7, w7), spatial map
            feats.retain_grad()
            # Reconstruct the xrv head up to the logit (features2 = relu+avgpool+flatten).
            pooled = F.adaptive_avg_pool2d(F.relu(feats, inplace=False), (1, 1))
            logits = module.classifier(torch.flatten(pooled, 1))  # (1, L), pre-sigmoid
            module.zero_grad(set_to_none=True)
            logits[0, label_index].backward()

        activations = feats.detach()  # (1, C, h7, w7)
        gradients = feats.grad.detach()  # (1, C, h7, w7)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))  # (1, 1, h7, w7)
        cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
        out: np.ndarray = cam[0, 0].cpu().numpy().astype(np.float32)

        lo = float(out.min())
        hi = float(out.max())
        if hi <= lo:
            return np.zeros(size, dtype=np.float32)
        normed: np.ndarray = (out - lo) / (hi - lo)
        return normed
