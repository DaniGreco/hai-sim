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
        h, w = self._input_hw(image)
        cams = self.member_cams(image, label_index)  # (M, h7, w7), NaN rows if uncovered
        if bool(np.isnan(cams).all()):
            return np.zeros((h, w), dtype=np.float32)
        # nanmean over members == mean over the covering ones; upsampling is linear,
        # so mean-then-upsample here equals the old upsample-then-mean, bit for bit.
        agg = np.nanmean(cams, axis=0)  # (h7, w7)
        up: np.ndarray = self._upsample(agg, (h, w)).astype(np.float32)
        return up

    def member_cams(self, image: Image, label_index: int) -> np.ndarray:
        """Per-member native-resolution Grad-CAMs, shape ``(M, h7, w7)``.

        Row ``m`` is member ``m``'s min-max-normalized CAM at the DenseNet's native
        spatial resolution (``7x7`` for a 224 input), or all-``NaN`` where member
        ``m`` was not trained on ``label_index``. This is the raw per-member
        material the localization cache stores: keeping members separate (rather
        than the fused heatmap) lets the aggregation (mean/max), the threshold and
        every metric stay swappable offline.
        """
        x = self._to_tensor(image)
        rows: list[np.ndarray | None] = []
        native: tuple[int, ...] | None = None
        for member, mask in zip(self.model.members, self.model.member_masks, strict=True):
            if not bool(mask[label_index]):
                rows.append(None)
                continue
            cam = self._member_cam(member, x, label_index)
            native = cam.shape
            rows.append(cam)
        if native is None:  # no member covers this label
            return np.full((len(rows), 1, 1), np.nan, dtype=np.float32)
        filled = [r if r is not None else np.full(native, np.nan, dtype=np.float32) for r in rows]
        return np.stack(filled).astype(np.float32)

    def _to_tensor(self, image: Image) -> Any:
        torch = self._torch
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:  # (H, W) -> (1, H, W)
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"expected a (1, H, W) model-ready image, got shape {arr.shape}")
        return torch.from_numpy(arr).unsqueeze(0).to(self.model.device)  # (1, 1, H, W)

    @staticmethod
    def _input_hw(image: Image) -> tuple[int, int]:
        arr = np.asarray(image)
        return int(arr.shape[-2]), int(arr.shape[-1])

    def _member_cam(self, module: Any, x: Any, label_index: int) -> np.ndarray:
        """One member's Grad-CAM at native resolution, min-max normalized to [0, 1]."""
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
        out: np.ndarray = cam[0, 0].cpu().numpy().astype(np.float32)

        lo = float(out.min())
        hi = float(out.max())
        if hi <= lo:
            return np.zeros_like(out)
        normed: np.ndarray = (out - lo) / (hi - lo)
        return normed

    def _upsample(self, cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        torch = self._torch
        import torch.nn.functional as F

        t = torch.from_numpy(np.asarray(cam, dtype=np.float32))[None, None]  # (1,1,h7,w7)
        up = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
        out: np.ndarray = up[0, 0].numpy()
        return out
