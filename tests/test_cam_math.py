"""The analytic Grad-CAM (from a feature map + linear head) must equal autograd.

This is the load-bearing equivalence for the feature-map cache: if it holds, the
cache can store raw feature maps and reconstruct any label's Grad-CAM offline with
no backprop. The core test is confound-free (a synthetic feature map + linear head
vs torch autograd on the same tensors). A second test checks the model exposes the
real-forward feature maps. Skipped without the [models] extra.
"""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.analysis.localization.evaluate import cam_from_features

pytest.importorskip("torch")


def _autograd_cam(a_np: np.ndarray, w_row: np.ndarray) -> np.ndarray:
    """Reference Grad-CAM via real autograd on head ``relu -> avgpool -> linear``."""
    import torch
    import torch.nn.functional as F

    a = torch.tensor(a_np, dtype=torch.float32, requires_grad=True)  # (C, h, w)
    pooled = F.adaptive_avg_pool2d(F.relu(a).unsqueeze(0), (1, 1)).flatten(1)  # (1, C)
    logit = (pooled * torch.tensor(w_row, dtype=torch.float32)).sum()
    logit.backward()
    weights = a.grad.mean(dim=(1, 2))  # (C,)
    cam = F.relu((weights[:, None, None] * a.detach()).sum(0)).numpy()  # (h, w)
    lo, hi = float(cam.min()), float(cam.max())
    return (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam)


def test_analytic_cam_matches_autograd_synthetic():
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.standard_normal((16, 7, 7)).astype(np.float32)  # feature map (C, h, w)
        w = rng.standard_normal(16).astype(np.float32)  # one label's head weights (C,)
        analytic = cam_from_features(a, w)
        autograd = _autograd_cam(a, w)
        assert analytic.shape == (7, 7)
        assert np.allclose(analytic, autograd, atol=1e-5)


def test_analytic_cam_all_negative_is_zero():
    # a ReLU that kills everything -> a zero map, both ways
    a = -np.ones((4, 5, 5), dtype=np.float32)
    w = np.ones(4, dtype=np.float32)
    assert np.count_nonzero(cam_from_features(a, w)) == 0


def test_model_exposes_real_forward_feature_maps():
    pytest.importorskip("torchxrayvision")
    from hai_sim.core.types import LabelSchema
    from hai_sim.models.torchxrv import TorchXRayVisionAI

    ai = TorchXRayVisionAI(schema=LabelSchema.xrv(), weights="densenet121-res224-nih")
    img = np.zeros((1, 224, 224), dtype=np.float32)
    probs, maps = ai.member_feature_maps_and_probs(img)
    assert probs.shape == (len(ai.members), ai.schema.n)
    assert maps.shape == (len(ai.members), 1024, 7, 7)
    assert np.isfinite(maps).all()
    # weights line up for offline CAM reconstruction
    assert ai.member_classifier_weights().shape == (len(ai.members), ai.schema.n, 1024)
