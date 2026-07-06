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

from hai_sim.analysis.localization.cam import cam_from_features

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


def test_fp16_feature_maps_preserve_cam():
    """The cache stores feature maps as float16; the CAM derived from the quantized
    map must stay close to the float32 one (the sign test is the sensitive part)."""
    rng = np.random.default_rng(1)
    a = (rng.standard_normal((32, 7, 7)) * 3.0).astype(np.float32)
    w = rng.standard_normal(32).astype(np.float32)
    cam32 = cam_from_features(a, w)
    cam16 = cam_from_features(a.astype(np.float16).astype(np.float32), w)  # round-trip fp16
    assert np.allclose(cam32, cam16, atol=0.05)


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


def test_captured_feature_maps_are_raw_not_relud():
    """Regression: the xrv head does F.relu(features, inplace=True), so the hook
    must clone to keep the RAW (signed) feature map — a ReLU'd map has no negatives
    and would make the analytic CAM (which relies on the sign of A) wrong."""
    pytest.importorskip("torchxrayvision")
    from hai_sim.core.types import LabelSchema
    from hai_sim.models.torchxrv import TorchXRayVisionAI

    ai = TorchXRayVisionAI(schema=LabelSchema.xrv(), weights="densenet121-res224-nih")
    img = np.random.default_rng(1).standard_normal((1, 224, 224)).astype(np.float32) * 100.0
    _probs, maps = ai.member_feature_maps_and_probs(img)
    assert (maps < 0).any()  # raw features have negatives; ReLU'd ones would not
    weights = ai.member_classifier_weights()
    covered = [i for i, v in enumerate(ai.member_masks[0]) if v]
    probs = ai.member_probabilities(img)[0]
    top = max(covered, key=lambda i: probs[i])
    cam = cam_from_features(maps[0], weights[0, top])  # analytic CAM on the raw map
    assert cam.shape == (7, 7)
    assert np.isfinite(cam).all()
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert cam.max() > 0.0  # a strongly-predicted label is not a flat/degenerate map
