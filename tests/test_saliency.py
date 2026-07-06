"""Grad-CAM saliency on the real xrv DenseNet (single NIH weight for speed).

Skipped when the [models] extra is absent. Validates the SaliencyProvider
contract (shape, range, coverage behavior), not the clinical quality of the map.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torchxrayvision")

from hai_sim.analysis.localization.saliency import GradCAMSaliency
from hai_sim.core.types import LabelSchema
from hai_sim.models.torchxrv import TorchXRayVisionAI


@pytest.fixture(scope="module")
def cam():
    model = TorchXRayVisionAI(schema=LabelSchema.xrv(), weights="densenet121-res224-nih")
    return GradCAMSaliency(model), model


def _covered_and_uncovered(model) -> tuple[int, int]:
    mask = model.member_masks[0]
    covered = int(np.argmax(mask))  # first True
    uncovered = [i for i, v in enumerate(mask) if not v]
    return covered, (uncovered[0] if uncovered else -1)


def test_heatmap_shape_and_range(cam):
    saliency, model = cam
    img = np.random.default_rng(0).standard_normal((1, 224, 224)).astype(np.float32) * 100.0
    # Use the label the model most strongly predicts: Grad-CAM for a class with
    # positive evidence is reliably non-degenerate, whereas a class the model
    # rejects can legitimately ReLU to an all-zero map.
    probs = model.member_probabilities(img)[0]  # (18,), NaN on uncovered slots
    covered = [i for i, v in enumerate(model.member_masks[0]) if v]
    top = max(covered, key=lambda i: probs[i])
    hm = saliency.heatmap(img, top)
    assert hm.shape == (224, 224)
    assert hm.dtype == np.float32
    assert np.isfinite(hm).all()
    assert hm.min() >= 0.0 and hm.max() <= 1.0
    assert hm.max() > 0.0  # min-max normalized -> peak is 1.0
    assert hm.std() > 0.0


def test_uncovered_label_returns_zeros(cam):
    saliency, model = cam
    _, uncovered = _covered_and_uncovered(model)
    if uncovered < 0:
        pytest.skip("this weight set covers every schema label")
    img = np.random.default_rng(1).standard_normal((1, 224, 224)).astype(np.float32) * 100.0
    hm = saliency.heatmap(img, uncovered)
    assert hm.shape == (224, 224)
    assert np.count_nonzero(hm) == 0


def test_accepts_2d_image(cam):
    saliency, model = cam
    covered, _ = _covered_and_uncovered(model)
    img2d = np.random.default_rng(2).standard_normal((224, 224)).astype(np.float32) * 100.0
    hm = saliency.heatmap(img2d, covered)
    assert hm.shape == (224, 224)


def test_member_cams_shape_and_heatmap_equivalence(cam):
    saliency, model = cam
    covered, _ = _covered_and_uncovered(model)
    img = np.random.default_rng(3).standard_normal((1, 224, 224)).astype(np.float32) * 100.0
    mc = saliency.member_cams(img, covered)
    m = len(model.members)
    assert mc.shape == (m, 7, 7)  # per-member native Grad-CAM resolution
    # the fused heatmap must equal upsample(nanmean(member_cams)) — the property the
    # cache relies on to store 7x7 per-member and aggregate offline.
    agg = np.nanmean(mc, axis=0)
    up = saliency._upsample(agg, (224, 224))
    assert np.allclose(saliency.heatmap(img, covered), up, atol=1e-5)


def test_uncovered_label_member_cams_all_nan(cam):
    saliency, model = cam
    _, uncovered = _covered_and_uncovered(model)
    if uncovered < 0:
        pytest.skip("this weight set covers every schema label")
    img = np.random.default_rng(4).standard_normal((1, 224, 224)).astype(np.float32) * 100.0
    mc = saliency.member_cams(img, uncovered)
    assert np.isnan(mc).all()
