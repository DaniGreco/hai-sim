"""evaluate_from_cache on a hand-built cache with known answers (pure numpy)."""

from __future__ import annotations

import math

import numpy as np

from hai_sim.analysis.localization.cache import LocalizationCache
from hai_sim.analysis.localization.evaluate import evaluate_from_cache
from hai_sim.core.types import LabelSchema

SCHEMA = LabelSchema.default_v0()  # L0..L4


def _pixel_mask(rc, res=8):
    m = np.zeros((res, res), dtype=bool)
    m[rc] = True
    return m


def _cache() -> LocalizationCache:
    # One member, one feature channel, hot at cell (0,0): cam_from_features then
    # peaks top-left, so the derived CAM is controllable end to end.
    feature_maps = np.zeros((2, 1, 1, 7, 7), dtype=np.float16)
    feature_maps[:, :, :, 0, 0] = 1.0
    classifier_weights = np.ones((1, SCHEMA.n, 1), dtype=np.float32)  # positive -> CAM ~ ReLU(A)
    member_covers = np.ones((1, SCHEMA.n), dtype=bool)
    # img0 probs -> predicted (1,1,0,0,0); img1 -> (0,1,0,0,1) at threshold 0.5
    member_probs = np.array(
        [[[0.9, 0.9, 0.1, 0.1, 0.1]], [[0.1, 0.9, 0.1, 0.1, 0.9]]], dtype=np.float32
    )
    ground_truth = np.array([[1, 0, -1, 1, 0], [0, 1, 1, -1, 0]], dtype=np.int8)
    # 3 localized instances: (img0,L0) peak on mask=hit; (img0,L3) mask far=miss;
    # (img1,L1) mask far=miss.
    inst_mask = np.stack([_pixel_mask((0, 0)), _pixel_mask((7, 7)), _pixel_mask((7, 7))])
    return LocalizationCache(
        feature_maps=feature_maps,
        member_probs=member_probs,
        ground_truth=ground_truth,
        inference_seconds=np.zeros(2),
        image_ids=("a", "b"),
        split=("train", "test"),
        dataset_indices=np.array([0, 1], dtype=np.int64),
        classifier_weights=classifier_weights,
        member_covers=member_covers,
        inst_image=np.array([0, 0, 1], dtype=np.int64),
        inst_label=np.array([0, 3, 1], dtype=np.int64),
        inst_mask=inst_mask,
        weights=("w",),
        pathologies=SCHEMA.labels,
        dataset="stub",
        mask_resolution=8,
    )


def test_label_confusion_from_cache():
    r = evaluate_from_cache(_cache(), threshold=0.5)
    assert r.n_images == 2
    assert r.label_confusion["L0"].sensitivity == 1.0
    assert r.label_confusion["L0"].specificity == 1.0
    assert r.label_confusion["L4"].specificity == 0.5
    assert r.label_confusion["L4"].sensitivity is None


def test_localization_from_cache():
    r = evaluate_from_cache(_cache(), threshold=0.5)
    pooled = {m.metric: m for m in r.localization_pooled}
    assert pooled["pointing_game"].n_valid == 3
    assert math.isclose(pooled["pointing_game"].mean, 1 / 3)  # one hit of three
    pred = {m.metric: m for m in r.localization_pooled_predicted}
    # predicted-positive instances: (img0,L0) hit and (img1,L1) miss
    assert pred["pointing_game"].n_valid == 2
    assert pred["pointing_game"].mean == 0.5


def test_split_filter():
    # train split has only img0 (instances L0 hit, L3 miss); test has img1 (L1 miss)
    r_train = evaluate_from_cache(_cache(), split="train")
    assert r_train.n_images == 1
    pooled = {m.metric: m for m in r_train.localization_pooled}
    assert pooled["pointing_game"].n_valid == 2
    assert pooled["pointing_game"].mean == 0.5  # one hit of two on img0

    r_test = evaluate_from_cache(_cache(), split="test")
    assert r_test.n_images == 1
    assert {m.metric: m for m in r_test.localization_pooled}["pointing_game"].mean == 0.0


def test_threshold_changes_predictions_offline():
    # raising the threshold above 0.9 makes every prediction negative -> no TPs
    r = evaluate_from_cache(_cache(), threshold=0.95)
    assert r.label_confusion["L0"].tp == 0
    assert r.label_confusion["L0"].fn == 1  # img0 L0 was a positive, now missed
