"""Localization metrics on synthetic heatmap/mask pairs with known answers."""

from __future__ import annotations

import math

import numpy as np

from hai_sim.analysis.localization.metrics import (
    Dice,
    EnergyInMask,
    IoU,
    PointingGame,
    default_metrics,
)


def _mask(h: int, w: int, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    m = np.zeros((h, w), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def test_perfect_overlap_scores_one():
    gt = _mask(10, 10, (2, 2, 6, 6))
    heat = gt.astype(float)  # heat exactly the mask
    assert PointingGame().score(heat, gt) == 1.0
    assert EnergyInMask().score(heat, gt) == 1.0
    assert IoU(0.5).score(heat, gt) == 1.0
    assert Dice(0.5).score(heat, gt) == 1.0


def test_disjoint_heatmap_scores_zero():
    gt = _mask(10, 10, (0, 0, 3, 3))
    heat = np.zeros((10, 10))
    heat[7:10, 7:10] = 1.0  # all energy far from the mask
    assert PointingGame().score(heat, gt) == 0.0
    assert EnergyInMask().score(heat, gt) == 0.0
    assert IoU(0.5).score(heat, gt) == 0.0
    assert Dice(0.5).score(heat, gt) == 0.0


def test_pointing_game_uses_only_the_peak():
    gt = _mask(10, 10, (0, 0, 2, 2))
    heat = np.zeros((10, 10))
    heat[:, :] = 0.1  # diffuse background
    heat[1, 1] = 9.0  # single hot pixel inside the mask
    assert PointingGame().score(heat, gt) == 1.0
    # peak outside -> miss, even though most of the plane is lukewarm
    heat[1, 1] = 0.1
    heat[9, 9] = 9.0
    assert PointingGame().score(heat, gt) == 0.0


def test_energy_in_mask_is_the_energy_fraction():
    gt = _mask(10, 10, (0, 0, 10, 5))  # left half
    heat = np.zeros((10, 10))
    heat[0, 0] = 3.0  # inside
    heat[0, 9] = 1.0  # outside
    assert math.isclose(EnergyInMask().score(heat, gt), 0.75)


def test_energy_in_mask_clips_negative_attribution():
    gt = _mask(4, 4, (0, 0, 2, 4))  # top half
    heat = np.array(
        [
            [2.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [-5.0, -5.0, -5.0, -5.0],  # negative -> clipped to 0, ignored
            [-5.0, -5.0, -5.0, -5.0],
        ]
    )
    assert EnergyInMask().score(heat, gt) == 1.0


def test_empty_mask_returns_nan():
    gt = np.zeros((8, 8), dtype=bool)
    heat = np.random.default_rng(0).random((8, 8))
    for m in default_metrics():
        assert math.isnan(m.score(heat, gt))


def test_flat_heatmap_is_nan_for_contrast_metrics():
    gt = _mask(8, 8, (1, 1, 4, 4))
    heat = np.full((8, 8), 0.3)  # no contrast -> no peak / normalization undefined
    assert math.isnan(IoU(0.5).score(heat, gt))
    assert math.isnan(Dice(0.5).score(heat, gt))
    # pointing-game needs a distinguishable peak; a flat map has none -> NaN, NOT a
    # corner-pixel hit/miss (the whole point of the no-contrast guard).
    assert math.isnan(PointingGame().score(heat, gt))
    assert math.isnan(PointingGame(tolerance=2).score(heat, gt))
    # energy-in-mask is still defined for a flat positive map (= mask area fraction)
    assert not math.isnan(EnergyInMask().score(heat, gt))


def test_zero_and_nan_heatmaps_are_nan_for_pointing_game():
    # A no-coverage / degenerate CAM arrives as an all-zero or all-NaN map. Its argmax
    # is the top-left pixel; if the mask happens to cover that corner the OLD behavior
    # scored a spurious 1.0. Both must now be NaN so they drop out of the denominator.
    gt = _mask(6, 6, (0, 0, 2, 2))  # mask covers the (0, 0) corner argmax would pick
    assert math.isnan(PointingGame().score(np.zeros((6, 6)), gt))
    assert math.isnan(PointingGame().score(np.full((6, 6), np.nan), gt))
    assert math.isnan(PointingGame(tolerance=3).score(np.zeros((6, 6)), gt))


def test_partial_overlap_iou_and_dice():
    # pred (normalized>=0.5) covers rows 0..4; gt covers rows 2..6. Overlap rows 2..4.
    gt = _mask(8, 8, (2, 8 * 0, 6, 8))  # rows 2..6, all cols
    heat = np.zeros((8, 8))
    heat[0:4, :] = 1.0  # top 4 rows hot -> after min-max, >=0.5 is exactly these
    iou = IoU(0.5).score(heat, gt)
    dice = Dice(0.5).score(heat, gt)
    # intersection = rows 2,3 (2x8=16); union = rows 0..6 (6x8=48) -> 16/48
    assert math.isclose(iou, 16 / 48)
    # dice = 2*16 / (32 + 32) = 0.5
    assert math.isclose(dice, 0.5)


def test_pointing_game_tolerance():
    gt = _mask(20, 20, (10, 10, 12, 12))  # small 2x2 finding
    heat = np.zeros((20, 20))
    heat[8, 8] = 1.0  # peak 2px up-left of the mask corner (10,10)
    assert PointingGame(tolerance=0).score(heat, gt) == 0.0  # strict miss
    assert PointingGame(tolerance=1).score(heat, gt) == 0.0  # still just out of reach
    assert PointingGame(tolerance=2).score(heat, gt) == 1.0  # within 2px -> hit


def test_metric_names_are_stable():
    assert PointingGame().name == "pointing_game"
    assert PointingGame(16).name == "pointing_game@16px"
    assert EnergyInMask().name == "energy_in_mask"
    assert IoU(0.5).name == "iou@0.5"
    assert Dice(0.25).name == "dice@0.25"
