"""LocalizationValidator aggregation + resume, on deterministic stubs.

No model or dataset download: a stub actor replays fixed predictions, a stub
saliency returns a fixed single-peak heatmap, and single-pixel masks make every
metric hand-computable.
"""

from __future__ import annotations

import math

import numpy as np

from hai_sim.analysis.localization.validator import LocalizationValidator
from hai_sim.core.types import Diagnosis, LabelSchema, LocalizationExample, ModelOutput

SCHEMA = LabelSchema.default_v0()  # L0..L4


def _pixel_mask(rc: tuple[int, int]) -> np.ndarray:
    m = np.zeros((4, 4), dtype=bool)
    m[rc] = True
    return m


class StubDataset:
    name = "stub"
    schema = SCHEMA

    def __init__(self) -> None:
        img = np.zeros((1, 4, 4), dtype=np.float32)
        self._examples = [
            LocalizationExample(
                image_id="a",
                image=img,
                labels=(1, 0, -1, 1, 0),
                masks=(_pixel_mask((0, 0)), None, None, _pixel_mask((3, 3)), None),
            ),
            LocalizationExample(
                image_id="b",
                image=img,
                labels=(0, 1, 1, -1, 0),
                masks=(None, _pixel_mask((1, 1)), None, None, None),
            ),
        ]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, i: int) -> LocalizationExample:
        return self._examples[i]


class StubActor:
    name = "stub_actor"
    decision_threshold = 0.5

    def __init__(self) -> None:
        # Predictions replayed in dataset order.
        self._preds = [(1, 1, 0, 0, 0), (0, 1, 0, 0, 1)]
        self.calls = 0

    def diagnose(self, images, *, context=None, ai_hint=None) -> ModelOutput:
        labels = self._preds[self.calls]
        self.calls += 1
        return ModelOutput(
            diagnosis=Diagnosis(labels=labels, confidence=(0.0,) * len(labels)),
            uncertainty=(0.0,) * len(labels),
        )

    def service_time(self, images, context=None) -> float:
        return 0.0


class StubSaliency:
    """A fixed heatmap with a single hot pixel at (0, 0)."""

    name = "stub_cam"

    def heatmap(self, image, label_index: int) -> np.ndarray:
        h = np.zeros((4, 4), dtype=np.float32)
        h[0, 0] = 1.0
        return h


def test_label_confusion():
    report = LocalizationValidator(StubActor(), StubDataset(), StubSaliency()).run()
    assert report.n_images == 2
    c = report.label_confusion
    # L0: hit on img a (gt1,pred1), correct-reject on img b (gt0,pred0)
    assert c["L0"].sensitivity == 1.0
    assert c["L0"].specificity == 1.0
    # L4: img a tn, img b fp -> spec 0.5, no positives so sensitivity undefined
    assert c["L4"].specificity == 0.5
    assert c["L4"].sensitivity is None
    # L2 only ever appears as gt1 on img b, predicted 0 -> a miss, no negatives
    assert c["L2"].sensitivity == 0.0
    assert c["L2"].specificity is None


def test_localization_pooled_and_per_label():
    report = LocalizationValidator(StubActor(), StubDataset(), StubSaliency()).run()
    pooled = {m.metric: m for m in report.localization_pooled}
    # three localized instances: L0@(0,0) hit, L3@(3,3) miss, L1@(1,1) miss
    assert pooled["pointing_game"].n_valid == 3
    assert math.isclose(pooled["pointing_game"].mean, 1 / 3)
    assert math.isclose(pooled["energy_in_mask"].mean, 1 / 3)
    assert math.isclose(pooled["iou@0.5"].mean, 1 / 3)

    per = report.localization_per_label
    pg_l0 = next(m for m in per["L0"] if m.metric == "pointing_game")
    pg_l3 = next(m for m in per["L3"] if m.metric == "pointing_game")
    assert pg_l0.mean == 1.0
    assert pg_l3.mean == 0.0


def test_predicted_positive_split():
    report = LocalizationValidator(StubActor(), StubDataset(), StubSaliency()).run()
    pred_pooled = {m.metric: m for m in report.localization_pooled_predicted}
    # predicted-positive localized instances: L0 (pred 1, hit) and L1 (pred 1, miss)
    assert pred_pooled["pointing_game"].n_valid == 2
    assert pred_pooled["pointing_game"].mean == 0.5


def test_metadata_carried_through():
    report = LocalizationValidator(StubActor(), StubDataset(), StubSaliency()).run()
    assert report.model == "stub_actor"
    assert report.dataset == "stub"
    assert report.saliency == "stub_cam"
    assert report.decision_threshold == 0.5


def test_label_only_without_saliency():
    report = LocalizationValidator(StubActor(), StubDataset(), saliency=None).run()
    assert report.localization_per_label == {}
    for agg in report.localization_pooled:
        assert agg.n_total == 0
        assert agg.mean is None
    # label validation still works
    assert report.label_confusion["L0"].sensitivity == 1.0


class StubSaliencyNaNForL3:
    """Single hot pixel at (0, 0), except label 3 is 'not localizable' (all-NaN)."""

    name = "stub_cam_nan"

    def heatmap(self, image, label_index: int) -> np.ndarray:
        if label_index == 3:
            return np.full((4, 4), np.nan, dtype=np.float32)
        h = np.zeros((4, 4), dtype=np.float32)
        h[0, 0] = 1.0
        return h


def test_non_localizable_label_is_skipped_like_the_cache():
    # L3 (a real mask on img a) reports an all-NaN saliency -> the validator must SKIP
    # the instance, exactly as evaluate_from_cache skips a label no member covers, so a
    # non-localizable finding is not counted as a pointing-game miss. Only L0 (hit) and
    # L1 (miss) stay in the pooled denominator.
    report = LocalizationValidator(StubActor(), StubDataset(), StubSaliencyNaNForL3()).run()
    pooled = {m.metric: m for m in report.localization_pooled}
    assert pooled["pointing_game"].n_total == 2  # L3 dropped, not scored 0.0
    assert pooled["pointing_game"].n_valid == 2
    assert math.isclose(pooled["pointing_game"].mean, 0.5)  # L0 hit, L1 miss
    assert "L3" not in report.localization_per_label


def test_resume_skips_done_indices(tmp_path):
    records = tmp_path / "rec.json"
    actor = StubActor()
    r1 = LocalizationValidator(actor, StubDataset(), StubSaliency()).run(records_path=records)
    assert actor.calls == 2
    assert records.exists()
    # second run with the same records path: everything already done, no new diagnose
    actor2 = StubActor()
    r2 = LocalizationValidator(actor2, StubDataset(), StubSaliency()).run(records_path=records)
    assert actor2.calls == 0
    assert r2.n_images == r1.n_images == 2
    assert r2.label_confusion["L0"].sensitivity == 1.0
