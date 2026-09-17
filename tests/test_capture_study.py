"""Capture study: an ordering is scored by how early it meets the target cases."""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.analysis.capture_study import (
    CaptureModel,
    CaptureStudyConfig,
    compose_members,
    run_capture_study,
)
from hai_sim.analysis.inference_cache import InferenceCache
from hai_sim.core.types import LabelSchema

_SCHEMA = LabelSchema.xrv()


def _cache(probs: np.ndarray, gt: np.ndarray, ids: list[str], name: str) -> InferenceCache:
    n = probs.shape[0]
    member = np.full((n, 1, _SCHEMA.n), np.nan, dtype=np.float32)
    member[:, 0, 0] = probs
    truth = np.full((n, _SCHEMA.n), -1, dtype=np.int8)
    truth[:, 0] = gt
    return InferenceCache(
        member_probs=member,
        ground_truth=truth,
        inference_seconds=np.zeros(n),
        image_ids=tuple(ids),
        dataset_indices=np.arange(n),
        weights=(name,),
        pathologies=_SCHEMA.labels,
        dataset="toy",
        batch_size=1,
    )


def _study(tmp_path, members: dict[str, np.ndarray], gt: np.ndarray, **kw) -> dict:
    ids = [f"img{i}" for i in range(gt.size)]
    for name, probs in members.items():
        _cache(probs, gt, ids, name).save(tmp_path / f"toy__{name}")
    cfg = CaptureStudyConfig(
        name="t",
        cache_dir=str(tmp_path),
        calibration_dir=str(tmp_path / "none"),
        datasets=("toy",),
        models={"m": CaptureModel(cells=tuple(members))},
        min_targets=5,
        **kw,
    )
    return run_capture_study(cfg)


def _row(result: dict, **where) -> dict:
    hits = [r for r in result["rows"] if all(r[k] == v for k, v in where.items())]
    assert len(hits) == 1, hits
    return hits[0]


def test_a_perfect_model_is_the_ideal_ordering_and_a_blind_one_is_random(tmp_path):
    rng = np.random.default_rng(0)
    gt = (rng.random(4000) < 0.1).astype(int)
    perfect = np.where(gt == 1, 0.9, 0.1) + rng.normal(0, 0.01, gt.size)
    res = _study(tmp_path, {"a": perfect}, gt, targets=("positive",))
    assert _row(res, ordering="probability")["skill"] == pytest.approx(1.0, abs=1e-3)

    blind = rng.random(gt.size)
    res = _study(tmp_path / "b", {"a": blind}, gt, targets=("positive",))
    assert abs(_row(res, ordering="probability")["skill"]) < 0.1


def test_member_orderings_exist_only_with_two_covering_members(tmp_path):
    rng = np.random.default_rng(1)
    gt = (rng.random(2000) < 0.2).astype(int)
    p = np.clip(0.2 + 0.5 * gt + rng.normal(0, 0.15, gt.size), 0.01, 0.99)
    one = _study(tmp_path / "one", {"a": p}, gt, targets=("positive",))
    assert {r["ordering"] for r in one["rows"]} == {"probability", "predictive_entropy"}
    q = np.clip(p + rng.normal(0, 0.1, gt.size), 0.01, 0.99)
    two = _study(tmp_path / "two", {"a": p, "b": q}, gt, targets=("positive",))
    assert {"mean_entropy", "mutual_information"} <= {r["ordering"] for r in two["rows"]}
    assert _row(two, ordering="mean_entropy")["n_members"] == 2


def test_the_study_reads_only_the_rows_of_its_split(tmp_path):
    gt = np.tile([1, 1, 0, 0, 0, 0, 0, 0], 250)
    p = np.where(gt == 1, 0.8, 0.2).astype(float)
    res = _study(tmp_path, {"a": p}, gt, targets=("positive",), split="odd")
    assert _row(res, ordering="probability")["n_images"] == 1000


def test_compose_members_keeps_only_shared_images_in_the_first_cache_order():
    gt = np.array([1, 0, 1])
    a = _cache(np.array([0.1, 0.2, 0.3]), gt, ["x", "y", "z"], "a")
    b = _cache(np.array([0.9, 0.7]), gt[[2, 0]], ["z", "x"], "b")
    members, truth, ids = compose_members([a, b])
    assert ids == ("x", "z")
    np.testing.assert_allclose(members[:, :, 0], [[0.1, 0.7], [0.3, 0.9]], rtol=1e-6)
    assert truth[:, 0].tolist() == [1, 1]
