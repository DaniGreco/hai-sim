"""LocalizationCache save/load round-trip + builder logic on synthetic stubs."""

from __future__ import annotations

import numpy as np

from hai_sim.analysis.localization.cache import LocalizationCache, build_localization_cache
from hai_sim.core.types import LabelSchema, LocalizationExample

SCHEMA = LabelSchema.default_v0()  # L0..L4


def _m(res=8):
    return np.ones((res, res), dtype=bool)


class _StubModel:
    def __init__(self):
        self.weight_list = ["a", "b"]

    def member_probabilities(self, image):
        return np.full((2, SCHEMA.n), 0.5, dtype=np.float32)


class _StubSaliency:
    """member_cams returns (M, 7, 7) filled with the label index (checkable)."""

    def __init__(self):
        self.model = _StubModel()

    def member_cams(self, image, label_index):
        return np.full((2, 7, 7), float(label_index), dtype=np.float32)


class _StubDataset:
    name = "stub"
    schema = SCHEMA

    def __init__(self):
        img = np.zeros((1, 8, 8), np.float32)
        self._ex = [
            LocalizationExample("a", img, (1, 0, -1, 1, 0), (_m(), None, None, _m(), None)),
            LocalizationExample("b", img, (0, 1, 1, -1, 0), (None, _m(), None, None, None)),
            LocalizationExample("c", img, (0, 0, 0, 0, 1), (None, None, None, None, _m())),
        ]

    def __len__(self):
        return len(self._ex)

    def __getitem__(self, i):
        return self._ex[i]


def _synthetic(n=4, m=3, ell=18, k=5, cam=7, res=112) -> LocalizationCache:
    rng = np.random.default_rng(0)
    return LocalizationCache(
        member_probs=rng.random((n, m, ell)).astype(np.float32),
        ground_truth=rng.integers(-1, 2, size=(n, ell)).astype(np.int8),
        image_ids=tuple(f"{i}.png" for i in range(n)),
        split=("train", "train", "test", "test"),
        dataset_indices=np.arange(n, dtype=np.int64),
        inst_image=rng.integers(0, n, size=k).astype(np.int64),
        inst_label=rng.integers(0, ell, size=k).astype(np.int64),
        inst_cam=rng.random((k, m, cam, cam)).astype(np.float32),
        inst_mask=(rng.random((k, res, res)) > 0.7),
        weights=("w0", "w1", "w2"),
        pathologies=tuple(f"L{i}" for i in range(ell)),
        dataset="chestxdet",
        mask_resolution=res,
        dataset_fingerprint="fp",
    )


def test_roundtrip(tmp_path):
    c = _synthetic()
    stem = tmp_path / "cache"
    c.save(stem)
    assert (tmp_path / "cache.npz").exists()
    assert (tmp_path / "cache.json").exists()

    loaded = LocalizationCache.load(stem)
    assert loaded.n_images == c.n_images == 4
    assert loaded.n_members == 3
    assert loaded.n_labels == 18
    assert loaded.n_instances == 5
    assert loaded.mask_resolution == 112
    assert loaded.image_ids == c.image_ids
    assert loaded.split == c.split
    assert loaded.weights == c.weights
    assert loaded.dataset == "chestxdet"
    assert np.array_equal(loaded.member_probs, c.member_probs)
    assert np.array_equal(loaded.ground_truth, c.ground_truth)
    assert np.array_equal(loaded.inst_cam, c.inst_cam)
    assert np.array_equal(loaded.inst_mask, c.inst_mask)
    assert loaded.inst_mask.dtype == np.bool_


def test_save_is_atomic_leaves_no_tmp(tmp_path):
    c = _synthetic()
    stem = tmp_path / "sub" / "cache"
    c.save(stem)
    leftovers = list((tmp_path / "sub").glob("*.tmp"))
    assert leftovers == []


def test_builder_creates_instances_for_masked_labels():
    cache = build_localization_cache(
        _StubSaliency(),
        _StubDataset(),
        indices=[0, 1, 2],
        split_tags=["train", "train", "test"],
        mask_resolution=8,
    )
    assert cache.n_images == 3
    assert cache.n_members == 2
    assert cache.n_instances == 4  # ex0: labels 0,3 ; ex1: label 1 ; ex2: label 4
    assert list(cache.inst_image) == [0, 0, 1, 2]
    assert list(cache.inst_label) == [0, 3, 1, 4]
    assert cache.inst_cam.shape == (4, 2, 7, 7)
    assert cache.inst_mask.shape == (4, 8, 8)
    assert cache.split == ("train", "train", "test")
    assert cache.member_probs.shape == (3, 2, SCHEMA.n)
    # member_cams stub fills each CAM with the label index -> checkable mapping
    assert cache.inst_cam[0, 0, 0, 0] == 0.0  # inst 0 is label 0
    assert cache.inst_cam[1, 0, 0, 0] == 3.0  # inst 1 is label 3


def test_builder_resume_continues_row_indexing():
    ds = _StubDataset()
    first = build_localization_cache(
        _StubSaliency(), ds, indices=[0, 1], split_tags=["train", "train"], mask_resolution=8
    )
    assert first.n_images == 2
    combined = build_localization_cache(
        _StubSaliency(), ds, indices=[2], split_tags=["test"], mask_resolution=8, prior=first
    )
    assert combined.n_images == 3
    # ex2 has a mask on label 4; its instance must reference the new row index 2
    assert list(combined.inst_image) == [0, 0, 1, 2]
    assert list(combined.inst_label) == [0, 3, 1, 4]
    assert combined.split == ("train", "train", "test")
