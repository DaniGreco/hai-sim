"""LocalizationCache save/load round-trip + forward-only builder on synthetic stubs."""

from __future__ import annotations

import numpy as np

from hai_sim.analysis.localization.cache import LocalizationCache, build_localization_cache
from hai_sim.core.types import LabelSchema, LocalizationExample

SCHEMA = LabelSchema.default_v0()  # L0..L4


def _synthetic(n=4, m=3, c=4, hw=3, ell=18, k=5, res=112) -> LocalizationCache:
    rng = np.random.default_rng(0)
    return LocalizationCache(
        feature_maps=rng.random((n, m, c, hw, hw)).astype(np.float16),
        member_probs=rng.random((n, m, ell)).astype(np.float32),
        ground_truth=rng.integers(-1, 2, size=(n, ell)).astype(np.int8),
        inference_seconds=rng.random(n),
        image_ids=tuple(f"{i}.png" for i in range(n)),
        split=("train", "train", "test", "test"),
        dataset_indices=np.arange(n, dtype=np.int64),
        classifier_weights=rng.random((m, ell, c)).astype(np.float32),
        member_covers=(rng.random((m, ell)) > 0.3),
        inst_image=rng.integers(0, n, size=k).astype(np.int64),
        inst_label=rng.integers(0, ell, size=k).astype(np.int64),
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
    assert loaded.n_images == 4
    assert loaded.n_members == 3
    assert loaded.n_labels == 18
    assert loaded.n_instances == 5
    assert loaded.mask_resolution == 112
    assert loaded.image_ids == c.image_ids
    assert loaded.split == c.split
    assert np.array_equal(loaded.feature_maps, c.feature_maps)
    assert loaded.feature_maps.dtype == np.float16
    assert np.array_equal(loaded.member_probs, c.member_probs)
    assert np.array_equal(loaded.classifier_weights, c.classifier_weights)
    assert np.array_equal(loaded.member_covers, c.member_covers)
    assert np.array_equal(loaded.inst_mask, c.inst_mask)


def test_save_is_atomic_leaves_no_tmp(tmp_path):
    c = _synthetic()
    stem = tmp_path / "sub" / "cache"
    c.save(stem)
    assert list((tmp_path / "sub").glob("*.tmp")) == []


class _StubModel:
    """A model exposing the forward-only interface the builder needs."""

    def __init__(self):
        self.weight_list = ["a", "b"]
        self.member_masks = [np.ones(SCHEMA.n, bool), np.ones(SCHEMA.n, bool)]

    def member_feature_maps_and_probs(self, image):
        probs = np.full((2, SCHEMA.n), 0.5, dtype=np.float64)
        maps = np.arange(2 * 4 * 3 * 3, dtype=np.float32).reshape(2, 4, 3, 3)
        return probs, maps

    def member_classifier_weights(self):
        return np.ones((2, SCHEMA.n, 4), dtype=np.float32)


class _StubDataset:
    name = "stub"
    schema = SCHEMA

    def __init__(self):
        img = np.zeros((1, 8, 8), np.float32)
        m = np.ones((8, 8), dtype=bool)
        self._ex = [
            LocalizationExample("a", img, (1, 0, -1, 1, 0), (m, None, None, m, None)),
            LocalizationExample("b", img, (0, 1, 1, -1, 0), (None, m, None, None, None)),
            LocalizationExample("c", img, (0, 0, 0, 0, 1), (None, None, None, None, m)),
        ]

    def __len__(self):
        return len(self._ex)

    def __getitem__(self, i):
        return self._ex[i]


def test_builder_stores_feature_maps_and_references(tmp_path):
    cache = build_localization_cache(
        _StubModel(),
        _StubDataset(),
        indices=[0, 1, 2],
        split_tags=["train", "train", "test"],
        out_stem=tmp_path / "cache",
        total_images=3,
        mask_resolution=8,
    )
    assert cache.n_images == 3
    assert cache.n_members == 2
    assert cache.feature_maps.shape == (3, 2, 4, 3, 3)  # memmap sized to total_images
    assert cache.feature_maps.dtype == np.float16
    assert cache.member_probs.shape == (3, 2, SCHEMA.n)
    assert cache.classifier_weights.shape == (2, SCHEMA.n, 4)
    assert cache.member_covers.shape == (2, SCHEMA.n)
    assert cache.inference_seconds.shape == (3,)
    # instances: ex0 labels 0,3 ; ex1 label 1 ; ex2 label 4
    assert cache.n_instances == 4
    assert list(cache.inst_image) == [0, 0, 1, 2]
    assert list(cache.inst_label) == [0, 3, 1, 4]
    assert cache.inst_mask.shape == (4, 8, 8)
    assert cache.split == ("train", "train", "test")


def test_builder_resume_continues_row_indexing(tmp_path):
    ds = _StubDataset()
    stem = tmp_path / "cache"
    first = build_localization_cache(
        _StubModel(),
        ds,
        indices=[0, 1],
        split_tags=["train", "train"],
        out_stem=stem,
        total_images=3,  # full planned total, so the memmap fits the later resume
        mask_resolution=8,
    )
    assert first.n_images == 2
    combined = build_localization_cache(
        _StubModel(),
        ds,
        indices=[2],
        split_tags=["test"],
        out_stem=stem,
        total_images=3,
        mask_resolution=8,
        prior=first,
    )
    assert combined.n_images == 3
    assert combined.feature_maps.shape == (3, 2, 4, 3, 3)
    assert list(combined.inst_image) == [0, 0, 1, 2]
    assert list(combined.inst_label) == [0, 3, 1, 4]
    assert combined.split == ("train", "train", "test")
