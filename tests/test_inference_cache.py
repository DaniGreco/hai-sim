"""Tests for the offline inference cache: round-trip + UQ consistency.

The round-trip and the measure-consistency tests are pure (synthetic arrays, no
model), so they run without the heavy [models] extra. A guarded test reproduces
the simulator's uncertainty from cached member probabilities when torch is present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hai_sim.analysis.inference_cache import (
    InferenceCache,
    build_inference_cache,
    measures_from_cache,
)
from hai_sim.core.types import LabelSchema
from hai_sim.uq.methods import EnsembleUQ


def _synthetic_cache(n: int = 6, m: int = 3) -> InferenceCache:
    rng = np.random.default_rng(0)
    mp = rng.random((n, m, 18)).astype(np.float32)
    mp[:, 1, 5] = np.nan  # member 1 does not cover label 5 (an uncovered slot)
    gt = rng.choice([1, 0, -1], size=(n, 18)).astype(np.int8)
    inf = rng.random(n)
    return InferenceCache(
        member_probs=mp,
        ground_truth=gt,
        inference_seconds=inf,
        image_ids=tuple(f"img{i:04d}.png" for i in range(n)),
        dataset_indices=np.arange(n, dtype=np.int64),
        weights=tuple(f"w{j}" for j in range(m)),
        pathologies=tuple(LabelSchema.xrv().labels),
        dataset="nih",
        batch_size=1,
    )


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = _synthetic_cache()
    stem = tmp_path / "cache" / "c"
    cache.save(stem)

    assert stem.with_suffix(".npz").exists()
    assert stem.with_suffix(".json").exists()

    loaded = InferenceCache.load(stem)
    np.testing.assert_array_equal(loaded.member_probs, cache.member_probs)  # NaN == NaN here
    np.testing.assert_array_equal(loaded.ground_truth, cache.ground_truth)
    np.testing.assert_array_equal(loaded.inference_seconds, cache.inference_seconds)
    np.testing.assert_array_equal(loaded.dataset_indices, cache.dataset_indices)
    assert loaded.image_ids == cache.image_ids
    assert loaded.weights == cache.weights
    assert loaded.pathologies == cache.pathologies
    assert loaded.dataset == cache.dataset
    assert loaded.batch_size == cache.batch_size
    assert loaded.n_images == 6
    assert loaded.n_members == 3
    assert loaded.n_labels == 18


def test_save_is_atomic_no_tmp_left(tmp_path: Path) -> None:
    cache = _synthetic_cache()
    stem = tmp_path / "c"
    cache.save(stem)
    # no leftover *.tmp temp files
    assert not list(tmp_path.glob("*.tmp"))


def test_measures_match_ensemble_uq() -> None:
    """measures_from_cache must reproduce EnsembleUQ image by image, measure by measure."""
    cache = _synthetic_cache()
    m = measures_from_cache(cache.member_probs)
    for measure in ("mutual_information", "predictive_entropy", "mean_entropy", "std"):
        uq = EnsembleUQ(measure=measure)  # type: ignore[arg-type]
        for i in range(cache.n_images):
            mean_p, unc, _ = uq.estimate(cache.member_probs[i])
            np.testing.assert_allclose(m["mean_p"][i], mean_p, rtol=1e-6, equal_nan=True)
            np.testing.assert_allclose(m[measure][i], unc, rtol=1e-6, equal_nan=True)


class _FakeDataset:
    """xrv-like dataset: img encodes its index so we can assert on merged rows."""

    def __init__(self, n: int, m: int, schema: LabelSchema) -> None:
        self.pathologies = list(schema.labels)
        self._n = n
        self._m = m

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        img = np.full((self._m, 18), float(i), dtype=np.float32)  # member_probs = i
        lab = np.where(np.arange(18) < 5, 1.0, np.nan)
        return {"img": img, "lab": lab}


class _FakeModel:
    """Minimal stand-in: returns each image's pre-baked (M, L) matrix as the probs."""

    def __init__(self, schema: LabelSchema, m: int) -> None:
        self.schema = schema
        self.weight_list = [f"w{j}" for j in range(m)]

    def member_probabilities_batch(self, images: list[np.ndarray]) -> np.ndarray:
        return np.stack(images)  # (b, M, L)


def test_build_with_prior_resumes() -> None:
    schema = LabelSchema.xrv()
    model = _FakeModel(schema, m=3)
    ds = _FakeDataset(n=10, m=3, schema=schema)

    first = build_inference_cache(model, ds, indices=[0, 1, 2], dataset_name="nih")  # type: ignore[arg-type]
    assert first.n_images == 3
    np.testing.assert_array_equal(first.dataset_indices, [0, 1, 2])

    merged = build_inference_cache(
        model, ds, indices=[3, 4], dataset_name="nih", prior=first  # type: ignore[arg-type]
    )
    assert merged.n_images == 5
    np.testing.assert_array_equal(merged.dataset_indices, [0, 1, 2, 3, 4])
    # Row order preserved: each row's member_probs equal its dataset index.
    for row, idx in enumerate([0, 1, 2, 3, 4]):
        assert merged.member_probs[row, 0, 0] == float(idx)


# ── Guarded: needs the real model (torch + weights) ──────────────────────────────


def test_cached_probs_reproduce_diagnose() -> None:
    pytest.importorskip("torchxrayvision")
    from hai_sim.core.types import PatientContext
    from hai_sim.models.torchxrv import TorchXRayVisionAI

    schema = LabelSchema.xrv()
    ai = TorchXRayVisionAI(
        schema=schema,
        weights=("densenet121-res224-nih", "densenet121-res224-chex"),
        uq=EnsembleUQ("mutual_information"),
    )
    img = (np.random.default_rng(1).random((1, 224, 224)) * 2048.0 - 1024.0).astype(np.float32)

    probs = ai.member_probabilities(img)
    assert probs.shape == (2, 18)

    # Feeding cached member probs back through the SAME UQ reproduces diagnose().
    _, unc_from_cache, _ = EnsembleUQ("mutual_information").estimate(probs)
    out = ai.diagnose([img], context=PatientContext(patient_id=0, arrival_time=0.0))
    np.testing.assert_allclose(
        np.nan_to_num(unc_from_cache, nan=0.0), out.uncertainty, rtol=1e-6
    )
