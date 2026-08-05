"""The dataset-agnostic cache builder and the ``materialize_cell`` fill primitive:
building over `DatasetExample` items (VinDr's shape) preserves the ``-1`` not-annotated
sentinel, and a cell grows to cover new requests, building only the gap."""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.analysis.cache_build import materialize_cell
from hai_sim.analysis.cache_matrix import CellSpec, cells_for_ai, inspect_cell
from hai_sim.analysis.inference_cache import build_inference_cache
from hai_sim.config.scenario import AIModelConfig, PatientGenConfig
from hai_sim.core.types import DatasetExample, LabelSchema

SCHEMA = LabelSchema(labels=("A", "B", "C"))


class FakeExampleDataset:
    """A minimal framework `Dataset`: DatasetExample items, ids without loading pixels."""

    def __init__(self, ids) -> None:
        self._ids = list(ids)
        self.schema = SCHEMA
        self.name = "fake"

    @property
    def example_ids(self) -> list[str]:
        return list(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, i: int) -> DatasetExample:
        return DatasetExample(
            example_id=self._ids[i],
            image=np.full((1, 4, 4), float(i), dtype=np.float32),
            ground_truth=(1, 0, -1),  # a -1 to prove the sentinel survives caching
        )


class FakeModel:
    """A one-member CacheableModel that records which images it was asked to score."""

    def __init__(self) -> None:
        self.schema = SCHEMA
        self.seen: list[float] = []

    @property
    def member_names(self) -> tuple[str, ...]:
        return ("m0",)

    def member_probabilities(self, image) -> np.ndarray:
        self.seen.append(float(np.asarray(image).flat[0]))
        return np.array([[0.1, 0.2, 0.3]])

    def member_probabilities_batch(self, images) -> np.ndarray:
        return np.stack([self.member_probabilities(im) for im in images], axis=0)


def _pg() -> PatientGenConfig:
    return PatientGenConfig(
        kind="image_folder", arrival_rate=1.0, n_patients=1, dataset_id="tds", resize=8
    )


def _chex_cell() -> CellSpec:
    return cells_for_ai(AIModelConfig(kind="torchxrv", weights="densenet121-res224-chex"))[0]


def test_builder_over_dataset_examples_keeps_minus_one() -> None:
    cache = build_inference_cache(
        FakeModel(), FakeExampleDataset(["a", "b"]), indices=[0, 1], dataset_name="tds"
    )
    assert cache.image_ids == ("a", "b")
    assert cache.member_probs.shape == (2, 1, 3)
    # ground truth (1, 0, -1) survives: the -1 is NOT turned into an annotated 0.
    assert list(cache.ground_truth[0]) == [1, 0, -1]


def test_materialize_builds_then_extends_only_the_gap(tmp_path) -> None:
    ds = FakeExampleDataset(["a", "b", "c"])
    model = FakeModel()
    spec, pg = _chex_cell(), _pg()

    def build(_cfg, _schema):
        return model

    c1 = materialize_cell(
        spec, pg, SCHEMA, indices=[0, 1], root=tmp_path, dataset=ds, model_builder=build
    )
    assert c1.image_ids == ("a", "b")
    assert model.seen == [0.0, 1.0]

    # Second call asks for the same two plus a new one — only the new image is run.
    c2 = materialize_cell(
        spec, pg, SCHEMA, indices=[0, 1, 2], root=tmp_path, dataset=ds, model_builder=build
    )
    assert c2.image_ids == ("a", "b", "c")
    assert model.seen == [0.0, 1.0, 2.0]  # 'c' only; 'a','b' were not re-run

    # The cell on disk now covers all three.
    st = inspect_cell(spec, "tds", "", root=tmp_path)
    assert st.present_ids == frozenset({"a", "b", "c"})


def test_materialize_full_hit_does_not_build(tmp_path) -> None:
    ds = FakeExampleDataset(["a", "b"])
    spec, pg = _chex_cell(), _pg()
    materialize_cell(
        spec, pg, SCHEMA, indices=[0, 1], root=tmp_path, dataset=ds,
        model_builder=lambda _c, _s: FakeModel(),
    )

    def boom(_cfg, _schema):
        raise AssertionError("model must not be built on a full cache hit")

    got = materialize_cell(
        spec, pg, SCHEMA, indices=[0, 1], root=tmp_path, dataset=ds, model_builder=boom
    )
    assert got.image_ids == ("a", "b")


def test_materialize_rejects_fingerprint_conflict(tmp_path) -> None:
    ds = FakeExampleDataset(["a"])
    pg = _pg()
    materialize_cell(
        _chex_cell(), pg, SCHEMA, indices=[0], root=tmp_path, dataset=ds,
        model_builder=lambda _c, _s: FakeModel(),
    )
    # Same alias/stem, different model identity -> must refuse to mix experiments.
    impostor = CellSpec(alias="chex", fingerprint="torchxrv|OTHER", ai_cfg=_chex_cell().ai_cfg)
    with pytest.raises(ValueError, match="conflict"):
        materialize_cell(
            impostor, pg, SCHEMA, indices=[0], root=tmp_path, dataset=ds,
            model_builder=lambda _c, _s: FakeModel(),
        )
