"""The standalone materialize command: resolve a model's cells for a dataset and, per
mode, report / require / fill — exercised with a fake model over a tiny dataset so the
orchestration is checked without torch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from hai_sim.analysis.cache_matrix import cells_for_ai
from hai_sim.config.scenario import AIModelConfig, PatientGenConfig
from hai_sim.core.types import DatasetExample, LabelSchema

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from materialize_cache import run_materialize

SCHEMA = LabelSchema(labels=("A", "B", "C"))


class FakeExampleDataset:
    def __init__(self, ids) -> None:
        self._ids = list(ids)

    @property
    def example_ids(self) -> list[str]:
        return list(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, i: int) -> DatasetExample:
        return DatasetExample(
            example_id=self._ids[i],
            image=np.full((1, 4, 4), float(i), dtype=np.float32),
            ground_truth=(1, 0, -1),
        )


class FakeModel:
    def __init__(self) -> None:
        self.schema = SCHEMA
        self.seen: list[str] = []

    @property
    def member_names(self) -> tuple[str, ...]:
        return ("m0",)

    def member_probabilities_batch(self, images) -> np.ndarray:
        self.seen.extend("x" for _ in images)
        return np.full((len(images), 1, 3), 0.3)


def _ai() -> AIModelConfig:
    return AIModelConfig(
        kind="torchxrv",
        weights=("densenet121-res224-chex", "densenet121-res224-pc"),
    )


def _pg() -> PatientGenConfig:
    return PatientGenConfig(
        kind="image_folder", arrival_rate=1.0, n_patients=1, dataset_id="tds", resize=8
    )


def _run(tmp_path, *, mode, only=None, model_builder=None, dataset=None) -> None:
    run_materialize(
        _ai(), _pg(), SCHEMA,
        limit=0, sample="sequential", seed=0, device=None,
        cache_root=str(tmp_path), only=only, mode=mode, batch_size=1, checkpoint_every=0,
        dataset=dataset if dataset is not None else FakeExampleDataset(["a", "b", "c"]),
        model_builder=model_builder,
    )


def _npz(tmp_path, alias: str) -> Path:
    return (tmp_path / f"tds__{alias}").with_suffix(".npz")


def test_fill_builds_every_cell(tmp_path) -> None:
    model = FakeModel()
    _run(tmp_path, mode="fill", model_builder=lambda _c, _s: model)
    for cell in cells_for_ai(_ai()):
        assert _npz(tmp_path, cell.alias).exists()
    # each of the 3 images run once per cell (2 cells) => 6 forward calls on the shared model
    assert len(model.seen) == 6


def test_report_does_not_build(tmp_path) -> None:
    def boom(_c, _s):
        raise AssertionError("report must not build a model")

    _run(tmp_path, mode="report", model_builder=boom)  # must not raise
    assert not _npz(tmp_path, "chex").exists()


def test_only_builds_the_named_cell(tmp_path) -> None:
    _run(tmp_path, mode="fill", only=("chex",), model_builder=lambda _c, _s: FakeModel())
    assert _npz(tmp_path, "chex").exists()
    assert not _npz(tmp_path, "pc").exists()


def test_only_rejects_unknown_alias(tmp_path) -> None:
    with pytest.raises(ValueError, match="not in this model"):
        _run(tmp_path, mode="fill", only=("nope",), model_builder=lambda _c, _s: FakeModel())
