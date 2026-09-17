"""The two knobs a calibrated-advice study needs outside the calibrator itself."""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.analysis.inference_cache import InferenceCache
from hai_sim.core.types import LabelSchema, ModelOutput
from hai_sim.doctors.assisted import LinearPoolPolicy
from hai_sim.sim.cache_generator import CachedPatientGenerator


def _out(p: float) -> ModelOutput:
    arr = np.asarray([p])
    return ModelOutput.from_arrays(arr >= 0.5, arr, np.zeros(1))


def test_team_threshold_is_the_policy_s_not_a_constant():
    doctor, ai = _out(0.30), _out(0.10)
    default = LinearPoolPolicy(weight_ai=0.5).combine(doctor, ai)
    liberal = LinearPoolPolicy(weight_ai=0.5, decision_threshold=0.09).combine(doctor, ai)
    assert default.diagnosis.confidence[0] == pytest.approx(0.20)
    assert default.diagnosis.labels[0] == 0
    assert liberal.diagnosis.labels[0] == 1
    assert liberal.extra["team_decision_threshold"] == pytest.approx(0.09)


def _cache(n: int) -> InferenceCache:
    schema = LabelSchema.xrv()
    return InferenceCache(
        member_probs=np.full((n, 1, schema.n), 0.5, dtype=np.float32),
        ground_truth=np.zeros((n, schema.n), dtype=np.int8),
        inference_seconds=np.zeros(n),
        dataset_indices=np.arange(n),
        image_ids=tuple(f"img{i}" for i in range(n)),
        weights=("w",),
        pathologies=schema.labels,
        dataset="toy",
        batch_size=1,
    )


@pytest.mark.parametrize(("split", "parity"), [("even", 0), ("odd", 1)])
def test_split_restricts_the_stream_to_one_parity(split, parity):
    gen = CachedPatientGenerator(
        cache=_cache(21),
        schema=LabelSchema.xrv(),
        arrival_rate=1.0,
        n_patients=200,
        rng=np.random.default_rng(0),
        sample_mode="random",
        split=split,
    )
    rows = {int(p.context.example_id.removeprefix("img")) for _, p in gen}
    assert rows and all(r % 2 == parity for r in rows)


def test_split_all_is_the_unrestricted_default():
    gen = CachedPatientGenerator(
        cache=_cache(6), schema=LabelSchema.xrv(), arrival_rate=1.0, n_patients=6
    )
    assert [p.context.example_id for _, p in gen] == [f"img{i}" for i in range(6)]
