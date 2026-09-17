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


# --- evidence-based fusion ------------------------------------------------------------


def test_log_odds_reads_a_calibrated_probability_as_evidence_against_its_prevalence():
    """0.10 on a 1% finding is ten times the base rate: it must push the team UP, where
    a probability average would drag it down."""
    from hai_sim.doctors.assisted import LogOddsFusionPolicy

    doctor, ai = _out(0.45), _out(0.10)
    blind = LogOddsFusionPolicy(beta=0.0).combine(doctor, ai)
    aware = LogOddsFusionPolicy(beta=0.0, ai_prior=[0.01]).combine(doctor, ai)
    pooled = LinearPoolPolicy(weight_ai=0.5).combine(doctor, ai)
    assert blind.diagnosis.confidence[0] < 0.45 and pooled.diagnosis.confidence[0] < 0.45
    assert aware.diagnosis.confidence[0] > 0.45 and aware.diagnosis.labels[0] == 1


def test_log_odds_with_no_ai_weight_is_the_unaided_doctor():
    from hai_sim.doctors.assisted import LogOddsFusionPolicy

    out = LogOddsFusionPolicy(beta=0.0, ai_weight=0.0, ai_prior=[0.01]).combine(
        _out(0.37), _out(0.9)
    )
    assert out.diagnosis.confidence[0] == pytest.approx(0.37)


def test_unknown_prevalence_falls_back_to_the_shared_prior():
    from hai_sim.doctors.assisted import LogOddsFusionPolicy

    a = LogOddsFusionPolicy(beta=0.0, ai_prior=[float("nan")]).combine(_out(0.4), _out(0.7))
    b = LogOddsFusionPolicy(beta=0.0).combine(_out(0.4), _out(0.7))
    assert a.diagnosis.confidence[0] == pytest.approx(b.diagnosis.confidence[0])


def test_scenario_fusion_overrides_the_rule_and_keeps_the_persona_weight():
    from hai_sim.config.scenario import FusionConfig
    from hai_sim.doctors.assisted import LogOddsFusionPolicy
    from hai_sim.factory import build_persona

    doc = build_persona(
        "senior_trusting",
        LabelSchema.xrv(),
        np.random.default_rng(0),
        fusion=FusionConfig(kind="log_odds_fusion", beta=0.0),
    )
    assert isinstance(doc.policy, LogOddsFusionPolicy)
    assert doc.policy.ai_weight == pytest.approx(0.45)
    assert doc.policy.weight_doctor == pytest.approx(0.55)
    assert doc.policy.beta == 0.0


def test_permutation_reads_every_image_of_the_split_exactly_once():
    gen = CachedPatientGenerator(
        cache=_cache(20),
        schema=LabelSchema.xrv(),
        arrival_rate=1.0,
        n_patients=10,
        rng=np.random.default_rng(3),
        sample_mode="permutation",
        split="odd",
    )
    ids = [p.context.example_id for _, p in gen]
    assert sorted(ids) == sorted(f"img{i}" for i in range(1, 20, 2))
    assert ids != sorted(ids, key=lambda s: int(s[3:])), "and not in cache order"
