"""Arrival processes, and the substream independence that makes comparing them valid."""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.arrivals.processes import (
    DeterministicArrival,
    NHPPArrival,
    PiecewiseConstantRate,
    PoissonArrival,
)
from hai_sim.config.scenario import ScenarioConfig
from hai_sim.core.protocols import ArrivalProcess
from hai_sim.core.types import LabelSchema
from hai_sim.factory import build_arrivals, build_patient_generator, build_rngs
from hai_sim.sim.patient_generator import PoissonPatientGenerator


@pytest.mark.parametrize(
    "process",
    [
        PoissonArrival(2.0, np.random.default_rng(0)),
        DeterministicArrival(2.0),
        NHPPArrival(PiecewiseConstantRate([1.0, 2.0]), rng=np.random.default_rng(0)),
    ],
)
def test_every_process_satisfies_the_protocol(process):
    assert isinstance(process, ArrivalProcess)
    assert process.next_delay(0.0) > 0.0


def test_poisson_mean_gap_matches_the_rate():
    process = PoissonArrival(4.0, np.random.default_rng(0))
    gaps = [process.next_delay(0.0) for _ in range(20_000)]
    assert np.mean(gaps) == pytest.approx(0.25, rel=0.05)


def test_deterministic_gaps_never_vary():
    process = DeterministicArrival(5.0)
    assert {process.next_delay(t) for t in range(10)} == {0.2}


def test_piecewise_rate_selects_the_slot_and_wraps_with_the_period():
    rate = PiecewiseConstantRate([1.0, 9.0], period=2.0)
    assert rate(0.0) == 1.0
    assert rate(1.5) == 9.0
    assert rate(2.5) == 1.0, "the table must repeat once the period elapses"
    assert rate.peak == 9.0


def test_piecewise_rate_handles_the_closed_upper_edge():
    """Floating point lands exactly on the period boundary often enough to matter."""
    rate = PiecewiseConstantRate([1.0, 9.0], period=2.0)
    assert rate(2.0) == 1.0


def test_nhpp_produces_more_arrivals_in_the_busy_slot():
    """The whole purpose: arrivals must concentrate where the rate is high.

    Counting into the two halves of the cycle, a 10:1 rate ratio has to show up as a
    large count ratio. This is what a homogeneous process cannot do, and what a
    department that is busier by day than by night requires.
    """
    rate = PiecewiseConstantRate([1.0, 10.0], period=2.0)
    process = NHPPArrival(rate, rng=np.random.default_rng(0))
    now = 0.0
    quiet = busy = 0
    for _ in range(4000):
        now += process.next_delay(now)
        if (now % 2.0) < 1.0:
            quiet += 1
        else:
            busy += 1
    assert busy > 5 * quiet


def test_nhpp_refuses_a_rate_without_a_bound():
    with pytest.raises(ValueError, match="rate_max"):
        NHPPArrival(lambda t: 1.0)


def test_nhpp_rejects_a_bound_the_rate_exceeds():
    """Thinning with too small a bound yields the wrong process *silently*, so this
    has to be an error rather than a shrug."""
    process = NHPPArrival(lambda t: 100.0, rate_max=1.0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="exceeds rate_max"):
        process.next_delay(0.0)


def test_nhpp_reports_a_permanently_closed_rate_instead_of_hanging():
    rate = PiecewiseConstantRate([0.0, 1.0], period=2.0)
    process = NHPPArrival(rate, rate_max=1.0, rng=np.random.default_rng(0), max_rejections=3)
    # Every proposal lands in the closed slot only if the rate is zero everywhere it
    # looks; force that by making the open slot unreachable within the retry budget.
    zero = NHPPArrival(lambda t: 0.0, rate_max=1.0, rng=np.random.default_rng(0), max_rejections=5)
    with pytest.raises(RuntimeError, match="no arrival accepted"):
        zero.next_delay(0.0)
    assert process.next_delay(0.0) > 0.0


def test_generator_defaults_to_poisson_and_is_unchanged_by_the_seam():
    """Passing `arrival_rate` must still mean exactly what it always did."""
    schema = LabelSchema.default_v0()
    old_style = PoissonPatientGenerator(
        schema, arrival_rate=2.0, n_patients=20, rng=np.random.default_rng(11)
    )
    new_style = PoissonPatientGenerator(
        schema,
        n_patients=20,
        rng=np.random.default_rng(11),
        arrivals=PoissonArrival(2.0, np.random.default_rng(11)),
    )
    assert [d for d, _ in old_style] != []
    assert isinstance(new_style.arrivals, PoissonArrival)


def test_generator_requires_a_rate_or_a_process():
    with pytest.raises(ValueError, match="arrival process or an arrival_rate"):
        PoissonPatientGenerator(LabelSchema.default_v0(), n_patients=5)


def _scenario(arrivals: dict | None) -> ScenarioConfig:
    pg: dict = {
        "kind": "mock_dataset",
        "arrival_rate": 1.0,
        "n_patients": 40,
        "n_examples": 40,
        "sample_mode": "random",
    }
    if arrivals is not None:
        pg["arrivals"] = arrivals
    return ScenarioConfig.model_validate(
        {
            "name": "t",
            "seed": 5,
            "patient_generator": pg,
            "ai": {"kind": "random_mock"},
            "persona": "senior_solo",
            "sort": {"kind": "random"},
        }
    )


def test_changing_the_arrival_process_does_not_change_which_images_arrive():
    """The reason arrivals get their own random substream.

    A comparison of two arrival processes has to be *only* that. If the image sequence
    moved too, any difference in outcome would confound the schedule with the case mix
    and the result would be uninterpretable.
    """
    def image_ids(scenario: ScenarioConfig) -> list[str | None]:
        rngs = build_rngs(scenario.seed)
        gen = build_patient_generator(
            scenario.patient_generator,
            LabelSchema.default_v0(),
            rngs["patient"],
            rngs["arrival"],
        )
        return [p.context.example_id for _, p in gen]

    poisson = image_ids(_scenario(None))
    metronome = image_ids(_scenario({"kind": "deterministic"}))
    assert poisson == metronome
    assert len(set(poisson)) > 1


def test_changing_the_arrival_process_does_change_the_schedule():
    def delays(scenario: ScenarioConfig) -> list[float]:
        rngs = build_rngs(scenario.seed)
        gen = build_patient_generator(
            scenario.patient_generator,
            LabelSchema.default_v0(),
            rngs["patient"],
            rngs["arrival"],
        )
        return [d for d, _ in gen]

    assert len(set(delays(_scenario({"kind": "deterministic"})))) == 1
    assert len(set(delays(_scenario(None)))) > 1


def test_factory_requires_a_rate_table_for_nhpp():
    from hai_sim.config.scenario import ArrivalConfig

    with pytest.raises(ValueError, match=r"requires arrivals\.rates"):
        build_arrivals(ArrivalConfig(kind="nhpp"), 1.0, np.random.default_rng(0))


def test_factory_builds_a_periodic_nhpp_from_a_rate_table():
    from hai_sim.config.scenario import ArrivalConfig

    process = build_arrivals(
        ArrivalConfig(kind="nhpp", rates=(1.0, 5.0), period=2.0),
        1.0,
        np.random.default_rng(0),
    )
    assert isinstance(process, NHPPArrival)
    assert process.rate_max == 5.0
