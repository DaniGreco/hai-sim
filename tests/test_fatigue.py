"""Fatigue curves and reading-time models, and how they meet in the engine."""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.core.protocols import FatigueCurve, ServiceTimeModel
from hai_sim.core.types import PatientContext, ReaderState
from hai_sim.fatigue.curves import (
    CircadianFatigue,
    CompositeFatigue,
    LinearFatigue,
    NoFatigue,
    VigilanceFatigue,
)
from hai_sim.fatigue.service import (
    ConstantServiceTime,
    FatigueModulatedServiceTime,
    LognormalServiceTime,
    ScaledServiceTime,
)

IMAGES = (np.zeros((1, 1), dtype=np.uint8),)


def _state(cases: int = 0, clock: float = 0.0, busy: float = 0.0) -> ReaderState:
    return ReaderState(cases_read=cases, clock=clock, shift_start=0.0, busy_time=busy)


def _context(**signals: float) -> PatientContext:
    return PatientContext(patient_id=0, arrival_time=0.0, signals=signals)


@pytest.mark.parametrize(
    "curve",
    [
        NoFatigue(),
        LinearFatigue(),
        VigilanceFatigue(),
        CircadianFatigue(),
        CompositeFatigue([LinearFatigue(), CircadianFatigue()]),
    ],
)
def test_every_curve_satisfies_the_protocol_and_stays_bounded(curve):
    """The level scales an accuracy penalty and a time multiplier; outside [0, 1] both
    become nonsense (a negative penalty would make fatigue *help*)."""
    assert isinstance(curve, FatigueCurve)
    for state in (_state(), _state(cases=1000, clock=97.3, busy=800.0)):
        assert 0.0 <= curve.level(state) <= 1.0


def test_reader_state_derives_its_clocks_consistently():
    state = ReaderState(cases_read=3, clock=10.0, shift_start=2.0, busy_time=4.0)
    assert state.time_on_shift == 8.0
    assert state.utilisation == pytest.approx(0.5)


def test_reader_state_utilisation_is_zero_before_time_passes():
    assert ReaderState().utilisation == 0.0


def test_reader_state_advance_keeps_the_shift_origin():
    """Advancing must not silently reset the shift, or time-on-task resets each read."""
    state = ReaderState(shift_start=100.0, clock=100.0)
    advanced = state.advanced(now=105.0, service_time=2.0)
    assert advanced.shift_start == 100.0
    assert advanced.time_on_shift == 5.0
    assert advanced.busy_time == 2.0
    assert advanced.cases_read == 1


def test_no_fatigue_is_flat():
    assert NoFatigue().level(_state(cases=10_000, clock=999.0)) == 0.0


def test_linear_fatigue_rises_then_saturates():
    curve = LinearFatigue(per_unit=0.1, basis="cases", cap=0.6)
    assert curve.level(_state(cases=3)) == pytest.approx(0.3)
    assert curve.level(_state(cases=100)) == 0.6


def test_vigilance_fatigue_front_loads_the_decrement():
    """The shape claim: more of the loss arrives in the first half than the second.

    This is what distinguishes it from a linear ramp, and it is the whole reason a
    sensitivity analysis over shapes is informative rather than decorative.
    """
    curve = VigilanceFatigue(asymptote=1.0, scale=20.0, basis="cases")
    first_half = curve.level(_state(cases=20)) - curve.level(_state(cases=0))
    second_half = curve.level(_state(cases=40)) - curve.level(_state(cases=20))
    assert first_half > 2 * second_half


def test_circadian_fatigue_depends_on_the_clock_not_on_the_workload():
    curve = CircadianFatigue(amplitude=0.3, mean=0.4, period=24.0, acrophase=5.0)
    idle_at_night = curve.level(_state(cases=0, clock=5.0))
    busy_at_noon = curve.level(_state(cases=500, clock=17.0))
    assert idle_at_night > busy_at_noon


def test_circadian_fatigue_peaks_at_the_acrophase():
    curve = CircadianFatigue(amplitude=0.3, mean=0.4, acrophase=5.0)
    peak = curve.level(_state(clock=5.0))
    assert peak >= max(curve.level(_state(clock=float(h))) for h in range(24))


def test_composite_max_is_the_binding_constraint():
    curve = CompositeFatigue(
        [LinearFatigue(per_unit=0.01, basis="cases"), CircadianFatigue(mean=0.5, amplitude=0.0)],
        combine="max",
    )
    assert curve.level(_state(cases=10)) == pytest.approx(0.5)
    assert curve.level(_state(cases=90)) == pytest.approx(0.9)


def test_composite_needs_at_least_one_curve():
    with pytest.raises(ValueError, match="at least one curve"):
        CompositeFatigue([])


def test_unknown_basis_is_reported_with_the_valid_ones():
    with pytest.raises(ValueError, match="unknown fatigue basis"):
        LinearFatigue(basis="vibes").level(_state())


@pytest.mark.parametrize(
    "model",
    [
        ConstantServiceTime(2.0),
        LognormalServiceTime(rng=np.random.default_rng(0)),
        ScaledServiceTime(ConstantServiceTime(2.0), 0.5),
        FatigueModulatedServiceTime(ConstantServiceTime(2.0), 0.5),
    ],
)
def test_every_service_time_model_satisfies_the_protocol(model):
    assert isinstance(model, ServiceTimeModel)
    assert model.sample(IMAGES, _context()) > 0.0


def test_scaled_service_time_can_model_either_sign_of_the_ai_effect():
    """The open question of thesis §5.7, made a swept parameter, not a buried constant."""
    base = ConstantServiceTime(10.0)
    assert ScaledServiceTime(base, 0.7).sample(IMAGES, _context()) == pytest.approx(7.0)
    assert ScaledServiceTime(base, 1.3).sample(IMAGES, _context()) == pytest.approx(13.0)


def test_fatigue_modulated_service_time_stretches_with_the_signal():
    model = FatigueModulatedServiceTime(ConstantServiceTime(10.0), slope=0.5)
    assert model.sample(IMAGES, _context(fatigue=0.0)) == pytest.approx(10.0)
    assert model.sample(IMAGES, _context(fatigue=1.0)) == pytest.approx(15.0)


def test_service_time_models_survive_a_run_with_no_signal_plane():
    """Which signals exist is a scenario choice; a model that hard-indexes breaks the
    moment someone runs it without a fatigue curve configured."""
    model = FatigueModulatedServiceTime(ConstantServiceTime(10.0), slope=0.5)
    assert model.sample(IMAGES, _context()) == pytest.approx(10.0)
    assert model.sample(IMAGES, None) == pytest.approx(10.0)


def test_factory_composes_fatigue_then_scale():
    from hai_sim.config.scenario import ServiceTimeConfig
    from hai_sim.factory import build_service_time

    model = build_service_time(
        ServiceTimeConfig(kind="constant", value=10.0, factor=2.0, fatigue_slope=1.0),
        np.random.default_rng(0),
    )
    # Fatigue first (10 -> 20), then the constant scale (20 -> 40): the factor must
    # multiply the read the reader would actually have taken.
    assert model.sample(IMAGES, _context(fatigue=1.0)) == pytest.approx(40.0)


def test_factory_composite_requires_parts():
    from hai_sim.config.scenario import FatigueConfig
    from hai_sim.factory import build_fatigue

    with pytest.raises(ValueError, match=r"requires fatigue\.parts"):
        build_fatigue(FatigueConfig(kind="composite"))


def test_factory_returns_none_when_fatigue_is_unconfigured():
    from hai_sim.config.scenario import FatigueConfig
    from hai_sim.factory import build_fatigue

    assert build_fatigue(None) is None
    assert build_fatigue(FatigueConfig(kind="none")) is None
