"""Probability calibration: the maps, the fitters, and where they enter a run."""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.config.scenario import AIModelConfig, CalibrationConfig
from hai_sim.core.protocols import Calibrator
from hai_sim.factory import _build_uq, build_calibrator
from hai_sim.uq.calibration import (
    CalibratedUQ,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureScaling,
    expected_calibration_error,
    fit_isotonic,
    fit_platt,
    fit_platt_per_label,
    fit_temperature,
    fit_temperature_per_label,
    load_platt_file,
    save_platt_file,
)
from hai_sim.uq.methods import EnsembleUQ, EntropyUQ


@pytest.mark.parametrize(
    "calibrator",
    [
        IdentityCalibrator(),
        TemperatureScaling(2.0),
        PlattCalibrator(0.8, -0.2),
        IsotonicCalibrator([0.0, 0.5, 1.0], [0.0, 0.3, 1.0]),
    ],
)
def test_every_calibrator_satisfies_the_protocol_and_returns_probabilities(calibrator):
    probs = np.linspace(0.01, 0.99, 20)
    out = calibrator.calibrate(probs)
    assert isinstance(calibrator, Calibrator)
    assert out.shape == probs.shape
    assert np.all((out >= 0.0) & (out <= 1.0))


@pytest.mark.parametrize(
    "calibrator",
    [
        IdentityCalibrator(),
        TemperatureScaling(2.0),
        PlattCalibrator(0.8, -0.2),
        IsotonicCalibrator([0.0, 0.5, 1.0], [0.0, 0.3, 1.0]),
    ],
)
def test_every_calibrator_preserves_nan(calibrator):
    """A pathology no ensemble member covers is *not measured*. Mapping it to a number
    would fabricate a prediction and quietly put it back in every denominator."""
    probs = np.asarray([0.2, np.nan, 0.8])
    out = calibrator.calibrate(probs)
    assert np.isnan(out[1])
    assert np.all(np.isfinite(out[[0, 2]]))


def test_temperature_above_one_softens_toward_a_half():
    probs = np.asarray([0.99, 0.01])
    out = TemperatureScaling(3.0).calibrate(probs)
    assert out[0] < 0.99 and out[1] > 0.01


def test_temperature_scaling_cannot_change_the_ranking():
    """The property that lets discrimination and calibration be reported separately:
    fixing the numbers can neither improve nor damage an AUROC or a triage order."""
    probs = np.asarray([0.1, 0.35, 0.4, 0.9, 0.95])
    for temperature in (0.3, 1.0, 4.0):
        out = TemperatureScaling(temperature).calibrate(probs)
        assert list(np.argsort(out)) == list(np.argsort(probs))


def test_per_label_temperature_applies_columnwise():
    probs = np.asarray([[0.9, 0.9]])
    out = TemperatureScaling([1.0, 5.0]).calibrate(probs)
    assert out[0, 0] == pytest.approx(0.9)
    assert out[0, 1] < 0.9


def test_temperature_must_be_positive():
    with pytest.raises(ValueError, match="temperature must be > 0"):
        TemperatureScaling(0.0)


def test_platt_intercept_shifts_where_temperature_cannot():
    """The reason the one-parameter fix is the wrong shape for an over-calling model."""
    probs = np.asarray([0.5])
    assert TemperatureScaling(4.0).calibrate(probs)[0] == pytest.approx(0.5)
    assert PlattCalibrator(1.0, -2.0).calibrate(probs)[0] < 0.5


def _overconfident(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Labels drawn from a true probability, with the model reporting a sharpened one."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.05, 0.95, size=n)
    labels = (rng.random(n) < true_p).astype(float)
    logit = np.log(true_p / (1.0 - true_p))
    reported = 1.0 / (1.0 + np.exp(-logit * 3.0))  # temperature 1/3
    return reported, labels


def test_fit_temperature_recovers_the_sharpening_factor():
    probs, labels = _overconfident()
    assert fit_temperature(probs, labels) == pytest.approx(3.0, rel=0.15)


def test_fitted_temperature_reduces_calibration_error():
    """The end-to-end claim: fitting actually makes the numbers more believable."""
    probs, labels = _overconfident()
    before = expected_calibration_error(probs, labels)
    after = expected_calibration_error(
        TemperatureScaling(fit_temperature(probs, labels)).calibrate(probs), labels
    )
    assert after < before / 2


def test_fit_temperature_returns_identity_when_there_is_nothing_to_fit():
    """A rare pathology with no positives must yield the identity, not a fitted value
    from an empty sample — which is why a per-label fit on a small cache is mostly 1.0."""
    assert fit_temperature(np.asarray([0.3, 0.4]), np.asarray([0.0, 0.0])) == 1.0
    assert fit_temperature(np.asarray([]), np.asarray([])) == 1.0


def test_fitters_ignore_unannotated_entries():
    """-1 means 'this dataset does not label this finding', not 'negative'."""
    probs = np.asarray([0.9, 0.9, 0.9])
    labels = np.asarray([1.0, 1.0, -1.0])
    assert fit_temperature(probs, labels) == fit_temperature(probs[:2], labels[:2])


def test_fit_temperature_per_label_returns_one_value_per_column():
    probs, labels = _overconfident(n=2000)
    stacked_p = np.column_stack([probs, probs])
    stacked_y = np.column_stack([labels, np.full_like(labels, -1.0)])
    fitted = fit_temperature_per_label(stacked_p, stacked_y)
    assert fitted.shape == (2,)
    assert fitted[0] == pytest.approx(3.0, rel=0.2)
    assert fitted[1] == 1.0, "an all-unannotated column has nothing to fit"


def test_fit_platt_reduces_calibration_error_on_a_shifted_model():
    rng = np.random.default_rng(1)
    labels = (rng.random(4000) < 0.2).astype(float)
    probs = np.clip(labels * 0.3 + rng.uniform(0.4, 0.95, size=4000), 0.01, 0.99)
    a, b = fit_platt(probs, labels)
    after = expected_calibration_error(PlattCalibrator(a, b).calibrate(probs), labels)
    assert after < expected_calibration_error(probs, labels)


def test_fit_isotonic_is_monotone():
    probs, labels = _overconfident(n=1000)
    _x, y = fit_isotonic(probs, labels)
    assert np.all(np.diff(y) >= -1e-9)


def test_expected_calibration_error_is_zero_for_a_perfect_model():
    p = np.repeat([0.1, 0.5, 0.9], 1000)
    rng = np.random.default_rng(2)
    y = (rng.random(p.size) < p).astype(float)
    assert expected_calibration_error(p, y) < 0.02


def test_calibrated_uq_wraps_any_method_and_records_which_one():
    inner = EnsembleUQ(measure="mutual_information")
    wrapped = CalibratedUQ(inner, TemperatureScaling(3.0))
    members = np.asarray([[0.9, 0.1], [0.8, 0.2]])
    raw_probs, _, _ = inner.estimate(members)
    probs, unc, extra = wrapped.estimate(members)
    assert probs[0] < raw_probs[0], "softening must reach the fused probability"
    assert extra["calibrator"] == "temperature:3"
    assert unc.shape == (2,)


def test_members_stage_keeps_uncertainty_consistent_with_the_probabilities():
    """Calibrating members *before* fusion is what keeps these two describing the same
    numbers; the output stage deliberately leaves the entropy on the raw output."""
    wrapped = CalibratedUQ(EntropyUQ(), TemperatureScaling(5.0), stage="members")
    probs, unc, _ = wrapped.estimate(np.asarray([[0.99]]))
    from hai_sim.uq.methods import binary_entropy

    assert unc[0] == pytest.approx(float(binary_entropy(probs)[0]))


def test_factory_returns_a_bare_uq_when_calibration_is_unconfigured():
    """No silent wrapper: an unconfigured run must be bit-identical to before."""
    assert isinstance(_build_uq(AIModelConfig(kind="torchxrv", uq="ensemble")), EnsembleUQ)
    cfg = AIModelConfig(kind="torchxrv", calibration=CalibrationConfig(kind="identity"))
    assert isinstance(_build_uq(cfg), EntropyUQ)


def test_factory_wraps_when_calibration_is_configured():
    cfg = AIModelConfig(
        kind="torchxrv", uq="ensemble", calibration=CalibrationConfig(
            kind="temperature", temperature=2.5
        )
    )
    built = _build_uq(cfg)
    assert isinstance(built, CalibratedUQ)
    assert built.name == "ensemble:mutual_information+temperature:2.5@output"


def test_factory_isotonic_requires_fitted_knots():
    with pytest.raises(ValueError, match="knots_x and knots_y"):
        build_calibrator(CalibrationConfig(kind="isotonic"))



# --- output stage, converged Platt, per-label files ---------------------------------


def test_output_stage_calibrates_the_fused_probability_and_leaves_uncertainty_raw():
    inner = EnsembleUQ(measure="mean_entropy")
    members = np.asarray([[0.9, 0.2], [0.7, 0.4]])
    raw_probs, raw_unc, _ = inner.estimate(members)
    cal = PlattCalibrator(a=1.0, b=-2.0)
    probs, unc, extra = CalibratedUQ(inner, cal, stage="output").estimate(members)
    np.testing.assert_allclose(probs, cal.calibrate(raw_probs))
    np.testing.assert_allclose(unc, raw_unc)
    assert extra["calibration_stage"] == "output"


def test_output_stage_cannot_reorder_cases():
    """A monotone map on the fused probability keeps every case ranking intact — the
    members stage gives no such guarantee, which is why output is the default."""
    rng = np.random.default_rng(0)
    members = rng.random((200, 4, 1))
    cal = PlattCalibrator(a=1.7, b=-3.0)
    wrapped = CalibratedUQ(EnsembleUQ(measure="mean_entropy"), cal, stage="output")
    raw = np.array([EnsembleUQ(measure="mean_entropy").estimate(m)[0][0] for m in members])
    out = np.array([wrapped.estimate(m)[0][0] for m in members])
    assert np.array_equal(np.argsort(raw), np.argsort(out))


def test_calibrated_uq_rejects_an_unknown_stage():
    with pytest.raises(ValueError, match="stage"):
        CalibratedUQ(EntropyUQ(), PlattCalibrator(), stage="after")


def test_fit_platt_converges_to_the_generating_map():
    """Rare finding, logits far from zero: the case a coordinate-wise step crawls on."""
    rng = np.random.default_rng(3)
    raw = rng.beta(1.0, 12.0, size=40_000)
    true_a, true_b = 1.8, 2.0
    z = np.log(raw / (1 - raw))
    y = (rng.random(raw.size) < 1 / (1 + np.exp(-(true_a * z + true_b)))).astype(float)
    a, b = fit_platt(raw, y)
    assert a == pytest.approx(true_a, abs=0.15)
    assert b == pytest.approx(true_b, abs=0.25)
    pred = 1 / (1 + np.exp(-(a * z + b)))
    assert abs(float(np.mean(pred - y))) < 1e-6, "at the optimum mean(p) == mean(y)"
    assert abs(float(np.mean((pred - y) * z))) < 1e-6


def test_fit_platt_per_label_leaves_unfittable_columns_at_identity():
    rng = np.random.default_rng(4)
    probs = rng.random((500, 3))
    labels = np.stack(
        [
            (rng.random(500) < probs[:, 0]).astype(float),
            np.full(500, -1.0),  # unannotated
            np.zeros(500),  # single class
        ],
        axis=1,
    )
    a, b, n_pos = fit_platt_per_label(probs, labels)
    assert (a[1], b[1]) == (1.0, 0.0) and (a[2], b[2]) == (1.0, 0.0)
    assert n_pos[0] > 0 and n_pos[1] == 0 and n_pos[2] == 0


def test_platt_file_roundtrips_aligned_by_label_name(tmp_path):
    path = tmp_path / "calibration" / "toy__model.json"
    save_platt_file(
        path,
        pathologies=["A", "B"],
        a=np.array([2.0, 0.5]),
        b=np.array([-1.0, 1.0]),
        n_positive=np.array([10, 20]),
        provenance={"split": "even"},
    )
    cal = load_platt_file(path, ["B", "A"])
    np.testing.assert_allclose(cal.a, [0.5, 2.0])
    np.testing.assert_allclose(cal.b, [1.0, -1.0])
    with pytest.raises(ValueError, match="no calibration entry"):
        load_platt_file(path, ["A", "C"])

