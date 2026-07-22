"""Making a model's probabilities mean what they say. Plus how to fit them.

**Discrimination and calibration are independent, and here spectacularly so.** Measured
on 200 shared NIH images (``../TESI.md`` §8.4): MedGemma reaches an expected calibration
error (ECE) of 0.798 — of the cases it calls 0.95, about 4% are positive — while
*beating* the DenseNet ensemble on several pathologies by area under the receiver
operating characteristic curve (AUROC). It ranks well and its numbers are nearly
meaningless. The ensemble is better but still badly over-confident at ECE 0.266.

That matters because every rule that mixes the doctor's belief with the AI's on a
common scale is only as sound as the number. §8.2 calls temperature scaling *"the most
significant technical debt in the code"* and states that without it a log-odds fusion
**can be worse than the doctor alone** — an over-confident model contributes a huge
log-odds term and simply overrides its partner. With ECE 0.8 the fusion is not a
compromise, it is a takeover.

Applied to the ``(n_members, n_labels)`` stack **before** fusion, so the uncertainty the
`UQMethod` derives describes the probabilities the run actually uses. Calibrating after
fusion would leave the reported uncertainty describing numbers nobody used.

Four maps and three fitters:

``IdentityCalibrator``  — the null model; also what an unconfigured run gets.
``TemperatureScaling``  — one parameter, in log-odds space (Guo et al. 2017). Cannot
                          change the ranking, so AUROC is invariant and only the
                          numbers move. The standard first thing to try, and per-label
                          temperatures are supported because pathologies differ.
``PlattCalibrator``     — two parameters (slope and intercept); can shift as well as
                          sharpen, which matters when a model is biased toward calling
                          findings present, as MedGemma measurably is.
``IsotonicCalibrator``  — non-parametric and monotone; fits any shape given enough
                          data, and overfits happily when given little.

⚠️ **A calibrator is fitted on data and must be fitted on data the run does not score.**
Fitting on the same images you then report is circular. And calibration is a property of
*(model + readout + prompt + population)*, not of a model in the abstract: a temperature
fitted on NIH is not valid for CheXpert, and for a generative backend it is not even
valid for a different prompt.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, _EPS, 1.0 - _EPS)
    return np.asarray(np.log(q / (1.0 - q)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-z)))


def _preserving_nan(probs: np.ndarray, mapped: np.ndarray) -> np.ndarray:
    """Keep NaN where it was: an uncovered pathology must not gain a fabricated value."""
    return np.where(np.isnan(probs), np.nan, mapped)


class IdentityCalibrator:
    """Leaves probabilities untouched. The null model, and the default."""

    name = "identity"

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        return np.asarray(probs, dtype=float)


class TemperatureScaling:
    """Divide the log-odds by ``T`` (Guo et al. 2017, *On calibration of modern NNs*).

    ``T > 1`` softens an over-confident model toward 0.5; ``T < 1`` sharpens an
    under-confident one. One parameter, fitted by minimising negative log-likelihood.

    **It is monotone, so it cannot change any ranking.** AUROC, the error-capture curve
    and the triage order are all invariant under it — only the probabilities move. That
    is a feature: it means fixing calibration cannot silently improve or damage a
    discrimination result, and the two can be reported independently.

    ``temperature`` may be a scalar or one value per label. Per-label is usually right
    for a multi-label chest X-ray model, where a common pathology and a rare one are
    miscalibrated to quite different degrees.
    """

    def __init__(self, temperature: float | Sequence[float] = 1.0) -> None:
        arr = np.asarray(temperature, dtype=float)
        if np.any(arr <= 0):
            raise ValueError(f"temperature must be > 0, got {temperature!r}")
        self.temperature = arr
        label = f"{float(arr):g}" if arr.ndim == 0 else f"per-label[{arr.size}]"
        self.name = f"temperature:{label}"

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        arr = np.asarray(probs, dtype=float)
        return _preserving_nan(arr, _sigmoid(_logit(arr) / self.temperature))


class PlattCalibrator:
    """Affine in log-odds space: ``sigmoid(a * logit(p) + b)`` (Platt 1999).

    The extra intercept is what temperature scaling lacks: ``b`` shifts the whole
    distribution, which is what a model that systematically over-calls findings needs.
    MedGemma is measurably such a model — it errs on 83% of entries against the
    ensemble's 23%, over-calling rather than missing — so the one-parameter fix is the
    wrong shape for it.

    Still monotone in ``p`` when ``a > 0``, so rankings again survive.
    """

    def __init__(
        self,
        a: float | Sequence[float] = 1.0,
        b: float | Sequence[float] = 0.0,
    ) -> None:
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.name = "platt"

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        arr = np.asarray(probs, dtype=float)
        return _preserving_nan(arr, _sigmoid(self.a * _logit(arr) + self.b))


class IsotonicCalibrator:
    """Piecewise-linear monotone map through fitted knots — non-parametric calibration.

    Fits any monotone shape, which is its strength and its risk: with a few hundred
    images and a rare pathology it will happily reproduce the noise. Prefer temperature
    scaling unless you can show it is insufficient, and never fit this per-label on a
    200-image cache.

    Stored as knots so the fitted object is data, not a scikit-learn dependency.
    """

    def __init__(self, x: Sequence[float], y: Sequence[float]) -> None:
        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)
        if xs.shape != ys.shape or xs.ndim != 1 or xs.size == 0:
            raise ValueError("isotonic knots must be two equal-length 1-D sequences")
        order = np.argsort(xs)
        self.x = xs[order]
        self.y = ys[order]
        self.name = f"isotonic:{xs.size}knots"

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        arr = np.asarray(probs, dtype=float)
        filled = np.where(np.isnan(arr), 0.5, arr)
        return _preserving_nan(arr, np.interp(filled, self.x, self.y))


class CalibratedUQ:
    """Wraps a `UQMethod`, calibrating member probabilities before it sees them.

    This is where calibration enters a run, and it is one object rather than an edit to
    each backend: every actor that produces uncertainty does so through a `UQMethod`, so
    wrapping the method covers the DenseNet, MedGemma and cache replay at once.

    Members are calibrated **individually** before fusion. Doing it the other way —
    fusing then calibrating — would leave the entropy decomposition describing the raw
    probabilities while the run used the calibrated ones, and the epistemic term would
    no longer be the disagreement between the numbers anyone saw.
    """

    def __init__(self, inner: object, calibrator: object) -> None:
        self.inner = inner
        self.calibrator = calibrator
        inner_name = getattr(inner, "name", type(inner).__name__)
        cal_name = getattr(calibrator, "name", type(calibrator).__name__)
        self.name = f"{inner_name}+{cal_name}"

    def estimate(
        self, member_probs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        calibrated = self.calibrator.calibrate(np.asarray(member_probs, dtype=float))  # type: ignore[attr-defined]
        probs, unc, extra = self.inner.estimate(calibrated)  # type: ignore[attr-defined]
        extra = dict(extra)
        extra["calibrator"] = getattr(self.calibrator, "name", "?")
        return probs, unc, extra


# --- Fitting ----------------------------------------------------------------
# Deliberately dependency-free: these run on the inference caches already on disk,
# and a calibrator that needed scipy or scikit-learn would not survive the
# inference-only constraint the project is built under.


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean binary negative log-likelihood over annotated, covered entries."""
    q = np.clip(probs, _EPS, 1.0 - _EPS)
    return float(-np.mean(labels * np.log(q) + (1.0 - labels) * np.log(1.0 - q)))


def _valid(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop NaN predictions and unannotated (-1) truths — never impute either."""
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    if p.shape != y.shape:
        raise ValueError(f"probs {p.shape} and labels {y.shape} must have equal size")
    keep = np.isfinite(p) & (y >= 0)
    return p[keep], y[keep]


def fit_temperature(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.05, 20.0),
    tolerance: float = 1e-4,
) -> float:
    """Fit one temperature by golden-section search on negative log-likelihood.

    The objective is unimodal in ``log T``, so golden section converges without
    gradients and without a dependency. Returns 1.0 when there is nothing to fit
    (no valid entries, or a degenerate single-class column) rather than a fitted
    value from an empty sample.
    """
    p, y = _valid(probs, labels)
    if p.size == 0 or np.all(y == y[0]):
        return 1.0
    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0

    def objective(log_t: float) -> float:
        return _nll(_sigmoid(_logit(p) / np.exp(log_t)), y)

    a, b = lo, hi
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fd = objective(c), objective(d)
    while abs(b - a) > tolerance:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = objective(d)
    return float(np.exp((a + b) / 2.0))


def fit_temperature_per_label(
    probs: np.ndarray,
    labels: np.ndarray,
    **kwargs: object,
) -> np.ndarray:
    """One temperature per column of an ``(n_images, n_labels)`` pair of arrays.

    A pathology with no annotated positives (or none annotated at all) gets 1.0 — the
    identity — because there is nothing to fit. That is the honest outcome for the rare
    findings in a small cache, and it is why a per-label fit on 200 images is mostly
    identity with a few real values.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.ndim != 2 or p.shape != y.shape:
        raise ValueError(f"expected two (n_images, n_labels) arrays, got {p.shape}/{y.shape}")
    return np.asarray(
        [fit_temperature(p[:, j], y[:, j], **kwargs) for j in range(p.shape[1])]  # type: ignore[arg-type]
    )


def fit_platt(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    iterations: int = 100,
    learning_rate: float = 0.1,
) -> tuple[float, float]:
    """Fit ``(a, b)`` by Newton steps on the logistic negative log-likelihood."""
    p, y = _valid(probs, labels)
    if p.size == 0 or np.all(y == y[0]):
        return 1.0, 0.0
    z = _logit(p)
    a, b = 1.0, 0.0
    for _ in range(iterations):
        pred = _sigmoid(a * z + b)
        resid = pred - y
        grad_a = float(np.mean(resid * z))
        grad_b = float(np.mean(resid))
        w = pred * (1.0 - pred)
        # Diagonal Newton: the full 2x2 Hessian buys little here and the diagonal
        # keeps the step well-defined when the weights collapse near 0 or 1.
        hess_a = float(np.mean(w * z * z)) + 1e-9
        hess_b = float(np.mean(w)) + 1e-9
        a -= learning_rate * grad_a / hess_a
        b -= learning_rate * grad_b / hess_b
    return float(a), float(b)


def fit_isotonic(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators fit; returns ``(x, y)`` knots for `IsotonicCalibrator`."""
    p, y = _valid(probs, labels)
    if p.size == 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])
    order = np.argsort(p, kind="stable")
    xs, ys = p[order], y[order]
    # Each block holds (sum of targets, count); merge left while monotonicity is violated.
    blocks: list[list[float]] = []
    for value in ys:
        blocks.append([float(value), 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            total, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += count
    fitted = np.concatenate(
        [np.full(int(count), total / count) for total, count in blocks]
    )
    return xs, fitted


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 10
) -> float:
    """ECE: mean |confidence - accuracy| over equal-width bins, weighted by bin size.

    The number quoted in §8.4. Reported on the *positive-class* probability rather than
    on max(p, 1-p), which is the right convention for multi-label detection where the
    negative class dominates and the max would be near-perfectly calibrated by default.
    """
    p, y = _valid(probs, labels)
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        total += float(np.sum(mask)) / p.size * abs(float(np.mean(p[mask]) - np.mean(y[mask])))
    return total
