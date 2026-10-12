"""Making a model's probabilities mean what they say. Plus how to fit them.

**Discrimination and calibration are independent, and here spectacularly so.** Measured
during development on 200 shared NIH images: MedGemma reaches an expected calibration
error (ECE) of 0.798 — of the cases it calls 0.95, about 4% are positive — while
*beating* the DenseNet ensemble on several pathologies by area under the receiver
operating characteristic curve (AUROC). It ranks well and its numbers are nearly
meaningless. The ensemble is better but still badly over-confident at ECE 0.266.

That matters because every rule that mixes the doctor's belief with the AI's on a
common scale is only as sound as the number. Without calibration a log-odds fusion
**can be worse than the doctor alone** (thesis §2.4) — an over-confident model contributes a huge
log-odds term and simply overrides its partner. With ECE 0.8 the fusion is not a
compromise, it is a takeover.

**Where it is applied is a choice with two defensible answers** (`CalibratedUQ.stage`):

- ``output`` (default) — calibrate the *fused* probability, the number a reader is shown.
  Pool-then-calibrate is what makes that number honest: the mean of individually
  calibrated members is itself under-confident and would need a second map anyway. The
  map is monotone on the final probability, so the ranking of cases by probability is
  untouched, and a single-member model (MedGemma) is treated exactly like an ensemble.
  The uncertainty stays the one derived from the raw members.
- ``members`` — calibrate the ``(n_members, n_labels)`` stack first, so the uncertainty
  the `UQMethod` derives describes calibrated numbers. Use it to study the uncertainty
  itself: an entropy is *not* invariant to calibration (it peaks at 0.5, and calibration
  moves which cases sit there), so which uncertainty measure ranks errors best can change.

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

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, _EPS, 1.0 - _EPS)
    return np.asarray(np.log(q / (1.0 - q)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0))))


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
    """Wraps a `UQMethod` and calibrates at one of two stages (see the module docstring).

    This is where calibration enters a run, and it is one object rather than an edit to
    each backend: every actor that produces uncertainty does so through a `UQMethod`, so
    wrapping the method covers the DenseNet, MedGemma and cache replay at once.

    ``stage="output"`` fuses the raw members, then calibrates the fused probability and
    leaves the uncertainty as derived from the raw members. ``stage="members"`` calibrates
    each member first, so both the fused probability and the uncertainty describe
    calibrated numbers.
    """

    def __init__(self, inner: object, calibrator: object, stage: str = "output") -> None:
        if stage not in ("output", "members"):
            raise ValueError(f"stage must be 'output' or 'members', got {stage!r}")
        self.inner = inner
        self.calibrator = calibrator
        self.stage = stage
        inner_name = getattr(inner, "name", type(inner).__name__)
        cal_name = getattr(calibrator, "name", type(calibrator).__name__)
        self.name = f"{inner_name}+{cal_name}@{stage}"

    def estimate(
        self, member_probs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        stack = np.asarray(member_probs, dtype=float)
        if self.stage == "members":
            stack = self.calibrator.calibrate(stack)  # type: ignore[attr-defined]
        probs, unc, extra = self.inner.estimate(stack)  # type: ignore[attr-defined]
        if self.stage == "output":
            probs = self.calibrator.calibrate(np.asarray(probs, dtype=float))  # type: ignore[attr-defined]
        extra = dict(extra)
        extra["calibrator"] = getattr(self.calibrator, "name", "?")
        extra["calibration_stage"] = self.stage
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
    iterations: int = 200,
    tol: float = 1e-9,
) -> tuple[float, float]:
    """Fit ``(a, b)`` by Newton steps on the logistic negative log-likelihood.

    Full 2x2 Newton with step halving, run to convergence (gradient norm below ``tol``)
    rather than for a fixed count: slope and intercept are strongly correlated when the
    logits sit far from zero, which is the normal case for a rare finding, and a
    coordinate-wise step then crawls. The loss is convex, so the optimum is unique.
    """
    p, y = _valid(probs, labels)
    if p.size == 0 or np.all(y == y[0]):
        return 1.0, 0.0
    z = _logit(p)

    def nll(a: float, b: float) -> float:
        return _nll(_sigmoid(a * z + b), y)

    a, b = 1.0, 0.0
    loss = nll(a, b)
    for _ in range(iterations):
        pred = _sigmoid(a * z + b)
        resid = pred - y
        grad = np.array([np.mean(resid * z), np.mean(resid)])
        if float(np.linalg.norm(grad)) < tol:
            break
        w = pred * (1.0 - pred)
        hess = np.array(
            [[np.mean(w * z * z), np.mean(w * z)], [np.mean(w * z), np.mean(w)]]
        ) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        scale = 1.0
        cand = loss
        while scale > 1e-6:
            cand = nll(a - scale * step[0], b - scale * step[1])
            if cand <= loss:
                break
            scale *= 0.5
        else:
            break
        a, b, loss = a - scale * step[0], b - scale * step[1], cand
    return float(a), float(b)


def fit_platt_per_label(
    probs: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One ``(a, b)`` per label column, returned as ``(a, b, n_positive)`` of length ``L``.

    A column with nothing to fit (unannotated, uncovered, or single-class) comes back as
    the identity ``(1, 0)``. ``n_positive`` is returned because it is what decides
    whether a fit is worth trusting: two parameters are well determined with a few
    hundred positives and fragile below a few dozen.
    """
    arr = np.asarray(probs, dtype=float)
    lab = np.asarray(labels, dtype=float)
    if arr.ndim != 2 or arr.shape != lab.shape:
        raise ValueError(f"expected matching (N, L) arrays, got {arr.shape} and {lab.shape}")
    n_labels = arr.shape[1]
    a, b = np.ones(n_labels), np.zeros(n_labels)
    n_pos = np.zeros(n_labels, dtype=int)
    for k in range(n_labels):
        _, y = _valid(arr[:, k], lab[:, k])
        n_pos[k] = int(np.sum(y == 1))
        a[k], b[k] = fit_platt(arr[:, k], lab[:, k])
    return a, b, n_pos


def save_platt_file(
    path: str | Path,
    *,
    pathologies: Sequence[str],
    a: np.ndarray,
    b: np.ndarray,
    n_positive: np.ndarray,
    prevalence: np.ndarray,
    provenance: dict[str, object],
    threshold_raw: np.ndarray | None = None,
    threshold_calibrated: np.ndarray | None = None,
) -> None:
    """Write a per-label Platt calibration next to the cache it was fitted on.

    A file rather than a YAML block because it is *data*: 2 x L numbers per
    (model, dataset), fitted once and reused by every study of that cell. ``provenance``
    records which caches and which split produced it, so a study can be checked against
    the rule that calibration is never fitted on the images it scores.
    """
    payload = {
        "kind": "platt",
        "provenance": provenance,
        "per_label": {
            name: {
                "a": float(a[k]),
                "b": float(b[k]),
                "n_positive": int(n_positive[k]),
                # The base rate the calibrated probability is anchored to: what an
                # evidence-based fusion subtracts to recover the likelihood ratio.
                "prevalence": None if not np.isfinite(prevalence[k]) else float(prevalence[k]),
                # The model's own operating point: the cut at which it calls as many positives
                # as there are (prevalence-matched). Needs no cost, and the raw and the
                # calibrated value are the same cut on two scales.
                "threshold_raw": _opt(threshold_raw, k),
                "threshold_calibrated": _opt(threshold_calibrated, k),
            }
            for k, name in enumerate(pathologies)
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _opt(values: np.ndarray | None, k: int) -> float | None:
    return None if values is None or not np.isfinite(values[k]) else float(values[k])


def load_thresholds(path: str | Path, labels: Sequence[str], *, calibrated: bool) -> np.ndarray:
    """Per-label prevalence-matched decision threshold; 0.5 where the file has none."""
    per_label = json.loads(Path(path).read_text(encoding="utf-8"))["per_label"]
    key = "threshold_calibrated" if calibrated else "threshold_raw"
    out = np.full(len(labels), 0.5)
    for k, name in enumerate(labels):
        value = per_label.get(name, {}).get(key)
        if value is not None:
            out[k] = float(value)
    return out


def load_prevalence(path: str | Path, labels: Sequence[str]) -> np.ndarray:
    """Per-label prevalence stored in a calibration file; NaN where it was not annotated."""
    per_label = json.loads(Path(path).read_text(encoding="utf-8"))["per_label"]
    out = np.full(len(labels), np.nan)
    for k, name in enumerate(labels):
        value = per_label.get(name, {}).get("prevalence")
        if value is not None and value > 0.0:
            out[k] = float(value)
    return out


def load_platt_file(path: str | Path, labels: Sequence[str]) -> PlattCalibrator:
    """Load a `save_platt_file` calibration, aligned to ``labels`` by name."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "platt":
        raise ValueError(f"{path}: expected kind 'platt', got {payload.get('kind')!r}")
    per_label = payload["per_label"]
    missing = [name for name in labels if name not in per_label]
    if missing:
        raise ValueError(f"{path}: no calibration entry for label(s) {missing}")
    cal = PlattCalibrator(
        a=[per_label[name]["a"] for name in labels],
        b=[per_label[name]["b"] for name in labels],
    )
    cal.name = f"platt:{Path(path).stem}"
    return cal


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

    The number quoted at the top of this module. Reported on the *positive-class*
    probability rather than on max(p, 1-p), which is the right convention for multi-label
    detection where the
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
