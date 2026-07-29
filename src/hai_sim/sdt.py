"""The equal-variance Gaussian signal-detection observer — shared by both readers.

SDT = signal detection theory. An observer receives noisy evidence ``x`` about whether a
finding is present and calls it positive when ``x`` clears a decision criterion ``k``. Two
parameters govern everything: **discriminability** ``d'`` (how far apart the present- and
absent-case evidence distributions sit — the reader's skill) and the **criterion** ``k``
(how much evidence the reader demands before calling positive — its bias).

Why this lives at the top level, imported by *both* the doctor and the synthetic AI: they
are the same observer at different skill. A model that "performs well" is one with a large
``d'``; a junior reader is one with a small ``d'``. Sharing the generative core is what
lets a study put either at either station and compare them on identical terms
(``../TESI.md`` §5, §8.1).

**What this buys over a fixed-rate coin flip** (`FixedAccuracyDoctor`): a genuine,
case-varying belief ``P(finding present | evidence)``. A reader who happens to draw strong
evidence is confident; one near the fence is not. That per-case variation is precisely
what turns an `IntegrationPolicy` from a *step* into a *curve* — with a constant confidence
the AI either always or never overturns a call, so a reliance study measures a threshold
rather than a reliance curve (the measured Δ=0 of ``../TESI.md`` §8.1).

The generative model, per finding, following ``../TESI.md`` §8.1:

    x       = d'_eff · y + ε           ε ~ N(0, 1),  y ∈ {0, 1}
    call    = 1[ x ≥ k ]
    belief  = sigmoid( logit(π) + d'_base · x - ½ d'_base² )

``d'_eff`` may sit below ``d'_base`` on a hard or fatigued read (see `observe`); the belief
keeps using ``d'_base``, so a reader that has quietly lost discriminability stays as
confident as if it had not — over-confidence on hard cases emerges rather than being coded
by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import NormalDist

import numpy as np

from hai_sim._numerics import logit as _logit
from hai_sim._numerics import sigmoid as _sigmoid

_ND = NormalDist()
_EPS = 1e-6


def _z(p: float) -> float:
    """The standard-normal quantile Φ⁻¹(p), clipped away from 0/1 so a perfect
    operating point maps to a large-but-finite ``d'`` instead of infinity."""
    return _ND.inv_cdf(float(np.clip(p, _EPS, 1.0 - _EPS)))


def dprime_criterion_from_accuracy(
    sensitivity: np.ndarray, specificity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Invert an operating point ``(sensitivity, specificity)`` into ``(d', k)``.

    With ``x ~ N(d'·y, 1)`` and ``call = 1[x ≥ k]``:

        hit rate    = Φ(d' - k) = sensitivity
        false alarm = Φ(-k)     = 1 - specificity

    so ``k = z(specificity)`` and ``d' = z(sensitivity) + z(specificity)``, where
    ``z = Φ⁻¹``. This is the standard 2x2-to-SDT conversion, and it is what lets the
    existing persona library (senior 0.95/0.95, junior 0.85/0.88) drive an SDT reader with
    no change to its numbers — only to what happens between them and the emitted decision.
    """
    sens = np.atleast_1d(np.asarray(sensitivity, dtype=float))
    spec = np.atleast_1d(np.asarray(specificity, dtype=float))
    z_sens = np.array([_z(s) for s in sens])
    z_spec = np.array([_z(s) for s in spec])
    return z_sens + z_spec, z_spec


def default_criterion(d_prime: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """The criterion at which ``call`` and ``belief ≥ 0.5`` agree, given a base rate.

    Solving ``logit(π) + d'·x - ½d'² = 0`` for ``x`` gives the evidence level where the
    posterior belief crosses ½, i.e. ``k = ½d' - logit(π)/d'``. Placing the criterion
    there makes the observer's *decision* consistent with its *belief* and calibrated to
    the prevalence ``π``: at a low base rate it rightly demands more evidence before
    calling a finding present, which reproduces the empirical asymmetry that rare findings
    are under- rather than over-called. A scenario may still override ``k`` to model a
    deliberately liberal or conservative reader.
    """
    d = np.asarray(d_prime, dtype=float)
    safe = np.where(d > _EPS, d, _EPS)
    return 0.5 * safe - _logit(np.asarray(prior, dtype=float)) / safe


def _as_vec(value: float | Sequence[float] | np.ndarray, n: int, what: str) -> np.ndarray:
    """Broadcast a scalar to length ``n``, or validate a vector already that length."""
    if isinstance(value, (int, float)):
        return np.full(n, float(value))
    arr = np.asarray(value, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"{what} length {arr.shape} != schema.n {n}")
    return arr


def resolve_observer(
    n: int,
    *,
    d_prime: float | Sequence[float] | None = None,
    criterion: float | Sequence[float] | None = None,
    sensitivity: float | Sequence[float] = 0.9,
    specificity: float | Sequence[float] = 0.9,
    prior: float | Sequence[float] = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve one observer's ``(d', k, prior)`` length-``n`` arrays from a scenario.

    Two ways in, because two audiences want different knobs. Give ``d_prime`` to dial the
    SDT-native capability directly (the natural axis for a quality sweep); otherwise the
    ``sensitivity``/``specificity`` operating point is inverted (the natural axis for the
    persona library, whose numbers are stated that way). ``criterion`` defaults to the
    base-rate-consistent value from `default_criterion` when a ``d'`` is given, or to the
    one implied by the operating point otherwise, and may always be overridden.
    """
    prior_arr = _as_vec(prior, n, "prior")
    if d_prime is not None:
        d = _as_vec(d_prime, n, "d_prime")
        k = default_criterion(d, prior_arr) if criterion is None else _as_vec(criterion, n, "criterion")
    else:
        d, k = dprime_criterion_from_accuracy(
            _as_vec(sensitivity, n, "sensitivity"), _as_vec(specificity, n, "specificity")
        )
        if criterion is not None:
            k = _as_vec(criterion, n, "criterion")
    return d, k, prior_arr


def observe(
    y: np.ndarray,
    *,
    d_prime: np.ndarray,
    criterion: np.ndarray,
    prior: np.ndarray,
    rng: np.random.Generator,
    d_prime_eff: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One SDT read of a per-finding truth vector → ``(labels, beliefs)``.

    ``y`` is the ground truth per finding in the ``{1, 0, -1}`` convention (``-1`` = not
    annotated). ``d_prime`` / ``criterion`` / ``prior`` are per-finding arrays of the same
    length. ``d_prime_eff`` overrides the discriminability used to *generate* the evidence
    (a hard or fatigued read), while the belief keeps using ``d_prime`` — the mechanism by
    which over-confidence on degraded reads emerges instead of being configured.

    Returns the emitted per-finding call (0/1) and the genuine belief
    ``P(finding present | evidence)`` in ``[0, 1]``. Unannotated findings (``y == -1``)
    return a benign negative with belief pinned at the prior; the metric layer skips them.
    """
    y = np.asarray(y)
    d_base = np.asarray(d_prime, dtype=float)
    d_gen = d_base if d_prime_eff is None else np.asarray(d_prime_eff, dtype=float)
    k = np.asarray(criterion, dtype=float)
    pri = np.asarray(prior, dtype=float)

    y_signal = np.where(y == 1, 1.0, 0.0)
    eps = rng.standard_normal(size=y.shape)
    x = d_gen * y_signal + eps
    call = (x >= k).astype(int)
    belief = _sigmoid(_logit(pri) + d_base * x - 0.5 * d_base**2)

    unannotated = y == -1
    call = np.where(unannotated, 0, call)
    belief = np.where(unannotated, np.clip(pri, _EPS, 1.0 - _EPS), belief)
    return call, belief
