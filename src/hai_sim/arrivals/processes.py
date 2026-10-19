"""Three arrival processes behind one contract, plus the rate function they share.

Splitting *when* a patient arrives from *who* arrives is what lets the same dataset
stream be driven by a flat rate, a time-varying one, or a metronome, with no change to
the generator that samples images.

Why it matters more than it looks: **order effects only exist under saturation**. A
triage rule can only reorder patients who are already waiting, so a study of triage is
a study of a *loaded* department, and load is exactly what the arrival process
controls. Getting arrivals wrong does not add noise to a triage result — it removes the
phenomenon.

The policy is a *testable* one: start from a
homogeneous Poisson process, escalate to a non-homogeneous one (NHPP) with a
piecewise-constant 24x7 rate, and go further only if a conditional
Kolmogorov-Smirnov test rejects — test, don't assume. Kim & Whitt (2014) explain the caution:
much of the "burstiness" reported in hospital arrivals is an **artefact of
aggregation**, since pooling the same hour across weeks of differing intensity
manufactures overdispersion that was never in the data. A framework that hardcoded an
overdispersed process would bake that artefact in.

The three:

``PoissonArrival``
    Constant rate, exponential gaps, memoryless. The baseline, kept for
    comparison even after escalating.

``NHPPArrival``
    Time-varying rate via **thinning** (Lewis & Shedler 1979): propose arrivals from a
    homogeneous process at the peak rate, keep each with probability
    ``lambda(t) / lambda_max``. Exact, no numerical integration, and it works with any
    rate function — including one that is not analytic.

``DeterministicArrival``
    Evenly spaced. Not a claim about reality: it is the control that removes arrival
    randomness so a queue effect can be attributed to the service side alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


class PoissonArrival:
    """Homogeneous Poisson process: exponential gaps at a constant rate.

    ``rate`` is arrivals per unit of simulation time, so the mean gap is ``1 / rate``.
    """

    def __init__(self, rate: float, rng: np.random.Generator | None = None) -> None:
        if rate <= 0:
            raise ValueError(f"arrival rate must be > 0, got {rate}")
        self.rate = float(rate)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.name = f"poisson:{rate:g}"

    def next_delay(self, now: float) -> float:
        return float(self.rng.exponential(1.0 / self.rate))


class DeterministicArrival:
    """Evenly spaced arrivals — the control that removes arrival randomness.

    Useful for isolating queue effects: with gaps fixed, any variation in waiting time
    comes from the service side, so a triage result cannot be an artefact of a lucky
    or unlucky arrival sequence.
    """

    def __init__(self, rate: float) -> None:
        if rate <= 0:
            raise ValueError(f"arrival rate must be > 0, got {rate}")
        self.rate = float(rate)
        self.name = f"deterministic:{rate:g}"

    def next_delay(self, now: float) -> float:
        return 1.0 / self.rate


class PiecewiseConstantRate:
    """A rate that is constant within each slot and repeats with a period.

    The usual shape is 24 hourly rates per day of week, i.e. 168
    slots over a period of 168 time units when one unit is one hour. Following De
    Santis et al. (2022), stratifying by day of week is *necessary rather than
    optional* — a Monday morning and a Sunday morning are not draws from the same
    distribution, and pooling them is one of the ways aggregation manufactures
    overdispersion.

    Callable, so it can be handed straight to `NHPPArrival`, and it exposes ``peak``
    so thinning needs no separate bound.
    """

    def __init__(self, rates: Sequence[float], period: float | None = None) -> None:
        arr = np.asarray(rates, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("rates must be a non-empty 1-D sequence")
        if np.any(arr < 0):
            raise ValueError("rates must be >= 0")
        if not np.any(arr > 0):
            raise ValueError("at least one slot must have a positive rate")
        self.rates = arr
        # Default: one time unit per slot, so 168 hourly rates span a week of hours.
        self.period = float(period) if period is not None else float(arr.size)
        self.slot_width = self.period / arr.size
        self.name = f"piecewise:{arr.size}slots/period={self.period:g}"

    @property
    def peak(self) -> float:
        return float(np.max(self.rates))

    def __call__(self, t: float) -> float:
        phase = float(t) % self.period
        index = int(phase / self.slot_width)
        # Guard the closed upper edge: floating point can land exactly on `period`.
        return float(self.rates[min(index, self.rates.size - 1)])


class NHPPArrival:
    """Non-homogeneous Poisson process by thinning (Lewis & Shedler 1979).

    Propose candidate arrivals from a homogeneous process at ``rate_max``, accept each
    with probability ``rate(t) / rate_max``, and the accepted points are an exact draw
    from the process with intensity ``rate(t)``. Exact rather than approximate, and it
    needs no integration of the rate — which is what makes an empirical, tabulated
    rate as easy to use as an analytic one.

    ``rate`` may be any callable; pass a `PiecewiseConstantRate` and ``rate_max`` is
    taken from it. Provide ``rate_max`` explicitly for an analytic rate, and make sure
    it really is an upper bound: thinning with too small a bound silently produces the
    wrong process rather than failing.
    """

    def __init__(
        self,
        rate: Callable[[float], float],
        rate_max: float | None = None,
        rng: np.random.Generator | None = None,
        max_rejections: int = 100_000,
    ) -> None:
        peak = rate_max if rate_max is not None else getattr(rate, "peak", None)
        if peak is None:
            raise ValueError(
                "NHPPArrival needs rate_max: an upper bound on rate(t). A "
                "PiecewiseConstantRate supplies its own via .peak; an analytic rate "
                "must state one, and it must genuinely bound the function — thinning "
                "with too low a bound yields the wrong process, silently."
            )
        if peak <= 0:
            raise ValueError(f"rate_max must be > 0, got {peak}")
        self.rate = rate
        self.rate_max = float(peak)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.max_rejections = int(max_rejections)
        self.name = f"nhpp:{getattr(rate, 'name', 'callable')}"

    def next_delay(self, now: float) -> float:
        t = float(now)
        elapsed = 0.0
        for _ in range(self.max_rejections):
            gap = float(self.rng.exponential(1.0 / self.rate_max))
            t += gap
            elapsed += gap
            current = float(self.rate(t))
            if current > self.rate_max + 1e-9:
                raise ValueError(
                    f"rate({t:g}) = {current:g} exceeds rate_max = {self.rate_max:g}; "
                    "the bound is wrong and the sampled process is not the one asked for"
                )
            if current > 0.0 and self.rng.random() < current / self.rate_max:
                return elapsed
        raise RuntimeError(
            f"no arrival accepted in {self.max_rejections} proposals from t={now:g}. "
            "The rate is effectively zero over a long stretch; either the run starts "
            "in a closed period or the rate table is mostly zeros."
        )
