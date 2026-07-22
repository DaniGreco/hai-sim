"""How worn a reading station is: four shapes plus a way to combine them.

**Why this is a seam and not a constant.** ``../TESI.md`` §10 declares as one of the
three methodological holes that *no analytic fatigue curve exists for radiology* — the
literature offers isolated pre/post measurements (Krupinski 2010/2012, Vosshenrich
2021, Hanna 2023) and no functional form. Its own conclusion is that the shape must be
imported from the human-factors literature and that **a sensitivity analysis over the
shape is mandatory**. A mandatory sensitivity analysis over shapes is a requirement
that the shape be swappable, which is what this module provides.

The level is always in ``[0, 1]``: 0 = fresh, 1 = as worn as this curve goes. What it
*does* is deliberately elsewhere — it reaches the reader as
``PatientContext.signals["fatigue"]``, where a `ServiceTimeModel` may stretch the read
and a reader may degrade its own accuracy. Keeping "how tired" apart from "what tired
does" is what lets one sensitivity analysis vary the shape and another vary the
consequence.

The four:

``NoFatigue``
    Always 0. The null model, and the default, so nothing changes for a scenario that
    does not ask for fatigue.

``LinearFatigue``
    Rises linearly in cases read and/or elapsed time until it saturates. The simplest
    thing that could be true, and the right first sensitivity anchor.

``VigilanceFatigue``
    Exponential approach to an asymptote — the **vigilance decrement** shape from the
    sustained-attention literature (Mackworth). Most of the loss happens early and
    then plateaus, which is qualitatively different from linear and is the shape the
    human-factors work actually reports.

``CircadianFatigue``
    A sinusoid on the wall clock (Akerstedt-Folkard): fatigue depends on the *hour*,
    not on what you have done. This is the one that makes a night shift different from
    a morning shift regardless of workload.

⚠️ **Time has no units here.** A curve's parameters are in simulation time units, and
the scenario decides what a unit means. The defaults assume **one unit = one hour**,
which makes ``period: 24`` a day. State the convention in any write-up: a fatigue curve
quoted without its time unit is uninterpretable.

⚠️ **Skinner & Giesbrecht (2025) is the reason a single "accuracy multiplier" is not
enough** (§4.2): the vigilance decrement is a shift in *both* discriminability and
criterion, not a uniform drop in detection rate. These curves produce the level; a
reader that treats it as a pure sensitivity penalty is modelling only half the effect,
and should say so.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from hai_sim.core.types import ReaderState

Basis = str  # "cases" | "time_on_shift" | "busy_time" | "utilisation"


def _basis_value(state: ReaderState, basis: Basis) -> float:
    """Read the quantity a curve advances against."""
    if basis == "cases":
        return float(state.cases_read)
    if basis == "time_on_shift":
        return float(state.time_on_shift)
    if basis == "busy_time":
        return float(state.busy_time)
    if basis == "utilisation":
        return float(state.utilisation)
    raise ValueError(
        f"unknown fatigue basis {basis!r}; expected one of "
        "'cases', 'time_on_shift', 'busy_time', 'utilisation'"
    )


class NoFatigue:
    """Always fresh. The null model, and what every scenario gets by default."""

    name = "none"

    def level(self, state: ReaderState) -> float:
        return 0.0


class LinearFatigue:
    """Rises linearly against ``basis`` until it saturates at ``cap``.

    ``per_unit`` is how much fatigue one unit of the basis adds — one more case read,
    or one more time unit on shift. The simplest shape that could be true, which is
    exactly what a sensitivity analysis needs as its anchor: if a conclusion survives
    both this and `VigilanceFatigue`, the shape was not driving it.
    """

    def __init__(
        self,
        per_unit: float = 0.01,
        basis: Basis = "cases",
        cap: float = 1.0,
    ) -> None:
        if per_unit < 0:
            raise ValueError(f"per_unit must be >= 0, got {per_unit}")
        if not 0.0 <= cap <= 1.0:
            raise ValueError(f"cap must be in [0, 1], got {cap}")
        self.per_unit = float(per_unit)
        self.basis = basis
        self.cap = float(cap)
        self.name = f"linear:{per_unit:g}/{basis}"

    def level(self, state: ReaderState) -> float:
        return min(self.cap, self.per_unit * _basis_value(state, self.basis))


class VigilanceFatigue:
    """Exponential approach to an asymptote — the vigilance-decrement shape.

    ``level = asymptote * (1 - exp(-x / scale))``, where ``x`` is the basis value.
    Most of the decrement arrives early and then plateaus, which is what the
    sustained-attention literature reports and what distinguishes this from a linear
    ramp: the difference between the two shapes is largest in the *first* part of a
    shift, which is when most reads happen.

    ``scale`` is the basis value at which roughly 63% of the asymptotic decrement has
    been reached, so ``scale=40`` with ``basis="cases"`` means most of the loss is in
    by the fortieth study.
    """

    def __init__(
        self,
        asymptote: float = 1.0,
        scale: float = 40.0,
        basis: Basis = "cases",
    ) -> None:
        if not 0.0 <= asymptote <= 1.0:
            raise ValueError(f"asymptote must be in [0, 1], got {asymptote}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        self.asymptote = float(asymptote)
        self.scale = float(scale)
        self.basis = basis
        self.name = f"vigilance:{asymptote:g}@{scale:g}/{basis}"

    def level(self, state: ReaderState) -> float:
        x = _basis_value(state, self.basis)
        return float(self.asymptote * (1.0 - math.exp(-x / self.scale)))


class CircadianFatigue:
    """A sinusoid on the wall clock: tiredness as a function of the hour.

    ``level = mean + amplitude * cos(2*pi*(clock - acrophase) / period)``, clipped to
    ``[0, 1]``. ``acrophase`` is the time of peak fatigue — the default 5.0 puts it at
    05:00 with a 24-unit period, i.e. the small hours, which is where the shift-work
    literature places the trough of alertness.

    Distinct from every other curve here in that it depends on ``clock`` alone: a
    reader who has done nothing all night is still impaired. That is the point, and it
    is why this composes with a time-on-task curve rather than replacing it.
    """

    def __init__(
        self,
        amplitude: float = 0.3,
        mean: float = 0.3,
        period: float = 24.0,
        acrophase: float = 5.0,
    ) -> None:
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")
        if amplitude < 0:
            raise ValueError(f"amplitude must be >= 0, got {amplitude}")
        self.amplitude = float(amplitude)
        self.mean = float(mean)
        self.period = float(period)
        self.acrophase = float(acrophase)
        self.name = f"circadian:{amplitude:g}+-{mean:g}@{period:g}h"

    def level(self, state: ReaderState) -> float:
        phase = 2.0 * math.pi * (state.clock - self.acrophase) / self.period
        return float(min(1.0, max(0.0, self.mean + self.amplitude * math.cos(phase))))


class CompositeFatigue:
    """Combine several curves — because the real thing is more than one mechanism.

    A night shift is circadian *and* time-on-task, and the two advance against
    different clocks. ``combine="max"`` (default) reads as "the binding constraint";
    ``"sum"`` as additive load, clipped at 1; ``"mean"`` as an average of equally
    plausible shapes, which is the conservative choice when a sensitivity analysis has
    not picked a winner.
    """

    def __init__(self, parts: Sequence[object], combine: str = "max") -> None:
        if not parts:
            raise ValueError("CompositeFatigue needs at least one curve")
        if combine not in ("max", "sum", "mean"):
            raise ValueError(f"unknown combine {combine!r}; use max, sum or mean")
        self.parts = list(parts)
        self.combine = combine
        inner = "+".join(getattr(p, "name", type(p).__name__) for p in self.parts)
        self.name = f"composite[{combine}]:{inner}"

    def level(self, state: ReaderState) -> float:
        levels = [float(p.level(state)) for p in self.parts]  # type: ignore[attr-defined]
        if self.combine == "max":
            value = max(levels)
        elif self.combine == "sum":
            value = sum(levels)
        else:
            value = sum(levels) / len(levels)
        return min(1.0, max(0.0, value))
