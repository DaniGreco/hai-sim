"""How long one read takes — the quantity that creates the queue.

Reading time used to be a hardcoded lognormal draw inside each actor. Pulling it out is
not tidying: three separate open questions in ``../TESI.md`` all turn on it, and none
of them can be answered by editing a doctor.

1. **Does AI assistance speed a reader up or slow them down?** §9 lists the sign of the
   modifier as unresolved and says explicitly that it is *"resolvable only by
   sensitivity analysis"*. Seah 2021 measured 81 s -> 56 s; the verification-cost
   argument predicts the opposite. `ScaledServiceTime` makes the factor a swept
   parameter instead of a buried constant.
2. **Does acute fatigue slow reading, or only degrade accuracy?** §6.7 records this as
   the one point left unresolved when the two doctor designs were reconciled: the older
   design wanted two regimes (mild fatigue = accuracy only, acute = accuracy *and*
   time), and the signal-detection model that superseded it has no time dimension at
   all. `FatigueModulatedServiceTime` is that missing dimension.
3. **What does an empirical reading-time distribution look like?** Lognormal is the
   conventional answer and the default here, but it is a modelling choice, and one a
   study should be able to challenge without a code change.

Every model reads its inputs from the context's signal plane, so it sees the same
``fatigue`` the reader sees, and none of them needs to know which station it is on.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hai_sim.core.types import Image, PatientContext


class ConstantServiceTime:
    """Every read takes the same time. The control: removes service variability."""

    def __init__(self, value: float = 1.0) -> None:
        if value < 0:
            raise ValueError(f"service time must be >= 0, got {value}")
        self.value = float(value)
        self.name = f"constant:{value:g}"

    def sample(
        self,
        images: Sequence[Image],
        context: PatientContext | None = None,
    ) -> float:
        return self.value


class LognormalServiceTime:
    """Lognormal reading time — the conventional shape, and the default.

    Right-skewed with a long tail: most studies are quick, a few take much longer.
    ``mu`` and ``sigma`` are the parameters of the *underlying normal*, so the median
    read is ``exp(mu)``. Note that this is a modelling convention rather than a
    measured law; §4.2 cites Sunshine 2017 and Sexauer 2022 for real reading times, and
    fitting them is a worthwhile substitution this seam now permits.
    """

    def __init__(
        self,
        mu: float = 2.5,
        sigma: float = 0.4,
        rng: np.random.Generator | None = None,
    ) -> None:
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.name = f"lognormal:{mu:g},{sigma:g}"

    def sample(
        self,
        images: Sequence[Image],
        context: PatientContext | None = None,
    ) -> float:
        return float(self.rng.lognormal(self.mu, self.sigma))


class ScaledServiceTime:
    """Multiply another model's draw by a fixed factor.

    The knob for §9's unresolved question. ``factor < 1`` is the acceleration
    hypothesis (the AI's pre-read saves the reader time); ``factor > 1`` is the
    verification-cost hypothesis (the reader now has a second opinion to adjudicate).
    Both are defensible and the literature is contradictory, so the honest treatment is
    to sweep it and report the range over which a conclusion holds — which is exactly
    what the study layer's factorial grid does with a scalar parameter.
    """

    def __init__(self, base: object, factor: float = 1.0) -> None:
        if factor < 0:
            raise ValueError(f"factor must be >= 0, got {factor}")
        self.base = base
        self.factor = float(factor)
        self.name = f"scaled:{factor:g}x{getattr(base, 'name', type(base).__name__)}"

    def sample(
        self,
        images: Sequence[Image],
        context: PatientContext | None = None,
    ) -> float:
        return self.factor * float(self.base.sample(images, context))  # type: ignore[attr-defined]


class FatigueModulatedServiceTime:
    """Stretch a read in proportion to how worn the reader is.

    ``time = base * (1 + slope * fatigue)``, with ``fatigue`` read from the context's
    signal plane. ``slope=0.5`` means a fully spent reader takes half again as long as
    a fresh one.

    This closes the gap §6.7 left open. It also has a consequence worth anticipating in
    any write-up: a slower fatigued reader lengthens the queue, which raises the load,
    which is itself an experimental condition — so fatigue and congestion feed back on
    each other rather than acting independently. That loop is the point of simulating a
    department instead of scoring isolated decisions, but it does mean the effect of
    ``slope`` is not linear in the outcome.
    """

    def __init__(self, base: object, slope: float = 0.5) -> None:
        if slope < 0:
            raise ValueError(f"slope must be >= 0, got {slope}")
        self.base = base
        self.slope = float(slope)
        self.name = f"fatigued:{slope:g}*{getattr(base, 'name', type(base).__name__)}"

    def sample(
        self,
        images: Sequence[Image],
        context: PatientContext | None = None,
    ) -> float:
        fatigue = context.signal("fatigue", 0.0) if context is not None else 0.0
        base = float(self.base.sample(images, context))  # type: ignore[attr-defined]
        return base * (1.0 + self.slope * fatigue)
