"""Scaffolding shared by the simulated doctor readers (`FixedAccuracyDoctor`, `SDTDoctor`).

Both derive their call from the injected ground truth and share two mechanisms that are
not about *how* they read: how long a read takes, and how the signal plane's difficulty
and fatigue degrade a read. Kept here so each reader states them once rather than twice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from hai_sim.core.types import Image, PatientContext


def signal_penalty(
    context: PatientContext | None,
    difficulty_slope: float,
    fatigue_slope: float,
) -> float:
    """Accuracy/discriminability lost to this read's difficulty and fatigue signals.

    ``slope · signal`` summed over the two signal-plane channels, floored at 0. Returns
    0 when no signals are configured (both slopes 0) or there is no context — the reader
    then falls back to its base capability, and a caller can treat 0 as "no penalty".
    """
    if context is None or (difficulty_slope == 0.0 and fatigue_slope == 0.0):
        return 0.0
    penalty = difficulty_slope * context.signal(
        "difficulty", 0.0
    ) + fatigue_slope * context.signal("fatigue", 0.0)
    return penalty if penalty > 0.0 else 0.0


def lognormal_service_time(
    rng: np.random.Generator,
    mu: float,
    sigma: float,
    *,
    model: object | None = None,
    images: Sequence[Image] | None = None,
    context: PatientContext | None = None,
) -> float:
    """One read's duration: the configured `ServiceTimeModel` if any, else lognormal(mu, sigma)."""
    if model is not None:
        return float(model.sample(images, context))  # type: ignore[attr-defined]
    return float(rng.lognormal(mu, sigma))
