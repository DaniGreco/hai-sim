"""Shared scalar numerics used by more than one actor/policy.

``logit`` and ``sigmoid`` were byte-identical copies in ``hai_sim.sdt`` and
``hai_sim.doctors.assisted``; this is their single home. ``logit`` clips its input away
from 0/1 at ``eps`` so a saturated probability maps to a large-but-finite log-odds
instead of ``±inf``; ``sigmoid`` is its inverse.

``binary_entropy`` is deliberately *not* here: ``hai_sim.uq.methods`` clips at 1e-12,
``hai_sim.doctors.assisted`` at 1e-6, and ``hai_sim.doctors.simple`` uses a scalar
0-at-the-boundary guard — three intentionally different numerics, not one duplicate.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def logit(p: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Log-odds ``log(p / (1 - p))``, with ``p`` clipped to ``[eps, 1 - eps]``."""
    p = np.clip(p, eps, 1.0 - eps)
    out: np.ndarray = np.log(p / (1.0 - p))
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    """The inverse of `logit`: ``1 / (1 + exp(-x))``."""
    out: np.ndarray = 1.0 / (1.0 + np.exp(-x))
    return out
