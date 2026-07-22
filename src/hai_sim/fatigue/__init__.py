"""Reader fatigue and how long a read takes.

Two seams that belong together: fatigue is a level, and the two things it acts on are
the reader's accuracy and the reader's speed. See `hai_sim.fatigue.curves` and
`hai_sim.fatigue.service`.
"""

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

__all__ = [
    "CircadianFatigue",
    "CompositeFatigue",
    "ConstantServiceTime",
    "FatigueModulatedServiceTime",
    "LinearFatigue",
    "LognormalServiceTime",
    "NoFatigue",
    "ScaledServiceTime",
    "VigilanceFatigue",
]
