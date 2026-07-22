"""When patients arrive — separated from which patient arrives.

See `hai_sim.arrivals.processes` for the implementations and
`hai_sim.core.protocols.ArrivalProcess` for the contract.
"""

from hai_sim.arrivals.processes import (
    DeterministicArrival,
    NHPPArrival,
    PiecewiseConstantRate,
    PoissonArrival,
)

__all__ = [
    "DeterministicArrival",
    "NHPPArrival",
    "PiecewiseConstantRate",
    "PoissonArrival",
]
