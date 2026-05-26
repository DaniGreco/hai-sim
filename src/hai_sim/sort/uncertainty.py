"""Uncertainty-based sort strategies.

`UncertaintyDescSort` schedules high-uncertainty patients first. The
underlying hypothesis (to be empirically tested) is that high model
uncertainty correlates with case complexity → clinical criticality.

The per-label uncertainty vector is aggregated to a scalar via a configurable
aggregator (mean / max / weighted). Default: mean.
"""

from __future__ import annotations

from typing import Literal

from hai_sim.core.types import ModelOutput, Patient

Aggregator = Literal["mean", "max", "weighted"]


def _aggregate(uncertainty: tuple[float, ...], how: Aggregator, weights: tuple[float, ...] | None) -> float:
    if how == "mean":
        return sum(uncertainty) / len(uncertainty)
    if how == "max":
        return max(uncertainty)
    if how == "weighted":
        if weights is None:
            raise ValueError("aggregator='weighted' requires weights")
        if len(weights) != len(uncertainty):
            raise ValueError(
                f"weights len {len(weights)} != uncertainty len {len(uncertainty)}"
            )
        total = sum(weights)
        if total == 0:
            raise ValueError("weights sum to zero")
        return sum(u * w for u, w in zip(uncertainty, weights, strict=True)) / total
    raise ValueError(f"unknown aggregator: {how}")


class UncertaintyDescSort:
    """High uncertainty → low priority value → served first."""

    def __init__(
        self,
        aggregator: Aggregator = "mean",
        weights: tuple[float, ...] | None = None,
    ) -> None:
        self.aggregator: Aggregator = aggregator
        self.weights = weights
        self.name: str = f"UncertaintyDescSort({aggregator})"

    def priority(self, output: ModelOutput, patient: Patient) -> float:
        agg = _aggregate(output.uncertainty, self.aggregator, self.weights)
        # PriorityStore: lower value → earlier. We want HIGH uncertainty first,
        # so we negate.
        return -agg
