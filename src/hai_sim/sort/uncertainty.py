"""Triage by uncertainty: let the AI's doubt decide who the doctor sees first.

This is the point of putting the AI upstream of the queue. Having read every patient
already, the model can say which cases it found hardest, and those go to the front.

**The hypothesis being tested, not assumed:** that where the model is unsure is also
where it is wrong, and where a human is most needed. It is plausible, not established —
measuring whether it holds is one of the questions this simulator exists to answer.

Uncertainty arrives as one number *per pathology*, so a triage rule must decide how to
collapse the vector into the single number a queue can order by. That choice is a
modelling decision, not a detail:

- ``mean``     — overall doubt across all findings; a broadly ambiguous image wins
- ``max``      — the single most doubtful finding; one alarming uncertainty wins
- ``weighted`` — a clinically weighted average, for when some findings matter more

A reordering can only change outcomes when patients are actually queued. Under an idle
department every strategy serves the same patients at the same time, and even under load
the *mean* wait is unchanged by ordering — only *whose* wait it is. Order effects
therefore have to be looked for in load-conditioned or severity-conditioned metrics.
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
    """Most-uncertain-first. Descending uncertainty, hence the name."""

    def __init__(
        self,
        aggregator: Aggregator = "mean",
        weights: tuple[float, ...] | None = None,
    ) -> None:
        self.aggregator: Aggregator = aggregator
        self.weights = weights
        self.name: str = f"UncertaintyDescSort({aggregator})"

    def priority(self, output: ModelOutput, patient: Patient) -> float:
        score = _aggregate(output.uncertainty, self.aggregator, self.weights)
        # The queue serves the LOWEST priority value first, and we want the HIGHEST
        # uncertainty first — so negate. Getting this sign wrong silently inverts the
        # experiment into "easiest cases first", which still runs and still looks fine.
        return -score
