"""Replication statistics and paired contrasts over a study's tidy results.

A study emitted one raw row per (cell x seed) and stopped there, which left the whole
point of the engine's seeding unclaimed. The simulator splits a run's seed into
independent per-module substreams, so two arms run at the same seed see the *same*
patient stream: the difference between them at that seed is the treatment effect with
the sampling noise cancelled, not merely averaged over. That is a **paired** comparison,
and it is far sharper than comparing two independent means — but only if something
actually computes the per-seed difference. This module does.

Confidence intervals are two-sided 95% on Student's t with n-1 degrees of freedom;
`_T95` carries the critical values so the package needs no scipy, and falls back to the
normal 1.96 past 30 degrees of freedom. Anything other than 95% is out of scope rather
than silently approximated.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

# Two-sided 95% critical values of Student's t by degrees of freedom.
_T95: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}
_T95_ASYMPTOTIC = 1.96


def _t_critical(df: int) -> float:
    return _T95.get(df, _T95_ASYMPTOTIC) if df >= 1 else _T95_ASYMPTOTIC


@dataclass(frozen=True)
class Aggregate:
    """Mean of a metric over replications, with its spread and 95% interval.

    ``sd`` and the interval are ``None`` for a single replication: one observation has no
    spread, and reporting 0 would read as "perfectly reproducible" rather than "unknown".
    """

    n: int
    mean: float
    sd: float | None
    ci_low: float | None
    ci_high: float | None

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval sits wholly on one side of zero.

        On a paired contrast this is the readable form of "the arms differ"; it is not a
        substitute for a stated hypothesis and a corrected error rate across a grid.
        """
        if self.ci_low is None or self.ci_high is None:
            return False
        return self.ci_low > 0.0 or self.ci_high < 0.0


def aggregate(values: Sequence[float]) -> Aggregate | None:
    """Mean, sample standard deviation and 95% t-interval; ``None`` for no values."""
    numbers = [float(v) for v in values]
    n = len(numbers)
    if n == 0:
        return None
    mean = sum(numbers) / n
    if n == 1:
        return Aggregate(n=1, mean=mean, sd=None, ci_low=None, ci_high=None)
    variance = sum((v - mean) ** 2 for v in numbers) / (n - 1)
    sd = math.sqrt(variance)
    half = _t_critical(n - 1) * sd / math.sqrt(n)
    return Aggregate(n=n, mean=mean, sd=sd, ci_low=mean - half, ci_high=mean + half)


@dataclass(frozen=True)
class TidyRow:
    """One measurement: which cell, which seed, which subgroup, which metric."""

    factors: dict[str, Any]
    seed: int
    group: str
    metric: str
    value: Any


@dataclass(frozen=True)
class CellSummary:
    """A metric for one cell in one subgroup, aggregated over the replication seeds."""

    factors: dict[str, Any]
    group: str
    metric: str
    stat: Aggregate


@dataclass(frozen=True)
class Contrast:
    """A cell measured against its baseline, paired seed by seed."""

    factors: dict[str, Any]
    baseline: dict[str, Any]
    group: str
    metric: str
    arm_mean: float | None
    baseline_mean: float | None
    delta: Aggregate
    """Aggregate of the *per-seed* differences — not the difference of the means. They
    coincide in the mean and differ where it matters: the interval on the paired
    difference is narrow exactly because common random numbers cancel the shared noise."""

    n_pairs: int = field(default=0)


# Index keys used below. A "series" is one metric of one cell in one subgroup, measured
# once per replication seed; a "measurement" is one of those replications.
_Series = tuple[str, str, str]  # (cell, subgroup, metric)
_Measurement = tuple[str, int, str, str]  # (cell, seed, subgroup, metric)


def _key(factors: dict[str, Any]) -> str:
    """A stable string identifying one cell of the factor grid, usable as a dict key."""
    return json.dumps(factors, sort_keys=True, default=str)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def summarize_cells(rows: Iterable[TidyRow]) -> list[CellSummary]:
    """Aggregate every (cell, subgroup, metric) over its replication seeds."""
    buckets: dict[tuple[str, str, str], list[float]] = {}
    factors_of: dict[str, dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    for row in rows:
        number = _numeric(row.value)
        if number is None:
            continue  # a non-numeric metric has no mean; it stays in the long table
        key = (_key(row.factors), row.group, row.metric)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(number)
        factors_of[key[0]] = row.factors
    summaries: list[CellSummary] = []
    for key in order:
        stat = aggregate(buckets[key])
        if stat is not None:
            summaries.append(
                CellSummary(
                    factors=factors_of[key[0]], group=key[1], metric=key[2], stat=stat
                )
            )
    return summaries


def paired_contrasts(rows: Iterable[TidyRow], baseline_for: Any) -> list[Contrast]:
    """Contrast every non-baseline cell against its reference, pairing on the seed.

    ``baseline_for(factors)`` returns the reference cell's factor assignment, or ``None``
    for a cell that *is* the reference — in practice `hai_sim.study.spec.baseline_of`
    bound to the study.

    Seeds measured on only one of the two arms are skipped. An unpaired difference would
    put back exactly the run-to-run noise that pairing exists to cancel, so half a pair
    is worse than none.
    """
    # One pass builds two indexes: the value of every measurement, and which seeds each
    # (cell, subgroup, metric) was measured at. Everything below is lookups.
    value_at: dict[_Measurement, float] = {}
    factors_of: dict[str, dict[str, Any]] = {}
    seeds_of: dict[_Series, list[int]] = {}
    for row in rows:
        number = _numeric(row.value)
        if number is None:
            continue  # a non-numeric metric has no difference to take
        cell = _key(row.factors)
        measurement = (cell, row.seed, row.group, row.metric)
        if measurement in value_at:
            continue  # duplicate; keep the first so the seed is not counted twice
        factors_of[cell] = row.factors
        value_at[measurement] = number
        seeds_of.setdefault((cell, row.group, row.metric), []).append(row.seed)

    contrasts: list[Contrast] = []
    for (cell, group, metric), seeds in seeds_of.items():
        reference = baseline_for(factors_of[cell])
        if reference is None:
            continue  # this cell IS the baseline: nothing to contrast it with
        reference_cell = _key(reference)
        if reference_cell not in factors_of:
            continue  # the reference arm was never run (a sampled design may drop it)

        arm_values: list[float] = []
        baseline_values: list[float] = []
        for seed in seeds:
            baseline_value = value_at.get((reference_cell, seed, group, metric))
            if baseline_value is None:
                continue  # unpaired: this seed has no counterpart on the baseline arm
            arm_values.append(value_at[(cell, seed, group, metric)])
            baseline_values.append(baseline_value)

        deltas = [a - b for a, b in zip(arm_values, baseline_values, strict=True)]
        delta = aggregate(deltas)
        if delta is None:
            continue  # no seed was measured on both arms
        contrasts.append(
            Contrast(
                factors=factors_of[cell],
                baseline=reference,
                group=group,
                metric=metric,
                arm_mean=sum(arm_values) / len(arm_values),
                baseline_mean=sum(baseline_values) / len(baseline_values),
                delta=delta,
                n_pairs=len(deltas),
            )
        )
    return contrasts
