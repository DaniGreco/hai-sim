"""Conditioned metrics: compute any bundle over subgroups of the run telemetry.

Every bundle folded the whole run into one number, so questions a workflow study
has to ask could not be asked of it: recall *conditioned on severity*, waiting
time *in the saturated regime* (order effects only exist under saturation, so an
unconditioned temporal mean hides exactly the effect a triage study is looking for), or
metrics *per pathology* rather than macro-averaged.

The mechanism is deliberately dumb, and that is what makes it general: **a grouping
splits the records, and the unchanged bundle runs on each piece**. Because a bundle is
a pure function over a record sequence, this makes *any* bundle — including one the
researcher wrote outside the library — conditionable without touching it.

Two groupings cover the ground:

- `ValueGrouping` slices *patients* by a record field or observation (``severity``,
  ``ctx.sex``, ``load.in_system_at_arrival``), with optional binning for continuous
  values.
- `LabelGrouping` slices the *label axis*: same patients, each record projected down to
  one pathology. A macro-averaged bundle thus becomes per-pathology for free.

Groupings compose: passing several crosses them, and the slice key names every level.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from hai_sim.metrics.collector import PatientRecord, as_sequence

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class RecordSlice:
    """One subgroup: a human-readable key and the records that fall in it."""

    key: str
    records: tuple[PatientRecord, ...]


@runtime_checkable
class Grouping(Protocol):
    """Splits the telemetry into named subgroups. Implementations are pure.

    ``name`` is declared read-only (unlike the framework's other protocols) because a
    grouping's name is derived from what it groups by, not configured independently —
    so implementations are free to expose it as a property.
    """

    @property
    def name(self) -> str: ...

    def split(self, records: Sequence[PatientRecord]) -> list[RecordSlice]: ...


# ── Resolving a value out of a record ───────────────────────────────────────────


_RECORD_FIELDS = frozenset(
    {
        "patient_id",
        "priority_key",
        "severity",
        "ts_arrival",
        "ts_model_start",
        "ts_model_end",
        "ts_doctor_start",
        "ts_doctor_end",
    }
)


def resolve_value(record: PatientRecord, key: str) -> Any:
    """Read ``key`` off a record: a plain field, or an observation like ``ctx.sex``.

    Returns ``None`` when the record does not carry it — records in one run may
    legitimately differ in which observations they hold.
    """
    if key in _RECORD_FIELDS:
        return getattr(record, key)
    return record.observations.get(key)


def _bin_label(value: float, edges: Sequence[float]) -> tuple[int, str]:
    """Place ``value`` in the half-open bins cut by ``edges``; return (order, label)."""
    index = bisect.bisect_right(list(edges), value)
    if index == 0:
        return 0, f"<{edges[0]:g}"
    if index == len(edges):
        return index, f">={edges[-1]:g}"
    return index, f"{edges[index - 1]:g}-{edges[index]:g}"


@dataclass(frozen=True)
class ValueGrouping:
    """Group patients by a record field or observation, optionally binned.

    ``edges`` turns a continuous quantity into ordered bins: ``edges=(1, 5)`` yields
    ``<1``, ``1-5``, ``>=5``. Without edges the raw value is the group, which is what
    you want for categoricals (sex, the UQ method's name).

    Records missing the key are dropped unless ``include_missing``, because a metric
    computed over "everyone we happen to know about" silently changes denominator.
    """

    key: str
    edges: tuple[float, ...] = ()
    include_missing: bool = False

    @property
    def name(self) -> str:
        return self.key

    def split(self, records: Sequence[PatientRecord]) -> list[RecordSlice]:
        buckets: dict[str, list[PatientRecord]] = {}
        order: dict[str, tuple[int, str]] = {}
        for record in records:
            value = resolve_value(record, self.key)
            if value is None:
                if not self.include_missing:
                    continue
                rank, shown = (1 << 30, "missing")
            elif self.edges:
                try:
                    rank, shown = _bin_label(float(value), self.edges)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"grouping {self.key!r} has bin edges, so its values must be "
                        f"numeric; got {value!r}"
                    ) from exc
            else:
                rank, shown = (0, str(value))
            group_key = f"{self.key}={shown}"
            buckets.setdefault(group_key, []).append(record)
            order[group_key] = (rank, shown)
        return [
            RecordSlice(key, tuple(buckets[key]))
            for key in sorted(buckets, key=lambda k: order[k])
        ]


@dataclass(frozen=True)
class LabelGrouping:
    """One slice per pathology: the same patients, each record projected to one label.

    Bundles average across labels, so a per-pathology view normally means rewriting
    every bundle. Projecting the *records* instead gets it for free, and gets it for
    custom bundles too.
    """

    label_names: tuple[str, ...]

    @property
    def name(self) -> str:
        return "label"

    def split(self, records: Sequence[PatientRecord]) -> list[RecordSlice]:
        return [
            RecordSlice(
                f"label={label}",
                tuple(project_record(record, index) for record in records),
            )
            for index, label in enumerate(self.label_names)
        ]


@dataclass(frozen=True)
class TruthGrouping:
    """For every pathology, two slices of *patients*: those who have it and those who do not.

    `LabelGrouping` keeps every patient and narrows the label axis, which is right for a
    diagnostic metric. A time metric asks something else — *do the patients with this
    finding wait less than the ones without?* — and that needs the patients themselves
    split by their ground truth. Records are left whole; a patient the dataset did not
    annotate for a pathology (``-1``) falls in neither slice; a slice that would be empty
    is omitted.
    """

    label_names: tuple[str, ...]

    @property
    def name(self) -> str:
        return "truth"

    def split(self, records: Sequence[PatientRecord]) -> list[RecordSlice]:
        slices: list[RecordSlice] = []
        for index, label in enumerate(self.label_names):
            for value, shown in ((1, "positive"), (0, "negative")):
                picked = tuple(r for r in records if int(r.ground_truth[index]) == value)
                if picked:
                    slices.append(RecordSlice(f"truth={label}:{shown}", picked))
        return slices


def project_record(record: PatientRecord, index: int) -> PatientRecord:
    """A copy of ``record`` restricted to label ``index`` on every per-label axis."""
    n_labels = len(record.ground_truth)
    if not 0 <= index < n_labels:
        raise IndexError(f"label index {index} out of range for {n_labels} labels")

    def pick_int(values: Sequence[int]) -> tuple[int, ...]:
        return (int(values[index]),)

    def pick_float(values: Sequence[float]) -> tuple[float, ...]:
        return (float(values[index]),)

    solo = record.doctor_solo_labels
    severity = record.severity
    if isinstance(severity, tuple):
        severity = severity[index]

    observations: dict[str, Any] = {}
    for key, value in record.observations.items():
        sequence = as_sequence(value)
        observations[key] = (
            sequence[index] if sequence is not None and len(sequence) == n_labels else value
        )

    return replace(
        record,
        ground_truth=pick_int(record.ground_truth),
        ai_labels=pick_int(record.ai_labels),
        ai_confidence=pick_float(record.ai_confidence),
        ai_uncertainty=pick_float(record.ai_uncertainty),
        doc_labels=pick_int(record.doc_labels),
        doc_confidence=pick_float(record.doc_confidence),
        doc_uncertainty=pick_float(record.doc_uncertainty),
        doctor_solo_labels=None if solo is None else pick_int(solo),
        severity=severity,
        observations=observations,
    )


# ── Crossing groupings ──────────────────────────────────────────────────────────


def split_records(
    records: Sequence[PatientRecord], groupings: Iterable[Grouping]
) -> list[RecordSlice]:
    """Split by each grouping in turn, crossing them; keys join with ``|``.

    With no groupings the whole run comes back as a single unnamed slice, so callers
    need no special case for "not conditioned".
    """
    slices = [RecordSlice("", tuple(records))]
    for grouping in groupings:
        crossed: list[RecordSlice] = []
        for current in slices:
            for sub in grouping.split(current.records):
                key = f"{current.key}|{sub.key}" if current.key else sub.key
                crossed.append(RecordSlice(key, sub.records))
        slices = crossed
    return slices
