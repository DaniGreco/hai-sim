"""Conditioned metrics: any bundle, over any subgroup.

The thesis' methodology asks for recall conditioned on severity, waiting time in the
saturated regime, and per-pathology figures. None of those were expressible while every
bundle folded the whole run into one number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hai_sim.config.scenario import KPIGroupConfig
from hai_sim.core.types import LabelSchema
from hai_sim.factory import build_kpi_groupings
from hai_sim.metrics.collector import PatientRecord
from hai_sim.metrics.kpi import compute_grouped_report, compute_report
from hai_sim.metrics.report import flatten_report
from hai_sim.metrics.slicing import (
    LabelGrouping,
    ValueGrouping,
    project_record,
    resolve_value,
    split_records,
)

_EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "examples" / "custom_metrics.py"
_SCHEMA = LabelSchema(labels=("A", "B"))


def _rec(gt, ai, doc, *, severity=None, observations=None, t=1.0) -> PatientRecord:
    zeros = tuple(0.0 for _ in gt)
    return PatientRecord(
        patient_id=0,
        ground_truth=gt,
        ts_arrival=0.0,
        ts_model_start=0.0,
        ts_model_end=0.0,
        ts_doctor_start=0.0,
        ts_doctor_end=t,
        ai_labels=ai,
        ai_confidence=zeros,
        ai_uncertainty=zeros,
        doc_labels=doc,
        doc_confidence=zeros,
        doc_uncertainty=zeros,
        priority_key=0.0,
        severity=severity,
        observations=observations or {},
    )


# ── Reading a value off a record ───────────────────────────────────────────────


def test_resolve_reads_fields_and_observations():
    record = _rec((1,), (1,), (1,), severity=0.7, observations={"ctx.sex": "F"})
    assert resolve_value(record, "severity") == 0.7
    assert resolve_value(record, "ctx.sex") == "F"
    assert resolve_value(record, "absent") is None


# ── Grouping patients by a value ───────────────────────────────────────────────


def test_categorical_grouping_splits_by_raw_value():
    records = [
        _rec((1,), (1,), (1,), observations={"ctx.sex": "F"}),
        _rec((1,), (1,), (1,), observations={"ctx.sex": "M"}),
        _rec((1,), (1,), (1,), observations={"ctx.sex": "F"}),
    ]
    slices = ValueGrouping(key="ctx.sex").split(records)
    assert [s.key for s in slices] == ["ctx.sex=F", "ctx.sex=M"]
    assert [len(s.records) for s in slices] == [2, 1]


def test_binned_grouping_labels_and_orders_by_magnitude():
    records = [_rec((1,), (1,), (1,), severity=v) for v in (0.0, 2.0, 3.0, 9.0)]
    slices = ValueGrouping(key="severity", edges=(1.0, 5.0)).split(records)
    # ordered by magnitude, not alphabetically — a sorted() on the labels would not be
    assert [s.key for s in slices] == ["severity=<1", "severity=1-5", "severity=>=5"]
    assert [len(s.records) for s in slices] == [1, 2, 1]


def test_bin_edges_are_right_open():
    records = [_rec((1,), (1,), (1,), severity=v) for v in (1.0, 5.0)]
    slices = ValueGrouping(key="severity", edges=(1.0, 5.0)).split(records)
    # the edge value itself falls in the bin that starts at it
    assert {s.key: len(s.records) for s in slices} == {"severity=1-5": 1, "severity=>=5": 1}


def test_records_missing_the_key_are_dropped_unless_asked_for():
    records = [
        _rec((1,), (1,), (1,), observations={"ctx.sex": "F"}),
        _rec((1,), (1,), (1,)),  # no ctx.sex
    ]
    assert sum(len(s.records) for s in ValueGrouping(key="ctx.sex").split(records)) == 1
    kept = ValueGrouping(key="ctx.sex", include_missing=True).split(records)
    assert "ctx.sex=missing" in [s.key for s in kept]


def test_non_numeric_value_under_bin_edges_fails_loudly():
    records = [_rec((1,), (1,), (1,), observations={"ctx.sex": "F"})]
    with pytest.raises(ValueError, match="must be numeric"):
        ValueGrouping(key="ctx.sex", edges=(1.0,)).split(records)


# ── Grouping the label axis ────────────────────────────────────────────────────


def test_project_record_restricts_every_per_label_axis():
    record = _rec(
        (1, 0),
        (1, 1),
        (0, 1),
        severity=(0.2, 0.8),
        observations={
            "ai.mutual_information": (0.1, 0.9),  # per-label → picked
            "ai.n_members": 5,  # scalar → kept whole
            "ctx.sex": "F",
        },
    )
    projected = project_record(record, 1)
    assert projected.ground_truth == (0,)
    assert projected.ai_labels == (1,)
    assert projected.doc_labels == (1,)
    assert projected.severity == 0.8
    assert projected.observations["ai.mutual_information"] == 0.9
    assert projected.observations["ai.n_members"] == 5
    assert projected.observations["ctx.sex"] == "F"


def test_label_grouping_makes_any_bundle_per_pathology():
    # label A: AI always right. label B: AI always wrong.
    records = [_rec((1, 1), (1, 0), (1, 0)) for _ in range(4)]
    grouped = compute_grouped_report(records, ["actors"], [LabelGrouping(_SCHEMA.labels)])
    assert grouped["label=A"]["actors"]["ai"].recall_macro == 1.0
    assert grouped["label=B"]["actors"]["ai"].recall_macro == 0.0


def test_project_record_rejects_an_out_of_range_label():
    with pytest.raises(IndexError):
        project_record(_rec((1, 0), (1, 0), (1, 0)), 5)


# ── Crossing, and the report shape ─────────────────────────────────────────────


def test_groupings_cross_and_name_every_level():
    records = [
        _rec((1, 1), (1, 1), (1, 1), observations={"ctx.sex": s}) for s in ("F", "M")
    ]
    slices = split_records(
        records, [ValueGrouping(key="ctx.sex"), LabelGrouping(_SCHEMA.labels)]
    )
    assert [s.key for s in slices] == [
        "ctx.sex=F|label=A",
        "ctx.sex=F|label=B",
        "ctx.sex=M|label=A",
        "ctx.sex=M|label=B",
    ]


def test_no_groupings_yields_the_whole_run_as_one_slice():
    records = [_rec((1,), (1,), (1,))]
    grouped = compute_grouped_report(records, ["system"])
    assert list(grouped) == [""]
    assert grouped[""]["system"]["n_patients"] == 1


def test_a_custom_bundle_is_conditionable_too():
    """The promise is 'any metric x any condition', so a bundle from outside the
    library must slice exactly like a built-in one."""
    records = [
        _rec((1,), (0,), (0,), observations={"ctx.sex": "F"}),  # one FN
        _rec((0,), (1,), (1,), observations={"ctx.sex": "M"}),  # one FP
    ]
    grouped = compute_grouped_report(
        records, [f"{_EXAMPLE}:CostBundle"], [ValueGrouping(key="ctx.sex")]
    )
    assert grouped["ctx.sex=F"]["cost"]["ai_total"] == 10.0  # the false negative
    assert grouped["ctx.sex=M"]["cost"]["ai_total"] == 1.0  # the false positive


# ── Flattening ─────────────────────────────────────────────────────────────────


def test_flatten_report_addresses_scalars_by_dotted_path():
    records = [_rec((1, 0), (1, 0), (1, 0))]
    flat = flatten_report(compute_grouped_report(records, ["system", "joint_table"])[""])
    assert flat["system.n_patients"] == 1
    assert flat["joint_table.joint.gt1_ai1_doc1"] == 1
    assert "actors" not in flat


def test_flatten_report_skips_vectors_rather_than_averaging_them():
    """A per-label vector has no single cell; silently meaning its mean would answer a
    different question than the one asked."""
    flat = flatten_report({"b": {"vec": (1.0, 2.0), "scalar": 3.0}})
    assert flat == {"b.scalar": 3.0}


# ── Config → grouping ──────────────────────────────────────────────────────────


def test_config_builds_both_grouping_kinds():
    groupings = build_kpi_groupings(
        (
            KPIGroupConfig(key="label"),
            KPIGroupConfig(key="load.in_system_at_arrival", edges=(1.0, 5.0)),
        ),
        _SCHEMA,
    )
    assert isinstance(groupings[0], LabelGrouping)
    assert isinstance(groupings[1], ValueGrouping)
    assert groupings[1].edges == (1.0, 5.0)


def test_label_grouping_rejects_edges():
    with pytest.raises(ValueError, match="no edges"):
        build_kpi_groupings((KPIGroupConfig(key="label", edges=(1.0,)),), _SCHEMA)


# ── End to end: the question the methodology actually asks ─────────────────────


def test_waiting_time_can_finally_be_conditioned_on_congestion():
    """Order effects only exist under saturation. An unconditioned mean wait hides
    that; splitting on the recorded queue state exposes it."""
    from hai_sim.doctors.mock import RandomMockDoctor
    from hai_sim.models.mock import RandomMockAI
    from hai_sim.sim.engine import SimulationEngine
    from hai_sim.sim.patient_generator import PoissonPatientGenerator
    from hai_sim.sort.random import RandomSort

    schema = LabelSchema.default_v0()
    res = SimulationEngine(
        patient_generator=PoissonPatientGenerator(
            schema=schema, arrival_rate=2.0, n_patients=80, prevalence=0.3,
            rng=np.random.default_rng(0),
        ),
        ai=RandomMockAI(schema=schema, rng=np.random.default_rng(1)),
        doctor=RandomMockDoctor(
            schema=schema, rng=np.random.default_rng(2),
            service_time_mu=0.0, service_time_sigma=0.01,
        ),
        sort_strategy=RandomSort(rng=np.random.default_rng(3)),
    ).run()

    grouped = compute_grouped_report(
        res.collector.records,
        ["system"],
        [ValueGrouping(key="load.in_system_at_arrival", edges=(5.0, 20.0))],
    )
    waits = [report["system"]["mean_wait_to_doctor"] for report in grouped.values()]
    assert len(waits) > 1, "the run must span more than one congestion regime"
    # Patients arriving into an emptier department wait least. (The *ordering* of the
    # busier bands is not monotone under a random service order — a late arrival can be
    # drawn immediately — which is itself why the conditioned view is the informative
    # one.)
    assert waits[0] == min(waits)
    overall = compute_report(res.collector.records, ["system"])["system"]
    # The single unconditioned number sits inside a spread it conceals: that gap is
    # exactly what an order study needs to see and previously could not.
    assert min(waits) < overall["mean_wait_to_doctor"] < max(waits)


# ── Patients with / without each finding ───────────────────────────────────────


def test_truth_grouping_splits_patients_by_their_ground_truth_per_pathology():
    from hai_sim.metrics.slicing import TruthGrouping

    records = [
        _rec((1, 0), (0, 0), (0, 0), t=10.0),
        _rec((0, 0), (0, 0), (0, 0), t=2.0),
        _rec((-1, 1), (0, 0), (0, 0), t=4.0),
    ]
    slices = {s.key: s.records for s in TruthGrouping(label_names=_SCHEMA.labels).split(records)}
    assert set(slices) == {"truth=A:positive", "truth=A:negative", "truth=B:positive", "truth=B:negative"}
    assert [r.ts_doctor_end for r in slices["truth=A:positive"]] == [10.0]
    assert [r.ts_doctor_end for r in slices["truth=A:negative"]] == [2.0], "-1 is in neither"
    assert len(slices["truth=B:negative"]) == 2
    # records are left whole: a time metric needs the patient, not a projected label
    assert slices["truth=A:positive"][0].ground_truth == (1, 0)


def test_truth_is_a_buildable_kpi_group():
    from hai_sim.metrics.slicing import TruthGrouping

    (grouping,) = build_kpi_groupings([KPIGroupConfig(key="truth")], _SCHEMA)
    assert isinstance(grouping, TruthGrouping)
    with pytest.raises(ValueError, match="truth"):
        build_kpi_groupings([KPIGroupConfig(key="truth", edges=(0.5,))], _SCHEMA)
