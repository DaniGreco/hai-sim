"""KPI computation from a list of `PatientRecord`s.

v0 KPI set:
- TTFE — Time To First Error (per-actor: AI, doctor, hybrid)
- MTTF — Mean Time To Failure = mean inter-arrival between consecutive errors
- mean-time-to-FN — mean time between consecutive false-negative events
- F1 macro per actor (AI, doctor, hybrid)
- Throughput, mean queue waits

The 'hybrid' actor uses the doctor's label when available (in v0 always),
which is the upper bound of what an AI-triaged human pipeline can deliver
while a real fusion policy is not implemented.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from hai_sim.metrics.collector import PatientRecord

Actor = Literal["ai", "doctor", "hybrid"]


@dataclass(frozen=True)
class ActorKPIs:
    f1_macro: float
    precision_macro: float
    recall_macro: float
    accuracy: float
    n_fn: int
    n_fp: int
    ttfe: float | None  # time of first error (any kind); None if no errors
    mttf: float | None  # mean inter-error gap; None if <2 errors
    mttfn: float | None  # mean inter-FN gap; None if <2 FN events


@dataclass(frozen=True)
class RunKPIs:
    n_patients: int
    sim_duration: float
    throughput: float  # patients / time unit
    mean_wait_to_model: float
    mean_wait_to_doctor: float
    mean_time_in_system: float
    ai: ActorKPIs
    doctor: ActorKPIs
    hybrid: ActorKPIs


def _labels_for(record: PatientRecord, actor: Actor) -> tuple[int, ...]:
    if actor == "ai":
        return record.ai_labels
    if actor == "doctor":
        return record.doc_labels
    if actor == "hybrid":
        # v0 placeholder: hybrid == doctor's call. Later: fusion of AI + doctor.
        return record.doc_labels
    raise ValueError(actor)


def _per_label_counts(records: Sequence[PatientRecord], actor: Actor) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return per-label (TP, FP, FN, TN) lists."""
    if not records:
        return [], [], [], []
    n_labels = len(records[0].ground_truth)
    tp = [0] * n_labels
    fp = [0] * n_labels
    fn = [0] * n_labels
    tn = [0] * n_labels
    for r in records:
        pred = _labels_for(r, actor)
        for i, (p, g) in enumerate(zip(pred, r.ground_truth, strict=True)):
            if p == 1 and g == 1:
                tp[i] += 1
            elif p == 1 and g == 0:
                fp[i] += 1
            elif p == 0 and g == 1:
                fn[i] += 1
            else:
                tn[i] += 1
    return tp, fp, fn, tn


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def _actor_kpis(records: Sequence[PatientRecord], actor: Actor) -> ActorKPIs:
    tp, fp, fn, tn = _per_label_counts(records, actor)
    if not tp:
        return ActorKPIs(0.0, 0.0, 0.0, 0.0, 0, 0, None, None, None)
    n_labels = len(tp)
    precisions = [_safe_div(tp[i], tp[i] + fp[i]) for i in range(n_labels)]
    recalls = [_safe_div(tp[i], tp[i] + fn[i]) for i in range(n_labels)]
    f1s = [
        _safe_div(2 * precisions[i] * recalls[i], precisions[i] + recalls[i])
        for i in range(n_labels)
    ]
    total = sum(tp) + sum(fp) + sum(fn) + sum(tn)
    accuracy = _safe_div(sum(tp) + sum(tn), total)

    # Error timeline (use doctor end time as the "event time" for each patient)
    error_times: list[float] = []
    fn_times: list[float] = []
    for r in records:
        pred = _labels_for(r, actor)
        is_error = any(p != g for p, g in zip(pred, r.ground_truth, strict=True))
        is_fn = any(p == 0 and g == 1 for p, g in zip(pred, r.ground_truth, strict=True))
        if is_error:
            error_times.append(r.ts_doctor_end)
        if is_fn:
            fn_times.append(r.ts_doctor_end)
    error_times.sort()
    fn_times.sort()

    ttfe = error_times[0] if error_times else None
    mttf: float | None = None
    if len(error_times) >= 2:
        # zip-with-next: lengths intentionally differ by 1
        gaps = [b - a for a, b in zip(error_times, error_times[1:])]
        mttf = sum(gaps) / len(gaps)
    mttfn: float | None = None
    if len(fn_times) >= 2:
        gaps = [b - a for a, b in zip(fn_times, fn_times[1:])]
        mttfn = sum(gaps) / len(gaps)

    return ActorKPIs(
        f1_macro=sum(f1s) / n_labels,
        precision_macro=sum(precisions) / n_labels,
        recall_macro=sum(recalls) / n_labels,
        accuracy=accuracy,
        n_fn=sum(fn),
        n_fp=sum(fp),
        ttfe=ttfe,
        mttf=mttf,
        mttfn=mttfn,
    )


def compute_kpis(records: Sequence[PatientRecord]) -> RunKPIs:
    if not records:
        return RunKPIs(
            n_patients=0,
            sim_duration=0.0,
            throughput=0.0,
            mean_wait_to_model=0.0,
            mean_wait_to_doctor=0.0,
            mean_time_in_system=0.0,
            ai=ActorKPIs(0.0, 0.0, 0.0, 0.0, 0, 0, None, None, None),
            doctor=ActorKPIs(0.0, 0.0, 0.0, 0.0, 0, 0, None, None, None),
            hybrid=ActorKPIs(0.0, 0.0, 0.0, 0.0, 0, 0, None, None, None),
        )

    n = len(records)
    last_t = max(r.ts_doctor_end for r in records)
    waits_model = [r.ts_model_start - r.ts_arrival for r in records]
    waits_doctor = [r.ts_doctor_start - r.ts_model_end for r in records]
    sys_times = [r.ts_doctor_end - r.ts_arrival for r in records]
    return RunKPIs(
        n_patients=n,
        sim_duration=last_t,
        throughput=_safe_div(n, last_t),
        mean_wait_to_model=sum(waits_model) / n,
        mean_wait_to_doctor=sum(waits_doctor) / n,
        mean_time_in_system=sum(sys_times) / n,
        ai=_actor_kpis(records, "ai"),
        doctor=_actor_kpis(records, "doctor"),
        hybrid=_actor_kpis(records, "hybrid"),
    )
