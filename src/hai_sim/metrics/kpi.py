"""KPI computation from a list of `PatientRecord`s.

Two KPI scopes:
- per-actor (`ActorKPIs`): classification quality of one decision maker (ai / doctor / hybrid)
  evaluated against the ground truth label vector.
- per-run (`RunKPIs`): system-level metrics (throughput, queue waits) that do not depend on
  which actor is being measured, plus three nested `ActorKPIs` (ai, doctor, hybrid).

Conventions in this module:
- "error" on a patient = the actor's predicted label vector differs from ground truth on
  at least one label (multi-label hamming-style error, NOT exact-match).
- "FN event" on a patient = the actor predicted 0 on at least one label that is actually 1
  (i.e. missed at least one pathology). FN events are the clinically dangerous ones — they
  drive `n_fn` and `mttfn`.
- All temporal KPIs use `ts_doctor_end` as the patient's "event time", because that is the
  moment at which the system's verdict on that patient becomes known.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from hai_sim.metrics.collector import PatientRecord

Actor = Literal["ai", "doctor", "hybrid"]


@dataclass(frozen=True)
class ActorKPIs:
    """Classification + reliability metrics for one actor on one run."""

    # ── Classification quality (averaged across labels: "macro" averaging) ────────────
    f1_macro: float
    """Harmonic mean of precision and recall, computed per label and then averaged.
    Range [0, 1]. Macro avg gives equal weight to every label regardless of prevalence —
    important here because some pathologies are rare and we don't want them ignored."""

    precision_macro: float
    """Mean per-label precision = mean over labels of TP_i / (TP_i + FP_i).
    "When the actor says positive, how often is it right?" Low precision → many FP."""

    recall_macro: float
    """Mean per-label recall = mean over labels of TP_i / (TP_i + FN_i). Aka sensitivity.
    "Of the truly positive cases, how many does the actor catch?" Low recall → many FN
    (the clinically expensive failure mode)."""

    accuracy: float
    """Micro accuracy across ALL (patient x label) entries = (sum TP + sum TN) / total entries.
    Easy to misread when prevalence is low: predicting "all-zero" can score high accuracy
    while having recall 0. Reported for completeness; do NOT use it as the primary KPI."""

    # ── Aggregated confusion counts (sum over labels, sum over patients) ──────────────
    n_fn: int
    """Total false negatives = number of (patient, label) pairs where ground truth = 1
    but the actor predicted 0. Clinically: missed pathologies — usually the worst kind
    of mistake in radiology, motivating the cost matrix mentioned in CONTEXT.md."""

    n_fp: int
    """Total false positives = number of (patient, label) pairs where ground truth = 0
    but the actor predicted 1. Clinically: unnecessary alarm / overdiagnosis."""

    # ── Reliability metrics over the patient timeline ─────────────────────────────────
    ttfe: float | None
    """Time To First Error. The simulation timestamp (`ts_doctor_end`) of the FIRST
    patient on which the actor makes ANY mistake. `None` if the actor never errs in
    this run. Operational meaning: "how long does the deployed system stay perfect
    before its first slip-up?" Lower = the system is fragile early; higher = robust."""

    mttf: float | None
    """Mean Time To Failure = mean of gaps between CONSECUTIVE errors on the timeline.
    If the actor errs at t1 < t2 < ... < tk, MTTF = mean(t2-t1, t3-t2, ..., tk-t_{k-1}).
    Reliability-engineering metric: large MTTF → errors are rare/spread out;
    small MTTF → errors come in bursts. `None` if fewer than 2 errors (no gap to measure)."""

    mttfn: float | None
    """Mean Time To find a False Negative = same as MTTF but counting only the patients
    on which the actor produces at least one FN. Singled out because FN events are the
    clinically expensive ones — knowing the rate at which they accumulate is what the
    quality team actually cares about. `None` if fewer than 2 FN events."""


@dataclass(frozen=True)
class RunKPIs:
    """System-level KPIs for one simulation run + nested per-actor KPIs."""

    # ── Sizing / throughput ───────────────────────────────────────────────────────────
    n_patients: int
    """Number of patients fully processed (arrived → AI → doctor → recorded).
    Equals `n_patients` in the scenario config unless the engine raised mid-run."""

    sim_duration: float
    """Total simulated time elapsed, in the scenario's time units. Defined as
    max(ts_doctor_end) across all patients = when the LAST patient leaves the system.
    NOT wall-clock time of the Python process; that is `wall_clock_s` in the JSON summary."""

    throughput: float
    """Effective system throughput = n_patients / sim_duration. Units: patients per
    simulated time unit. Compare against the arrival rate λ to see if the system is
    saturated (throughput ≈ λ → not saturated; throughput < λ → backlog growing)."""

    # ── Queue-time decomposition (averaged across patients) ───────────────────────────
    mean_wait_to_model: float
    """Mean of (ts_model_start - ts_arrival) over all patients. How long a patient sits
    in the input queue before the AI starts processing. With v0's single AI worker and
    fast mock AI this is usually ~0 unless arrivals burst faster than AI service time."""

    mean_wait_to_doctor: float
    """Mean of (ts_doctor_start - ts_model_end) over all patients. How long a patient
    sits in the triaged_store (after AI inference) before the doctor picks them up.
    THIS is the queue that the SortStrategy reorders. The MEAN does not depend on the
    sort policy (sum is invariant under permutation), but the per-patient distribution
    of wait times — and which patients wait — does change."""

    mean_time_in_system: float
    """Mean of (ts_doctor_end - ts_arrival) over all patients. End-to-end latency:
    from when the patient enters the simulation to when the verdict is recorded.
    Equals approximately mean_wait_to_model + ai_service_time + mean_wait_to_doctor
    + doctor_service_time."""

    # ── Per-actor classification KPIs ─────────────────────────────────────────────────
    ai: ActorKPIs
    """Metrics computed treating the AI's prediction as the final verdict.
    Baseline for "what would the deployed model do without a human in the loop"."""

    doctor: ActorKPIs
    """Metrics computed treating the doctor's prediction as the final verdict.
    Baseline for "what would the doctor do without the AI" (only meaningful when
    `doctor.use_ai_hint = false`; otherwise the doctor saw the AI suggestion)."""

    hybrid: ActorKPIs
    """Metrics for the joint Human-AI verdict. v0: identical to doctor (placeholder).
    Later: implement a fusion policy (e.g. defer-on-low-confidence, weighted vote)
    and this will diverge from `doctor`. This is the KPI block that ultimately tells
    us whether the AI-in-the-loop is worth deploying."""


def _labels_for(record: PatientRecord, actor: Actor) -> tuple[int, ...]:
    """Pick which predicted label vector to score for the given actor."""
    if actor == "ai":
        return record.ai_labels
    if actor == "doctor":
        return record.doc_labels
    if actor == "hybrid":
        # v0 placeholder: hybrid == doctor's call. Later: fusion of AI + doctor.
        return record.doc_labels
    raise ValueError(actor)


def _per_label_counts(
    records: Sequence[PatientRecord], actor: Actor
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return per-label (TP, FP, FN, TN) lists across all patients.

    For each label index `i`, counts how often the actor's bit i matched / mismatched
    the ground truth bit i. Output lists have length `n_labels` (= LabelSchema.n).
    """
    if not records:
        return [], [], [], []
    n_labels = len(records[0].ground_truth)
    tp = [0] * n_labels  # predicted 1, truth 1 — correctly flagged
    fp = [0] * n_labels  # predicted 1, truth 0 — false alarm
    fn = [0] * n_labels  # predicted 0, truth 1 — missed (clinically dangerous)
    tn = [0] * n_labels  # predicted 0, truth 0 — correctly cleared
    for r in records:
        pred = _labels_for(r, actor)
        for i, (p, g) in enumerate(zip(pred, r.ground_truth, strict=True)):
            if g not in (0, 1):
                # g == -1 → label not annotated in this dataset for this patient.
                # Real CXR datasets (NIH, CheXpert) only label a subset of the
                # model's pathologies, so we can't score these entries: skip them.
                continue
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
    """Division that returns 0 instead of NaN/ZeroDivisionError when the denominator
    is zero. Used so a label never seen in either ground truth or predictions doesn't
    poison the macro averages."""
    return a / b if b > 0 else 0.0


def _actor_kpis(records: Sequence[PatientRecord], actor: Actor) -> ActorKPIs:
    """Compute the full `ActorKPIs` block for one actor on one run."""
    tp, fp, fn, tn = _per_label_counts(records, actor)
    if not tp:
        # No records at all → return a neutral KPI object so downstream code can
        # always assume the dataclass shape.
        return ActorKPIs(0.0, 0.0, 0.0, 0.0, 0, 0, None, None, None)
    n_labels = len(tp)

    # Per-label precision / recall / F1, then macro-average.
    # precision_i = TP_i / (TP_i + FP_i): of the labels we flagged as positive, how many were real?
    precisions = [_safe_div(tp[i], tp[i] + fp[i]) for i in range(n_labels)]
    # recall_i    = TP_i / (TP_i + FN_i): of the truly positive labels, how many did we catch?
    recalls = [_safe_div(tp[i], tp[i] + fn[i]) for i in range(n_labels)]
    # F1_i        = 2·P·R / (P+R): harmonic mean — penalises imbalances between P and R.
    f1s = [
        _safe_div(2 * precisions[i] * recalls[i], precisions[i] + recalls[i])
        for i in range(n_labels)
    ]

    # Micro accuracy across all (patient x label) entries. Note: with rare pathologies
    # the all-zero predictor can get high accuracy → also look at F1 / recall.
    total = sum(tp) + sum(fp) + sum(fn) + sum(tn)
    accuracy = _safe_div(sum(tp) + sum(tn), total)

    # Build the error timeline. `ts_doctor_end` is the moment the verdict is known.
    # A patient contributes to `error_times` if the actor mispredicted AT LEAST ONE label
    # for that patient (hamming-style error: any mismatch counts). FN events are scored
    # separately because they're the clinically expensive ones.
    error_times: list[float] = []
    fn_times: list[float] = []
    for r in records:
        pred = _labels_for(r, actor)
        # Only labeled entries (g in {0,1}) count; g == -1 is "not annotated".
        is_error = any(
            p != g for p, g in zip(pred, r.ground_truth, strict=True) if g in (0, 1)
        )
        is_fn = any(p == 0 and g == 1 for p, g in zip(pred, r.ground_truth, strict=True))
        if is_error:
            error_times.append(r.ts_doctor_end)
        if is_fn:
            fn_times.append(r.ts_doctor_end)
    # Sort by simulated time so consecutive gaps in the next step are well-defined.
    # In practice records are already roughly time-ordered but we don't assume it.
    error_times.sort()
    fn_times.sort()

    # TTFE = Time To First Error. The simulated timestamp of the earliest mistake.
    # `None` when the actor was flawless on this run.
    ttfe = error_times[0] if error_times else None

    # MTTF = Mean Time To Failure. Mean gap between consecutive error events.
    # Reliability-engineering staple: large MTTF -> errors well spread; small -> bursty.
    # Needs >= 2 events to have at least one gap.
    mttf: float | None = None
    if len(error_times) >= 2:
        gaps = [b - a for a, b in itertools.pairwise(error_times)]
        mttf = sum(gaps) / len(gaps)

    # mean-time-to-FN. Same shape as MTTF but restricted to false-negative events
    # (missed pathologies). Operationally the most important reliability KPI for
    # radiology, since missing disease is much worse than overcalling it.
    mttfn: float | None = None
    if len(fn_times) >= 2:
        gaps = [b - a for a, b in itertools.pairwise(fn_times)]
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
    """Reduce a list of per-patient records to a single `RunKPIs` summary."""
    if not records:
        # Empty run guard. Return zeros / neutral objects so callers don't have to
        # special-case "no patients".
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
    # sim_duration: last verdict timestamp. Defines the "end of simulation" for
    # throughput, even though Python wall-clock may differ.
    last_t = max(r.ts_doctor_end for r in records)

    # Three queue-time decompositions, per patient:
    #   wait_to_model  = time spent in input_queue (Store, FIFO)
    #   wait_to_doctor = time spent in triaged_store (PriorityStore, sort-aware)
    #   time_in_system = end-to-end latency (arrival → verdict)
    # The mean of each gives a single number characterizing the run.
    waits_model = [r.ts_model_start - r.ts_arrival for r in records]
    waits_doctor = [r.ts_doctor_start - r.ts_model_end for r in records]
    sys_times = [r.ts_doctor_end - r.ts_arrival for r in records]

    return RunKPIs(
        n_patients=n,
        sim_duration=last_t,
        throughput=_safe_div(n, last_t),  # patients per simulated time unit
        mean_wait_to_model=sum(waits_model) / n,
        mean_wait_to_doctor=sum(waits_doctor) / n,
        mean_time_in_system=sum(sys_times) / n,
        ai=_actor_kpis(records, "ai"),
        doctor=_actor_kpis(records, "doctor"),
        hybrid=_actor_kpis(records, "hybrid"),
    )
