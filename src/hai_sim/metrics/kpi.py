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
from typing import Any, Literal

from hai_sim.metrics.collector import PatientRecord
from hai_sim.metrics.registry import register_bundle, resolve_bundle

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
class TeamKPIs:
    """Human-AI team metrics: is the aided doctor better than either alone, and does
    it rely on the AI *appropriately*? All computed from the per-(patient, label) 2x2
    of (doctor-solo correct?) x (AI correct?), on labeled entries only. Because the
    simulator knows both counterfactuals for every case, the reliance ratios — which
    normally need a two-stage user study — are computed directly. Ratios follow
    Schemmer et al. (2023, IUI)."""

    complementarity_gap_f1: float
    """team macro-F1 - max(AI macro-F1, doctor-solo macro-F1). > 0 ⇒ the team beats the
    better of its two members ("the whole exceeds its parts")."""

    complementarity_gap_accuracy: float
    """Same, on micro accuracy. Reported alongside F1; prefer F1 under class imbalance."""

    rair: float | None
    """Relative AI Reliance: of entries where doctor-solo was WRONG but the AI was RIGHT
    (AI-reliance opportunities), the fraction the team got right (switched to the AI).
    None if there were no such opportunities."""

    rsr: float | None
    """Relative Self-Reliance: of entries where doctor-solo was RIGHT but the AI was WRONG
    (self-reliance opportunities), the fraction the team got right (kept its own answer).
    None if there were no such opportunities."""

    over_reliance: float | None
    """1 - RSR: fraction of self-reliance opportunities lost by following wrong AI — the
    'agency loss' the reliance literature warns explanations inflate."""

    under_reliance: float | None
    """1 - RAIR: fraction of AI-reliance opportunities missed by ignoring correct AI."""

    n_ai_reliance_opportunities: int
    """# labeled entries where doctor-solo wrong and AI right (RAIR denominator)."""

    n_self_reliance_opportunities: int
    """# labeled entries where doctor-solo right and AI wrong (RSR denominator)."""


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
    """Metrics for the joint Human-AI verdict = the doctor stage's actual output. When
    the doctor is an `AssistedDoctor`, this is the *aided* (team) call and diverges from
    `doctor` (the unaided solo call); otherwise the two coincide. This is the KPI block
    that tells us whether the AI-in-the-loop is worth deploying."""

    team: TeamKPIs
    """Complementarity + appropriate-reliance of the team vs its two members (see
    `TeamKPIs`). Non-trivial only when the doctor is AI-assisted; otherwise the gaps are
    ~0 and there are no reliance opportunities."""


def _labels_for(record: PatientRecord, actor: Actor) -> tuple[int, ...]:
    """Pick which predicted label vector to score for the given actor.

    - ``ai``: the AI's solo call.
    - ``doctor``: the *unaided* (solo) doctor call — the counterfactual an
      `AssistedDoctor` stashes in ``doctor_solo_labels``; falls back to
      ``doc_labels`` when the doctor was not AI-assisted (then the two are identical).
    - ``hybrid``: the doctor stage's actual output = the *aided* (team) call when
      assisted, otherwise the solo call.
    """
    if actor == "ai":
        return record.ai_labels
    if actor == "doctor":
        if record.doctor_solo_labels is not None:
            return record.doctor_solo_labels
        return record.doc_labels
    if actor == "hybrid":
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


def _team_kpis(
    records: Sequence[PatientRecord],
    ai: ActorKPIs,
    doctor: ActorKPIs,
    hybrid: ActorKPIs,
) -> TeamKPIs:
    """Complementarity + appropriate reliance from the solo / AI / team label vectors.

    Walks every labeled (patient, label) entry and bins it by the (doctor-solo correct?
    x AI correct?) 2x2. The two off-diagonal cells are the reliance opportunities:
    doctor-wrong/AI-right (should adopt the AI) and doctor-right/AI-wrong (should keep
    own). RAIR/RSR measure how often the team did the right thing in each.
    """
    n_air = n_sr = rair_num = rsr_num = 0
    for r in records:
        ai_pred = _labels_for(r, "ai")
        doc_pred = _labels_for(r, "doctor")
        team_pred = _labels_for(r, "hybrid")
        for a, d, t, g in zip(ai_pred, doc_pred, team_pred, r.ground_truth, strict=True):
            if g not in (0, 1):
                continue  # unannotated entry — skip (matches _per_label_counts)
            ai_ok = a == g
            doc_ok = d == g
            team_ok = t == g
            if doc_ok and not ai_ok:  # self-reliance opportunity: keep own correct answer
                n_sr += 1
                rsr_num += int(team_ok)
            elif ai_ok and not doc_ok:  # AI-reliance opportunity: adopt correct AI answer
                n_air += 1
                rair_num += int(team_ok)
    rair = rair_num / n_air if n_air else None
    rsr = rsr_num / n_sr if n_sr else None
    return TeamKPIs(
        complementarity_gap_f1=hybrid.f1_macro - max(ai.f1_macro, doctor.f1_macro),
        complementarity_gap_accuracy=hybrid.accuracy - max(ai.accuracy, doctor.accuracy),
        rair=rair,
        rsr=rsr,
        over_reliance=(1.0 - rsr) if rsr is not None else None,
        under_reliance=(1.0 - rair) if rair is not None else None,
        n_ai_reliance_opportunities=n_air,
        n_self_reliance_opportunities=n_sr,
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
            team=TeamKPIs(0.0, 0.0, None, None, None, None, 0, 0),
        )

    sm = _system_metrics(records)
    ai_k = _actor_kpis(records, "ai")
    doctor_k = _actor_kpis(records, "doctor")
    hybrid_k = _actor_kpis(records, "hybrid")
    return RunKPIs(
        n_patients=int(sm["n_patients"]),
        sim_duration=sm["sim_duration"],
        throughput=sm["throughput"],
        mean_wait_to_model=sm["mean_wait_to_model"],
        mean_wait_to_doctor=sm["mean_wait_to_doctor"],
        mean_time_in_system=sm["mean_time_in_system"],
        ai=ai_k,
        doctor=doctor_k,
        hybrid=hybrid_k,
        team=_team_kpis(records, ai_k, doctor_k, hybrid_k),
    )


# ── System metrics ────────────────────────────────────────────────────────────


def _system_metrics(records: Sequence[PatientRecord]) -> dict[str, float]:
    """System-level metrics (throughput, queue waits) as a name→value dict.

    Single source of truth for both `compute_kpis` (packed into `RunKPIs`) and the
    swappable ``system`` KPI bundle.
    """
    if not records:
        return {
            "n_patients": 0.0,
            "sim_duration": 0.0,
            "throughput": 0.0,
            "mean_wait_to_model": 0.0,
            "mean_wait_to_doctor": 0.0,
            "mean_time_in_system": 0.0,
        }
    n = len(records)
    last_t = max(r.ts_doctor_end for r in records)
    waits_model = [r.ts_model_start - r.ts_arrival for r in records]
    waits_doctor = [r.ts_doctor_start - r.ts_model_end for r in records]
    sys_times = [r.ts_doctor_end - r.ts_arrival for r in records]
    return {
        "n_patients": float(n),
        "sim_duration": last_t,
        "throughput": _safe_div(n, last_t),
        "mean_wait_to_model": sum(waits_model) / n,
        "mean_wait_to_doctor": sum(waits_doctor) / n,
        "mean_time_in_system": sum(sys_times) / n,
    }


# ── AI x Doctor x ground-truth joint contingency table ──────────────────────────


@dataclass(frozen=True)
class JointTable:
    """AI x Doctor x ground-truth joint contingency over labeled (patient, label) entries.

    The AIxDoctorxGT table the thesis' contingency analysis needs (cf. tabella-TPFNFPTN):
    two 2x2 confusion matrices (``ai_confusion`` = AI vs GT, ``doctor_confusion`` =
    Doctor vs GT) plus ``joint`` — the 8 cells that, for each ground-truth sign, give the
    2x2 count of (AI positive/negative) x (Doctor positive/negative). 'Doctor' is the final
    team (hybrid) decision. Raw counts; the reliance ratios in `TeamKPIs` summarize the
    same joint structure. Keys of ``joint``: ``"gt{g}_ai{a}_doc{d}"`` with g,a,d in {1,0}.
    """

    n_entries: int
    ai_confusion: dict[str, int]
    doctor_confusion: dict[str, int]
    joint: dict[str, int]


def _bump(conf: dict[str, int], pred: int, gt: int) -> None:
    if pred == 1 and gt == 1:
        conf["tp"] += 1
    elif pred == 1 and gt == 0:
        conf["fp"] += 1
    elif pred == 0 and gt == 1:
        conf["fn"] += 1
    else:
        conf["tn"] += 1


def _joint_table(records: Sequence[PatientRecord]) -> JointTable:
    ai_conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    doc_conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    joint = {f"gt{g}_ai{a}_doc{d}": 0 for g in (1, 0) for a in (1, 0) for d in (1, 0)}
    n = 0
    for r in records:
        ai_pred = _labels_for(r, "ai")
        doc_pred = _labels_for(r, "hybrid")  # the doctor's final (team) call
        for a, d, g in zip(ai_pred, doc_pred, r.ground_truth, strict=True):
            if g not in (0, 1):
                continue  # unannotated entry — skip (matches the actor/team KPIs)
            n += 1
            _bump(ai_conf, a, g)
            _bump(doc_conf, d, g)
            joint[f"gt{g}_ai{a}_doc{d}"] += 1
    return JointTable(
        n_entries=n, ai_confusion=ai_conf, doctor_confusion=doc_conf, joint=joint
    )


# ── The built-in KPI bundles ────────────────────────────────────────────────────
# The registry itself lives in `metrics.registry`: a bundle can equally well come from
# the researcher's own file, referenced from the scenario as 'my_metrics.py:CostBundle'.


class SystemBundle:
    name = "system"

    def compute(self, records: Sequence[PatientRecord]) -> dict[str, float]:
        return _system_metrics(records)


class ActorsBundle:
    name = "actors"

    def compute(self, records: Sequence[PatientRecord]) -> dict[str, ActorKPIs]:
        return {
            "ai": _actor_kpis(records, "ai"),
            "doctor": _actor_kpis(records, "doctor"),
            "hybrid": _actor_kpis(records, "hybrid"),
        }


class TeamBundle:
    name = "team"

    def compute(self, records: Sequence[PatientRecord]) -> TeamKPIs:
        ai_k = _actor_kpis(records, "ai")
        doctor_k = _actor_kpis(records, "doctor")
        hybrid_k = _actor_kpis(records, "hybrid")
        return _team_kpis(records, ai_k, doctor_k, hybrid_k)


class JointTableBundle:
    name = "joint_table"

    def compute(self, records: Sequence[PatientRecord]) -> JointTable:
        return _joint_table(records)


for _builtin in (SystemBundle(), ActorsBundle(), TeamBundle(), JointTableBundle()):
    register_bundle(_builtin, replace=True)

DEFAULT_KPI_BUNDLES: tuple[str, ...] = ("system", "actors", "team", "joint_table")


def compute_report(
    records: Sequence[PatientRecord], bundle_names: Sequence[str] | None = None
) -> dict[str, Any]:
    """Compute the selected KPI bundles over the run telemetry.

    ``bundle_names`` selects the swappable set (default = all standard bundles). Each
    entry is a registered name *or* a reference to the researcher's own bundle, e.g.
    ``"analysis/my_metrics.py:CostBundle"`` — see `hai_sim.metrics.registry`. The
    report is keyed by each bundle's own ``name``, so a custom bundle is reachable by
    the study layer's dotted paths exactly like a built-in one.
    """
    names = tuple(bundle_names) if bundle_names is not None else DEFAULT_KPI_BUNDLES
    report: dict[str, Any] = {}
    for name in names:
        bundle = resolve_bundle(name)
        report[bundle.name] = bundle.compute(records)
    return report
