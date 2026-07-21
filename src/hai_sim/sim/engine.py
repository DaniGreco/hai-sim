"""The reading room: runs the AI and the doctor **in series** over a patient stream.

Built on SimPy, a discrete-event simulator — time jumps from one event to the next
(an arrival, a read finishing) rather than ticking, so a thousand patients take
milliseconds and "simulated time" has nothing to do with Python's wall clock.

    PatientGenerator
        │  arrivals, one at a time
        ▼
    input_queue          plain first-in-first-out Store
        │
        ▼
    model_worker  x n_ai_workers          ← the AI reads first
        │  produces a per-label prediction + uncertainty,
        │  from which SortStrategy derives one priority number
        ▼
    triaged_store        PriorityStore — THIS is where triage happens
        │  lower priority value is served first, so the AI's
        │  opinion decides who the doctor sees next
        ▼
    doctor_worker x n_doctor_workers      ← the doctor reads second,
        │                                   with the AI's output as a hint
        ▼
    MetricsCollector     one record per patient, plus what each stage published

Three things about this arrangement are load-bearing:

1. **The AI is upstream of the queue, not beside it.** Its uncertainty is available
   *before* the doctor picks the next patient, which is the only reason triage by
   uncertainty is possible at all. Reordering only changes outcomes when patients are
   waiting — with an idle queue there is nothing to reorder, so any triage study has to
   be run at a load that actually saturates the doctor.

2. **Only the doctor sees the ground truth.** The engine injects it into the context
   handed to the doctor stage and nowhere else. It is a simulation device — the
   mechanism by which a *simulated* reader is made to err at a chosen rate — not
   information any real reader would have. An image-reading model must never receive it,
   and the AI stage above is deliberately handed the plain context.

3. **The doctor's own opinion is formed before the AI's is folded in.** That happens
   inside `AssistedDoctor`, not here, but it is why the telemetry can carry both a solo
   and a team verdict for every patient.

Both stations can be widened (``n_ai_workers``, ``n_doctor_workers``). In practice the
doctor is the bottleneck, which is the situation of interest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import simpy

from hai_sim.core.types import ModelOutput, Patient
from hai_sim.metrics.collector import MetricsCollector

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np

    from hai_sim.core.protocols import DiagnosticActor, SeverityModel, SortStrategy


@dataclass
class _InFlight:
    """Bookkeeping passed between stations."""

    patient: Patient
    ts_arrival: float
    ts_model_start: float = 0.0
    ts_model_end: float = 0.0
    ai_out: ModelOutput | None = None
    priority_key: float = 0.0
    severity: float | np.ndarray | None = None
    # Queue occupancy sampled as this patient crosses each stage boundary; reaches the
    # telemetry as the ``load.*`` observations. Without it no metric can be conditioned
    # on workload — and order effects only exist under saturation, so a triage-order
    # study is not answerable without these.
    load: dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationResult:
    collector: MetricsCollector
    sort_name: str
    ai_name: str
    doctor_name: str
    metadata: dict[str, object] = field(default_factory=dict)


class SimulationEngine:
    """Drives one simulation run end-to-end."""

    def __init__(
        self,
        patient_generator: Iterable[tuple[float, Patient]],
        ai: DiagnosticActor,
        doctor: DiagnosticActor,
        sort_strategy: SortStrategy,
        severity_model: SeverityModel | None = None,
        n_ai_workers: int = 1,
        n_doctor_workers: int = 1,
    ) -> None:
        self.patient_generator = patient_generator
        self.ai = ai
        self.doctor = doctor
        self.sort_strategy = sort_strategy
        self.severity_model = severity_model
        self.n_ai_workers = n_ai_workers
        self.n_doctor_workers = n_doctor_workers

    def run(self) -> SimulationResult:
        env = simpy.Environment()
        input_queue: simpy.Store = simpy.Store(env)
        triaged_store: simpy.PriorityStore = simpy.PriorityStore(env)
        collector = MetricsCollector()
        state = {"emitted": 0, "completed": 0, "done_emitting": False}

        def arrival_proc():
            for delay, patient in self.patient_generator:
                yield env.timeout(delay)
                ts_arrival = env.now
                stamped = Patient(
                    id=patient.id,
                    images=patient.images,
                    ground_truth=patient.ground_truth,
                    context=replace(patient.context, arrival_time=ts_arrival),
                )
                inflight = _InFlight(patient=stamped, ts_arrival=ts_arrival)
                # Sampled *before* the put and before the counter bumps: the congestion
                # this patient sees on arrival, excluding itself.
                inflight.load["input_queue_at_arrival"] = float(len(input_queue.items))
                inflight.load["in_system_at_arrival"] = float(
                    state["emitted"] - state["completed"]
                )
                yield input_queue.put(inflight)
                state["emitted"] += 1
            state["done_emitting"] = True

        def model_worker():
            while True:
                inflight: _InFlight = yield input_queue.get()
                inflight.ts_model_start = env.now
                yield env.timeout(
                    self.ai.service_time(inflight.patient.images, inflight.patient.context)
                )
                ai_out = self.ai.diagnose(
                    inflight.patient.images,
                    context=inflight.patient.context,
                )
                inflight.ts_model_end = env.now
                inflight.ai_out = ai_out
                # Severity is a produced-and-logged signal; nothing here consumes
                # it yet (triage/KPI/cost consumers are opt-in, added separately).
                if self.severity_model is not None:
                    inflight.severity = self.severity_model.score(
                        patient=inflight.patient,
                        ai_output=ai_out,
                        context=inflight.patient.context,
                    )
                inflight.priority_key = self.sort_strategy.priority(ai_out, inflight.patient)
                # How deep the triaged queue was when this patient joined it: the sort
                # strategy can only reorder what is already waiting, so this is the
                # variable that says whether triage had any room to act.
                inflight.load["triaged_at_enqueue"] = float(len(triaged_store.items))
                yield triaged_store.put(
                    simpy.PriorityItem(priority=inflight.priority_key, item=inflight)
                )

        def doctor_worker():
            while True:
                pitem = yield triaged_store.get()
                inflight: _InFlight = pitem.item
                assert inflight.ai_out is not None
                ts_doctor_start = env.now
                inflight.load["triaged_at_read"] = float(len(triaged_store.items))
                # Inject ground truth into the doctor's context only: simulated
                # readers use it as their accuracy mechanism. The AI stage above
                # is handed the plain context and never sees the truth.
                doc_context = replace(
                    inflight.patient.context,
                    ground_truth=inflight.patient.ground_truth,
                )
                yield env.timeout(
                    self.doctor.service_time(inflight.patient.images, doc_context)
                )
                doc_out = self.doctor.diagnose(
                    inflight.patient.images,
                    context=doc_context,
                    ai_hint=inflight.ai_out,
                )
                ts_doctor_end = env.now
                collector.add(
                    patient=inflight.patient,
                    ts_arrival=inflight.ts_arrival,
                    ts_model_start=inflight.ts_model_start,
                    ts_model_end=inflight.ts_model_end,
                    ts_doctor_start=ts_doctor_start,
                    ts_doctor_end=ts_doctor_end,
                    ai_out=inflight.ai_out,
                    doc_out=doc_out,
                    priority_key=inflight.priority_key,
                    severity=inflight.severity,
                    load=inflight.load,
                )
                state["completed"] += 1

        env.process(arrival_proc())
        for _ in range(self.n_ai_workers):
            env.process(model_worker())
        for _ in range(self.n_doctor_workers):
            env.process(doctor_worker())

        # Step until all emitted patients have completed end-to-end. Worker
        # processes never terminate (they loop forever on store.get) — that's
        # fine; once `state["completed"] == state["emitted"]` we just stop
        # stepping and return.
        while True:
            if (
                state["done_emitting"]
                and state["emitted"] > 0
                and state["completed"] >= state["emitted"]
            ):
                break
            next_t = env.peek()
            if math.isinf(next_t):
                raise RuntimeError(
                    "SimPy event queue empty before completion: "
                    f"emitted={state['emitted']}, completed={state['completed']}, "
                    f"done_emitting={state['done_emitting']}"
                )
            env.step()

        return SimulationResult(
            collector=collector,
            sort_name=self.sort_strategy.name,
            ai_name=self.ai.name,
            doctor_name=self.doctor.name,
            metadata={
                "sim_end_time": env.now,
                "n_ai_workers": self.n_ai_workers,
                "n_doctor_workers": self.n_doctor_workers,
            },
        )
