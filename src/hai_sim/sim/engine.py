"""SimPy orchestration of the v0 pipeline.

Pipeline:
    PoissonPatientGenerator → input_queue (Store)
                            → model_worker × N_ai (loops on input_queue)
                            → triaged_store (PriorityStore, key=SortStrategy)
                            → doctor_worker × N_doc (loops on triaged_store)
                            → MetricsCollector
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

    from hai_sim.core.protocols import DiagnosticActor, SortStrategy


@dataclass
class _InFlight:
    """Bookkeeping passed between stations."""

    patient: Patient
    ts_arrival: float
    ts_model_start: float = 0.0
    ts_model_end: float = 0.0
    ai_out: ModelOutput | None = None
    priority_key: float = 0.0


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
        n_ai_workers: int = 1,
        n_doctor_workers: int = 1,
    ) -> None:
        self.patient_generator = patient_generator
        self.ai = ai
        self.doctor = doctor
        self.sort_strategy = sort_strategy
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
                yield input_queue.put(_InFlight(patient=stamped, ts_arrival=ts_arrival))
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
                inflight.priority_key = self.sort_strategy.priority(ai_out, inflight.patient)
                yield triaged_store.put(
                    simpy.PriorityItem(priority=inflight.priority_key, item=inflight)
                )

        def doctor_worker():
            while True:
                pitem = yield triaged_store.get()
                inflight: _InFlight = pitem.item
                assert inflight.ai_out is not None
                ts_doctor_start = env.now
                yield env.timeout(
                    self.doctor.service_time(inflight.patient.images, inflight.patient.context)
                )
                doc_out = self.doctor.diagnose(
                    inflight.patient.images,
                    context=inflight.patient.context,
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
