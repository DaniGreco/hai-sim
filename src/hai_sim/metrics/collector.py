"""Per-patient record collector. The KPI module reads from this."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from hai_sim.core.types import ModelOutput, Patient


@dataclass
class PatientRecord:
    """One row in the simulation telemetry."""

    patient_id: int
    ground_truth: tuple[int, ...]
    ts_arrival: float
    ts_model_start: float
    ts_model_end: float
    ts_doctor_start: float
    ts_doctor_end: float
    ai_labels: tuple[int, ...]
    ai_confidence: tuple[float, ...]
    ai_uncertainty: tuple[float, ...]
    doc_labels: tuple[int, ...]
    doc_confidence: tuple[float, ...]
    doc_uncertainty: tuple[float, ...]
    priority_key: float


@dataclass
class MetricsCollector:
    records: list[PatientRecord] = field(default_factory=list)

    def add(
        self,
        patient: Patient,
        ts_arrival: float,
        ts_model_start: float,
        ts_model_end: float,
        ts_doctor_start: float,
        ts_doctor_end: float,
        ai_out: ModelOutput,
        doc_out: ModelOutput,
        priority_key: float,
    ) -> None:
        self.records.append(
            PatientRecord(
                patient_id=patient.id,
                ground_truth=patient.ground_truth,
                ts_arrival=ts_arrival,
                ts_model_start=ts_model_start,
                ts_model_end=ts_model_end,
                ts_doctor_start=ts_doctor_start,
                ts_doctor_end=ts_doctor_end,
                ai_labels=ai_out.diagnosis.labels,
                ai_confidence=ai_out.diagnosis.confidence,
                ai_uncertainty=ai_out.uncertainty,
                doc_labels=doc_out.diagnosis.labels,
                doc_confidence=doc_out.diagnosis.confidence,
                doc_uncertainty=doc_out.uncertainty,
                priority_key=priority_key,
            )
        )

    def write_csv(self, path: Path, label_names: tuple[str, ...]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.records:
            path.write_text("")
            return
        header = [
            "patient_id",
            "ts_arrival",
            "ts_model_start",
            "ts_model_end",
            "ts_doctor_start",
            "ts_doctor_end",
            "priority_key",
            *[f"gt__{name}" for name in label_names],
            *[f"ai_label__{name}" for name in label_names],
            *[f"ai_conf__{name}" for name in label_names],
            *[f"ai_unc__{name}" for name in label_names],
            *[f"doc_label__{name}" for name in label_names],
            *[f"doc_conf__{name}" for name in label_names],
            *[f"doc_unc__{name}" for name in label_names],
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in self.records:
                writer.writerow(
                    [
                        r.patient_id,
                        f"{r.ts_arrival:.6f}",
                        f"{r.ts_model_start:.6f}",
                        f"{r.ts_model_end:.6f}",
                        f"{r.ts_doctor_start:.6f}",
                        f"{r.ts_doctor_end:.6f}",
                        f"{r.priority_key:.6f}",
                        *r.ground_truth,
                        *r.ai_labels,
                        *[f"{x:.6f}" for x in r.ai_confidence],
                        *[f"{x:.6f}" for x in r.ai_uncertainty],
                        *r.doc_labels,
                        *[f"{x:.6f}" for x in r.doc_confidence],
                        *[f"{x:.6f}" for x in r.doc_uncertainty],
                    ]
                )
