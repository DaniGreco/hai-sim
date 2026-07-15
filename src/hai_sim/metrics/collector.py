"""Per-patient record collector. The KPI module reads from this."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hai_sim.core.types import ModelOutput, Patient


def _norm_severity(sev: float | np.ndarray | None) -> float | tuple[float, ...] | None:
    """Normalize a severity value for storage: per-label arrays → tuples."""
    if sev is None:
        return None
    if isinstance(sev, np.ndarray):
        return tuple(float(x) for x in sev.ravel())
    return float(sev)


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
    # Unaided-doctor counterfactual: set by an AssistedDoctor via
    # ``extra["doctor_solo_labels"]``; None when the doctor was not AI-assisted (then
    # doc_labels already IS the solo verdict). Lets the KPI layer separate the aided
    # (team = doc_labels) call from the solo call for complementarity/reliance.
    doctor_solo_labels: tuple[int, ...] | None = None
    # Per-patient severity from an optional SeverityModel: a scalar, a per-label
    # vector (stored as a tuple), or None when no severity model is configured.
    severity: float | tuple[float, ...] | None = None


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
        severity: float | np.ndarray | None = None,
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
                doctor_solo_labels=doc_out.extra.get("doctor_solo_labels"),
                severity=_norm_severity(severity),
            )
        )

    def write_csv(self, path: Path, label_names: tuple[str, ...]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.records:
            path.write_text("")
            return
        # Severity is optional and either scalar or per-label; add columns only if
        # present, and pick the layout from the first record that carries one.
        sev_vals = [r.severity for r in self.records if r.severity is not None]
        sev_mode = "none"
        if sev_vals:
            sev_mode = "vector" if isinstance(sev_vals[0], tuple) else "scalar"

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
        if sev_mode == "scalar":
            header.append("severity")
        elif sev_mode == "vector":
            header.extend(f"severity__{name}" for name in label_names)

        def _sev_cells(r: PatientRecord) -> list[str]:
            if sev_mode == "scalar":
                return [f"{r.severity:.6f}"] if isinstance(r.severity, float) else [""]
            if sev_mode == "vector":
                if isinstance(r.severity, tuple):
                    return [f"{x:.6f}" for x in r.severity]
                return [""] * len(label_names)
            return []

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
                        *_sev_cells(r),
                    ]
                )
