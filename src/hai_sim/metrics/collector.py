"""Per-patient record collector. The KPI module reads from this.

The record has a **fixed core** (identity, timestamps, the three label vectors,
priority, severity) plus an open ``observations`` bag. The bag is what makes the
telemetry lossless: whatever a module deposits in `ModelOutput.extra` reaches it,
namespaced by producing stage, so a new module can surface a new observable
without anyone editing this file. Namespaces:

- ``ai.*``   — from the AI station's `ModelOutput.extra` (e.g. the aleatoric /
  epistemic decomposition of the uncertainty, the cache row, the image id)
- ``doc.*``  — from the doctor station's `ModelOutput.extra`
- ``ctx.*``  — the `PatientContext` fields worth conditioning on (image identity,
  age, sex, priority)
- ``load.*`` — queue occupancy sampled by the engine at the stage transitions;
  the workload variables, without which order effects cannot be studied at all
  (they only exist under saturation).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hai_sim.core.types import ModelOutput, Patient, PatientContext

# Observation keys already promoted to first-class record fields: keeping them in
# the bag too would just duplicate columns.
_PROMOTED_DOC_KEYS = frozenset({"doctor_solo_labels"})

# PatientContext fields exposed as ``ctx.*`` observations. patient_id, arrival_time
# and ground_truth are excluded: they are already first-class record fields.
_CTX_FIELDS = ("example_id", "age", "sex", "priority")


def _norm_severity(sev: float | np.ndarray | None) -> float | tuple[float, ...] | None:
    """Normalize a severity value for storage: per-label arrays → tuples."""
    if sev is None:
        return None
    if isinstance(sev, np.ndarray):
        return tuple(float(x) for x in sev.ravel())
    return float(sev)


def _observations(
    context: PatientContext | None,
    ai_out: ModelOutput,
    doc_out: ModelOutput,
    load: Mapping[str, float] | None,
) -> dict[str, Any]:
    """Assemble the namespaced observation bag for one patient."""
    obs: dict[str, Any] = {}
    if context is not None:
        for name in _CTX_FIELDS:
            value = getattr(context, name, None)
            if value is not None:
                obs[f"ctx.{name}"] = value
    for key, value in ai_out.extra.items():
        obs[f"ai.{key}"] = value
    for key, value in doc_out.extra.items():
        if key not in _PROMOTED_DOC_KEYS:
            obs[f"doc.{key}"] = value
    if load:
        for key, value in load.items():
            obs[f"load.{key}"] = value
    return obs


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
    # Open, namespaced observation bag — see the module docstring. Anything a module
    # produces lands here, so KPI bundles can read observables this file never
    # anticipated.
    observations: dict[str, Any] = field(default_factory=dict)

    def observation(self, key: str, default: Any = None) -> Any:
        """Read one observation, or ``default`` when this record does not carry it.

        Records may legitimately have different keys (a run can swap the AI mid-study),
        so consumers should never assume a key is present.
        """
        return self.observations.get(key, default)


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
        load: Mapping[str, float] | None = None,
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
                observations=_observations(patient.context, ai_out, doc_out, load),
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
        obs_plan = _observation_plan(self.records, len(label_names))

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
        for key, per_label in obs_plan:
            if per_label:
                header.extend(f"{key}__{name}" for name in label_names)
            else:
                header.append(key)

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
                        *_observation_cells(r, obs_plan, len(label_names)),
                    ]
                )


# ── Flattening the observation bag into CSV columns ─────────────────────────────


def _as_sequence(value: Any) -> list[Any] | None:
    """``value`` as a plain list when it is a non-string sequence, else ``None``."""
    if isinstance(value, np.ndarray):
        return [x.item() for x in value.ravel()]
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _is_label_vector(value: Any, n_labels: int) -> bool:
    seq = _as_sequence(value)
    return seq is not None and len(seq) == n_labels


def _observation_plan(
    records: Sequence[PatientRecord], n_labels: int
) -> list[tuple[str, bool]]:
    """Column plan for the observation bag: ``(key, is_per_label)``, key-sorted.

    A key is expanded into one column per label only when *every* record carrying it
    holds a sequence of exactly ``n_labels`` entries. Mixed or odd-length values fall
    back to a single JSON column, so a key can never silently misalign.
    """
    keys: set[str] = set()
    for r in records:
        keys.update(r.observations)
    plan: list[tuple[str, bool]] = []
    for key in sorted(keys):
        values = [r.observations[key] for r in records if key in r.observations]
        per_label = bool(values) and all(_is_label_vector(v, n_labels) for v in values)
        plan.append((key, per_label))
    return plan


def _fmt_observation(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):  # before int: bool is an int subclass
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _observation_cells(
    record: PatientRecord, plan: Sequence[tuple[str, bool]], n_labels: int
) -> list[str]:
    cells: list[str] = []
    for key, per_label in plan:
        if key not in record.observations:
            cells.extend([""] * (n_labels if per_label else 1))
            continue
        value = record.observations[key]
        if per_label:
            cells.extend(_fmt_observation(x) for x in (_as_sequence(value) or []))
        else:
            cells.append(_fmt_observation(value))
    return cells
