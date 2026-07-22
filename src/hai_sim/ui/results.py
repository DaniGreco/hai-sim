"""Reading an output directory: what ran, what it measured, and how two runs differ.

Everything here works off the **JSON** a run leaves behind, not the CSVs. That is not an
arbitrary preference: the JSON is self-describing — a study's carries its whole
`StudyConfig` including the base scenario, and a single run's carries its whole
`ScenarioConfig` plus the module names the engine resolved. So an artefact from six
months ago can be read without knowing what the flags meant at the time, and the
configuration that produced a number is always available *next to* the number.

The CSVs remain the right thing for anything row-oriented and large — per-patient
records — and `load_records` reads those.

Two artefact shapes, told apart by their keys rather than by filename, so a renamed file
still classifies correctly:

``study``  keys ``study`` / ``rows`` / ``summaries`` / ``contrasts``
    A factorial sweep: one row per (cell x seed), aggregated summaries with intervals,
    and paired contrasts against a baseline arm.
``run``    keys ``scenario`` / ``kpis``
    One scenario, one seed, one KPI report.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Columns of a study's long format that identify the cell rather than measure it.
_LONG_ID_COLUMNS = ("seed", "group")


@dataclass(frozen=True)
class Artifact:
    """One result file on disk, already parsed and classified."""

    path: Path
    kind: str  # "study" | "run"
    name: str
    stamp: str
    payload: Mapping[str, Any] = field(repr=False)

    @property
    def pretty_stamp(self) -> str:
        """``20260721-162226`` -> ``07-21 16:22``.

        The year is dropped and the seconds with it: a picker line is scanned, not
        parsed, and what disambiguates two entries is almost always the day and the
        hour. `path` remains the identity.
        """
        date, _, clock = self.stamp.partition("-")
        if len(date) != 8 or len(clock) < 4:
            return self.stamp
        return f"{date[4:6]}-{date[6:8]} {clock[:2]}:{clock[2:4]}"

    @property
    def label(self) -> str:
        """What the picker shows: **what is in this artefact**, not just when it ran.

        A name and a timestamp identify a file; they do not tell you whether it is the
        smoke run or the thousand-patient one, which persona it used, or what it swept.
        Everything needed for that is already in the payload, so the picker has no
        excuse for making you open a file to find out what it is.
        """
        parts = [self.name, describe(self)]
        if self.pretty_stamp:
            parts.append(self.pretty_stamp)
        return "  ·  ".join(p for p in parts if p)

    @property
    def scenario(self) -> Mapping[str, Any]:
        """The configuration that produced this artefact, whichever shape it is.

        For a study that is the *base* scenario — the factors then vary paths within
        it, and `factor_names` says which.
        """
        if self.kind == "study":
            base = self.payload.get("study", {}).get("base")
            return dict(base) if isinstance(base, Mapping) else {"base": base}
        scenario = self.payload.get("scenario", {})
        return dict(scenario) if isinstance(scenario, Mapping) else {}


# --- saying what an artefact is ---------------------------------------------
# Everything below reads the payload only. It exists so a picker line answers "what is
# this?" instead of "when did it run?", which is the question a directory of thirty
# similarly-named files actually raises.

_MAX_LISTED_FACTORS = 2


def _short_path(dotted: str) -> str:
    """``persona.capability.difficulty_slope`` -> ``capability.difficulty_slope``.

    The last two segments: one is often ambiguous (half the blocks have a ``kind``),
    three is too long for a picker line.
    """
    parts = dotted.split(".")
    return ".".join(parts[-2:]) if len(parts) > 1 else dotted


def _persona_name(persona: Any) -> str:
    """A persona is a library name or an inline block; both must render."""
    if isinstance(persona, str):
        return persona
    if isinstance(persona, Mapping):
        name = str(persona.get("name") or "inline")
        # Whether the doctor sees the AI, and on what terms, is the most consequential
        # thing about a persona — and an inline one need not say so in its name, so
        # "inline" alone would tell the reader nothing.
        if persona.get("sees_ai") is False:
            return name if name.startswith("solo") else f"{name}/solo"
        trust = persona.get("trust")
        if isinstance(trust, Mapping) and trust.get("kind"):
            return f"{name}/{trust['kind']}"
        return name
    return "?"


def _describe_study(artifact: Artifact) -> str:
    study = artifact.payload.get("study", {})
    if not isinstance(study, Mapping):
        return ""
    bits: list[str] = []
    factors = study.get("factors", {})
    if isinstance(factors, Mapping) and factors:
        if len(factors) <= _MAX_LISTED_FACTORS:
            bits.append(
                ", ".join(f"{_short_path(k)}x{len(v)}" for k, v in factors.items())
            )
        else:
            cells = 1
            for levels in factors.values():
                cells *= max(1, len(levels))
            bits.append(f"{len(factors)} factors, {cells} cells")
    reps = study.get("replications") or []
    if len(reps) > 1:
        bits.append(f"{len(reps)} seeds")
    baseline = study.get("baseline")
    if isinstance(baseline, Mapping) and baseline:
        # Without a baseline there are no paired contrasts, so its presence changes
        # what the artefact can answer, not merely how it was configured.
        bits.append("vs " + ", ".join(str(v) for v in baseline.values()))
    return " · ".join(bits)


def _describe_run(artifact: Artifact) -> str:
    scenario = artifact.scenario
    bits: list[str] = []
    generator = scenario.get("patient_generator")
    if isinstance(generator, Mapping) and generator.get("n_patients"):
        bits.append(f"{generator['n_patients']} pt")
    ai = scenario.get("ai")
    if isinstance(ai, Mapping) and ai.get("kind"):
        bits.append(str(ai["kind"]))
    if "persona" in scenario:
        bits.append(_persona_name(scenario["persona"]))
    sort = scenario.get("sort")
    if isinstance(sort, Mapping) and sort.get("kind"):
        bits.append(str(sort["kind"]))
    # Which optional modules were configured. Two runs identical apart from a difficulty
    # model are not comparable in the way their shared name suggests, so the flags earn
    # their space.
    meta = artifact.payload.get("metadata", {})
    flags = [
        short
        for key, short in (
            ("difficulty_model", "diff"),
            ("fatigue_curve", "fatigue"),
            ("severity_model", "sev"),
        )
        if isinstance(meta, Mapping) and meta.get(key) not in (None, "none", "")
    ]
    if flags:
        bits.append("+" + " +".join(flags))
    return " · ".join(bits)


def describe(artifact: Artifact) -> str:
    """One line saying what is *in* an artefact — its content, not its filename."""
    return _describe_study(artifact) if artifact.kind == "study" else _describe_run(artifact)


def _classify(payload: Mapping[str, Any]) -> str | None:
    if "rows" in payload and "study" in payload:
        return "study"
    if "kpis" in payload and "scenario" in payload:
        return "run"
    return None


def _split_stem(stem: str) -> tuple[str, str]:
    """Split ``<name>_<YYYYmmdd>_<HHMMSS>`` into a name and a timestamp.

    Falls back to the whole stem when a file was named by hand — a browser that only
    listed conventionally-named files would quietly hide exactly the ad-hoc runs
    someone most wants to look at.
    """
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2]), f"{parts[-2]}-{parts[-1]}"
    return stem, ""


def discover(out_dir: str | Path) -> list[Artifact]:
    """Every readable result under ``out_dir``, newest first.

    Unreadable or unrecognised JSON is skipped silently: an output directory
    accumulates all sorts of things, and a browser that refused to start because of one
    stray file would be useless precisely when there is most to look at.
    """
    root = Path(out_dir)
    if not root.is_dir():
        return []
    found: list[Artifact] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        kind = _classify(payload)
        if kind is None:
            continue
        stem = path.stem
        if kind == "study" and stem.startswith("study_"):
            stem = stem[len("study_") :]
        elif kind == "run" and stem.startswith("summary_"):
            stem = stem[len("summary_") :]
        name, stamp = _split_stem(stem)
        found.append(Artifact(path=path, kind=kind, name=name, stamp=stamp, payload=payload))
    return sorted(found, key=lambda a: (a.stamp, a.name), reverse=True)


def factor_names(artifact: Artifact) -> list[str]:
    """The swept factors — the columns a chart can put on an axis or a colour."""
    if artifact.kind != "study":
        return []
    return [str(n) for n in artifact.payload.get("factor_names", [])]


def study_rows(artifact: Artifact) -> list[dict[str, Any]]:
    """One row per (cell x seed), wide: factors + seed + group + one column per metric."""
    rows = artifact.payload.get("rows", [])
    return [dict(r) for r in rows if isinstance(r, Mapping)]


def metric_names(artifact: Artifact) -> list[str]:
    """Metric columns, in the order the study reported them.

    Derived from the rows rather than from ``study.report`` because ``report`` is
    optional — leaving it out means "every scalar", and then only the rows know what
    that turned out to be.
    """
    if artifact.kind == "run":
        return [name for name, _ in run_kpi_rows(artifact)]
    seen: dict[str, None] = {}
    known_factors = set(factor_names(artifact)) | set(_LONG_ID_COLUMNS)
    for row in study_rows(artifact):
        for key, value in row.items():
            if key not in known_factors and isinstance(value, (int, float)):
                seen.setdefault(key, None)
    return list(seen)


def summary_rows(artifact: Artifact) -> list[dict[str, Any]]:
    """Per-cell aggregates: mean over seeds with a 95% interval."""
    out: list[dict[str, Any]] = []
    for entry in artifact.payload.get("summaries", []):
        if not isinstance(entry, Mapping):
            continue
        row = dict(entry.get("factors", {}))
        row.update(
            {
                "group": entry.get("group", ""),
                "metric": entry.get("metric"),
                "n": entry.get("n"),
                "mean": entry.get("mean"),
                "sd": entry.get("sd"),
                "ci_low": entry.get("ci_low"),
                "ci_high": entry.get("ci_high"),
            }
        )
        out.append(row)
    return out


def contrast_rows(artifact: Artifact) -> list[dict[str, Any]]:
    """Paired contrasts against the baseline arm — what a forest plot draws.

    ``delta_mean`` is the mean **per-seed** difference, not the difference of the means.
    That distinction is the entire value of the common random numbers: paired at the
    seed, the sampling noise cancels instead of being averaged over, and the interval
    around it is correspondingly narrow. ``excludes_zero`` is the study layer's own
    verdict on whether the interval clears zero.
    """
    out: list[dict[str, Any]] = []
    for entry in artifact.payload.get("contrasts", []):
        if not isinstance(entry, Mapping):
            continue
        arm = entry.get("factors", {})
        baseline = entry.get("baseline", {})
        row = dict(arm)
        row.update(
            {
                "arm": _render_cell(arm),
                "baseline": _render_cell(baseline),
                "group": entry.get("group", ""),
                "metric": entry.get("metric"),
                "delta_mean": entry.get("delta_mean"),
                "delta_ci_low": entry.get("delta_ci_low"),
                "delta_ci_high": entry.get("delta_ci_high"),
                "n_pairs": entry.get("n_pairs"),
                "excludes_zero": bool(entry.get("excludes_zero")),
                "arm_mean": entry.get("arm_mean"),
                "baseline_mean": entry.get("baseline_mean"),
            }
        )
        out.append(row)
    return out


def _render_cell(factors: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in factors.items()) or "(base)"


def run_kpi_rows(artifact: Artifact) -> list[tuple[str, float]]:
    """A single run's KPI report flattened to ``(dotted.path, value)`` pairs.

    Per-label vectors are skipped rather than averaged: a macro average of a vector
    already has a name and a definition in the KPI bundle that produced it, and
    inventing a second one here would put two different numbers under one label.
    """
    return _flatten(artifact.payload.get("kpis", {}), prefix="")


def run_group_rows(artifact: Artifact) -> dict[str, list[tuple[str, float]]]:
    """The same, once per conditioning subgroup (per difficulty band, per load, ...)."""
    groups = artifact.payload.get("kpis_by_group", {})
    if not isinstance(groups, Mapping):
        return {}
    return {str(key): _flatten(value, prefix="") for key, value in groups.items()}


def _flatten(node: Any, prefix: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_flatten(value, path))
    elif isinstance(node, (bool, int, float)):
        # bool is a subclass of int, so this one branch covers it; a KPI that reports a
        # flag still plots as 0/1 rather than being silently dropped.
        out.append((prefix, float(node)))
    return out


def load_records(artifact: Artifact) -> list[dict[str, str]]:
    """The per-patient telemetry beside a single-run summary, if it is still there.

    Records are written as ``records_<name>_<stamp>.csv`` next to
    ``summary_<name>_<stamp>.json``, and they are the only place the per-read signal
    plane (``sig.difficulty``, ``sig.fatigue``) and the load variables live. Returns an
    empty list when the file was moved or never written — the summary is useful on its
    own and should not stop being readable because its sibling went missing.
    """
    if artifact.kind != "run":
        return []
    stem = artifact.path.stem
    if stem.startswith("summary_"):
        stem = "records_" + stem[len("summary_") :]
    path = artifact.path.with_name(stem).with_suffix(".csv")
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def numeric_column(records: Sequence[Mapping[str, str]], key: str) -> list[float]:
    """One column of the records CSV as floats, dropping blanks and non-numbers.

    Records from different runs legitimately carry different columns (a run without a
    fatigue curve has no ``sig.fatigue``), so a missing key is empty, not an error.
    """
    out: list[float] = []
    for row in records:
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def observation_keys(records: Sequence[Mapping[str, str]], prefix: str = "") -> list[str]:
    """Numeric record columns, optionally restricted to one namespace (``sig.``)."""
    if not records:
        return []
    head = records[0]
    keys = []
    for key in head:
        if prefix and not key.startswith(prefix):
            continue
        if numeric_column(records[: min(len(records), 20)], key):
            keys.append(key)
    return keys


# --- comparing two runs ------------------------------------------------------


def flatten_config(node: Any, prefix: str = "") -> dict[str, Any]:
    """A nested configuration as ``dotted.path -> scalar``, lists rendered inline.

    Lists stay whole rather than being expanded to ``path.0``, ``path.1``: an ensemble's
    weight list or a rate table is one decision, and a diff that reported it as fourteen
    changed lines would bury the one thing that actually differs.
    """
    out: dict[str, Any] = {}
    if isinstance(node, Mapping):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_config(value, path))
    elif isinstance(node, (list, tuple)):
        out[prefix] = list(node)
    else:
        out[prefix] = node
    return out


@dataclass(frozen=True)
class ConfigDelta:
    """One line of a configuration diff."""

    path: str
    left: Any
    right: Any

    @property
    def status(self) -> str:
        if self.left is None and self.right is not None:
            return "only right"
        if self.right is None and self.left is not None:
            return "only left"
        return "differs"


def config_diff(left: Any, right: Any) -> list[ConfigDelta]:
    """Every configuration key on which two artefacts disagree, sorted by path.

    This is the question a comparison exists to answer: two runs give different numbers
    — *what was different about them?* Reading two YAML files side by side answers it
    badly, because they are long and mostly identical. Keys present in only one side are
    included, since an added block (a difficulty model appearing) is exactly the kind of
    change that explains a result.
    """
    lhs = flatten_config(left)
    rhs = flatten_config(right)
    deltas: list[ConfigDelta] = []
    for path in sorted(set(lhs) | set(rhs)):
        a, b = lhs.get(path), rhs.get(path)
        if a != b:
            deltas.append(ConfigDelta(path=path, left=a, right=b))
    return deltas


def common_metrics(artifacts: Iterable[Artifact]) -> list[str]:
    """Metrics every given artefact reports — the only ones comparable across them."""
    shared: set[str] | None = None
    for artifact in artifacts:
        names = set(metric_names(artifact))
        shared = names if shared is None else (shared & names)
    return sorted(shared or set())
