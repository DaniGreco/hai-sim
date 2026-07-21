"""The metric axis is swappable from config, like every other axis.

A researcher must be able to invent a metric in their own file and name it in a
scenario, without patching the library. These tests pin that seam.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from hai_sim.analysis.inference_cache import InferenceCache
from hai_sim.core.types import LabelSchema
from hai_sim.doctors.mock import RandomMockDoctor
from hai_sim.metrics.collector import PatientRecord
from hai_sim.metrics.kpi import compute_report
from hai_sim.metrics.registry import (
    KPI_BUNDLES,
    KPIBundle,
    register_bundle,
    resolve_bundle,
)
from hai_sim.models.cached import CachedEnsembleAI
from hai_sim.sim.cache_generator import CachedPatientGenerator
from hai_sim.sim.engine import SimulationEngine
from hai_sim.sort.random import RandomSort
from hai_sim.uq.methods import EnsembleUQ

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE = _REPO_ROOT / "configs" / "examples" / "custom_metrics.py"
_SCHEMA = LabelSchema.default_v0()


@pytest.fixture
def clean_registry():
    """Restore the registry, so a test registering a bundle cannot leak into others."""
    before = dict(KPI_BUNDLES)
    yield
    KPI_BUNDLES.clear()
    KPI_BUNDLES.update(before)


def _rec(gt, ai, doc, observations=None) -> PatientRecord:
    zeros = tuple(0.0 for _ in gt)
    return PatientRecord(
        patient_id=0,
        ground_truth=gt,
        ts_arrival=0.0,
        ts_model_start=0.0,
        ts_model_end=0.0,
        ts_doctor_start=0.0,
        ts_doctor_end=1.0,
        ai_labels=ai,
        ai_confidence=zeros,
        ai_uncertainty=zeros,
        doc_labels=doc,
        doc_confidence=zeros,
        doc_uncertainty=zeros,
        priority_key=0.0,
        observations=observations or {},
    )


# ── Resolution ─────────────────────────────────────────────────────────────────


def test_registered_name_resolves():
    assert resolve_bundle("team").name == "team"


def test_unknown_name_names_the_alternatives():
    with pytest.raises(ValueError, match="unknown KPI bundle"):
        resolve_bundle("nope")


def test_file_reference_resolves_a_bundle_outside_the_library():
    bundle = resolve_bundle(f"{_EXAMPLE}:CostBundle")
    assert isinstance(bundle, KPIBundle)
    assert bundle.name == "cost"
    # resolving does NOT register: a one-off reference must not mutate global state
    assert "cost" not in KPI_BUNDLES


def test_relative_file_reference_resolves(monkeypatch):
    """The documented form is a path relative to where the study is run from."""
    monkeypatch.chdir(_REPO_ROOT)
    assert resolve_bundle("configs/examples/custom_metrics.py:CostBundle").name == "cost"


def test_reference_accepts_class_instance_and_factory(tmp_path):
    plugin = tmp_path / "p.py"
    plugin.write_text(
        "class B:\n"
        "    name = 'b'\n"
        "    def compute(self, records):\n"
        "        return len(records)\n"
        "INSTANCE = B()\n"
        "def factory():\n"
        "    return B()\n",
        encoding="utf-8",
    )
    for attr in ("B", "INSTANCE", "factory"):
        assert resolve_bundle(f"{plugin}:{attr}").name == "b"


def test_missing_file_attribute_and_malformed_references_fail_clearly(tmp_path):
    with pytest.raises(ValueError, match="no such file"):
        resolve_bundle(f"{tmp_path / 'absent.py'}:B")
    with pytest.raises(ValueError, match="has no attribute"):
        resolve_bundle(f"{_EXAMPLE}:NotThere")
    with pytest.raises(ValueError, match="malformed"):
        resolve_bundle("bare:")
    with pytest.raises(ValueError, match="cannot import module"):
        resolve_bundle("no_such_module_xyz:B")


def test_non_bundle_attribute_is_rejected(tmp_path):
    plugin = tmp_path / "bad.py"
    plugin.write_text("class NotABundle:\n    pass\nVALUE = 3\n", encoding="utf-8")
    with pytest.raises(TypeError, match="not a KPIBundle"):
        resolve_bundle(f"{plugin}:NotABundle")
    with pytest.raises(TypeError, match="neither a class"):
        resolve_bundle(f"{plugin}:VALUE")


# ── Registration ───────────────────────────────────────────────────────────────


class _Counter:
    name = "counter"

    def compute(self, records: Sequence[PatientRecord]) -> int:
        return len(records)


def test_register_then_use_by_bare_name(clean_registry):
    register_bundle(_Counter())
    report = compute_report([_rec((1,), (1,), (1,))], ["counter"])
    assert report["counter"] == 1


def test_register_refuses_to_shadow_silently(clean_registry):
    register_bundle(_Counter())
    with pytest.raises(ValueError, match="already registered"):
        register_bundle(_Counter())
    register_bundle(_Counter(), replace=True)  # explicit override is fine


def test_register_rejects_a_non_bundle(clean_registry):
    with pytest.raises(TypeError, match="not a KPIBundle"):
        register_bundle(object())  # type: ignore[arg-type]


# ── End-to-end through compute_report ──────────────────────────────────────────


def test_custom_bundle_runs_through_compute_report():
    records = [
        _rec((1, 0), (0, 0), (0, 0)),  # AI and team each miss one positive → 1 FN
        _rec((0, 0), (1, 0), (1, 0)),  # and each raise one false alarm → 1 FP
    ]
    report = compute_report(records, ["system", f"{_EXAMPLE}:CostBundle"])
    assert set(report) == {"system", "cost"}  # keyed by the bundle's own name
    assert report["cost"]["ai_total"] == 10.0 * 1 + 1.0 * 1
    assert report["cost"]["ai_per_patient"] == 5.5


def test_bundle_reading_observations_answers_a_previously_unaskable_question():
    """`UncertaintySplitBundle` compares epistemic against aleatoric uncertainty on the
    AI's errors. Before the observation bag existed the decomposition never reached a
    metric, so this could not be computed at all — let alone from outside the library."""
    cache = InferenceCache(
        member_probs=np.array(
            [
                # members agree and match the truth → no error, low epistemic
                [[0.95, 0.02, 0.02, 0.1, 0.9], [0.93, 0.04, 0.03, 0.1, 0.88]],
                # members maximally disagree → the mean lands on the fence and errs
                [[0.9, 0.1, 0.9, 0.2, 0.8], [0.1, 0.9, 0.1, 0.8, 0.2]],
            ],
            dtype=np.float32,
        ),
        ground_truth=np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 0]], dtype=np.int8),
        inference_seconds=np.array([5.0, 5.0]),
        image_ids=("ok.png", "hard.png"),
        dataset_indices=np.array([0, 1]),
        weights=("w1", "w2"),
        pathologies=_SCHEMA.labels,
        dataset="mock",
        batch_size=1,
    )
    gen = CachedPatientGenerator(
        cache, _SCHEMA, arrival_rate=1.0, n_patients=8, rng=np.random.default_rng(0)
    )
    res = SimulationEngine(
        patient_generator=gen,
        ai=CachedEnsembleAI(cache, _SCHEMA, uq=EnsembleUQ(measure="mutual_information")),
        doctor=RandomMockDoctor(
            schema=_SCHEMA, rng=np.random.default_rng(2),
            service_time_mu=0.0, service_time_sigma=0.01,
        ),
        sort_strategy=RandomSort(rng=np.random.default_rng(3)),
    ).run()

    split = compute_report(
        res.collector.records, [f"{_EXAMPLE}:UncertaintySplitBundle"]
    )["uncertainty_split"]
    assert split.n == 8
    # The disagreeing image is both the erroneous one and the epistemically uncertain
    # one, so the separation must come out positive — the premise of uncertainty triage.
    assert split.epistemic_on_error is not None
    assert split.epistemic_on_correct is not None
    assert split.epistemic_separation > 0


def test_bundle_reports_none_when_the_observation_is_absent():
    """A mock AI publishes no decomposition; the bundle must say so, not invent a 0."""
    split = compute_report(
        [_rec((1, 0), (1, 0), (1, 0))], [f"{_EXAMPLE}:UncertaintySplitBundle"]
    )["uncertainty_split"]
    assert split.n == 1
    assert split.epistemic_on_error is None
    assert split.epistemic_separation is None
