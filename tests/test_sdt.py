"""The signal-detection observer: core math, the SDT doctor, the synthetic AI, wiring."""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim import sdt
from hai_sim.config.scenario import (
    AIModelConfig,
    CapabilityConfig,
    PatientGenConfig,
    PersonaConfig,
    ScenarioConfig,
    SortConfig,
    TrustConfig,
)
from hai_sim.core.types import LabelSchema, PatientContext
from hai_sim.doctors.sdt import SDTDoctor
from hai_sim.factory import build_ai, build_capability, build_from_scenario, build_label_schema
from hai_sim.metrics.kpi import compute_kpis
from hai_sim.models.synthetic import FixedAccuracyAI
from hai_sim.sim.engine import SimulationEngine
from hai_sim.sim.patient_generator import PoissonPatientGenerator


def _ctx(gt: tuple[int, ...]) -> PatientContext:
    return PatientContext(patient_id=0, arrival_time=0.0, ground_truth=gt)


# ── Core SDT math ────────────────────────────────────────────────────────────


def test_dprime_criterion_inverts_operating_point() -> None:
    # sens = spec = 0.95 → d' = 2·z(0.95), k = z(0.95); symmetric observer.
    d, k = sdt.dprime_criterion_from_accuracy(np.array([0.95]), np.array([0.95]))
    assert d[0] == pytest.approx(2 * 1.6448, abs=1e-3)
    assert k[0] == pytest.approx(1.6448, abs=1e-3)


def test_default_criterion_makes_call_and_belief_agree() -> None:
    # At the default criterion, `call` and `belief >= 0.5` must coincide by construction,
    # for any prior — so a reader's decision never contradicts its own stated confidence.
    rng = np.random.default_rng(0)
    for prior in (0.2, 0.5, 0.8):
        d = np.array([2.5])
        k = sdt.default_criterion(d, np.array([prior]))
        for y in (np.array([0]), np.array([1])):
            for _ in range(200):
                call, belief = sdt.observe(
                    y, d_prime=d, criterion=k, prior=np.array([prior]), rng=rng
                )
                assert int(call[0]) == int(belief[0] >= 0.5)


def test_observe_skips_unannotated() -> None:
    d = np.array([3.0, 3.0, 3.0])
    k = sdt.default_criterion(d, np.full(3, 0.5))
    call, belief = sdt.observe(
        np.array([1, -1, 0]), d_prime=d, criterion=k, prior=np.full(3, 0.5),
        rng=np.random.default_rng(0),
    )
    assert int(call[1]) == 0
    assert belief[1] == pytest.approx(0.5)  # pinned at the prior


# ── SDTDoctor ────────────────────────────────────────────────────────────────


def test_sdt_doctor_hits_target_sensitivity_and_specificity() -> None:
    schema = LabelSchema.single()
    doc = SDTDoctor(schema, rng=np.random.default_rng(1), sensitivity=0.9, specificity=0.9)
    hits = sum(
        doc.diagnose([np.zeros(1)], context=_ctx((1,))).diagnosis.labels[0]
        for _ in range(4000)
    )
    correct_rejections = sum(
        1 - doc.diagnose([np.zeros(1)], context=_ctx((0,))).diagnosis.labels[0]
        for _ in range(4000)
    )
    assert 0.87 < hits / 4000 < 0.93
    assert 0.87 < correct_rejections / 4000 < 0.93


def test_sdt_doctor_confidence_varies_case_by_case() -> None:
    """Unlike FixedAccuracyDoctor, the belief is not a constant — that is the point."""
    schema = LabelSchema.single()
    doc = SDTDoctor(schema, rng=np.random.default_rng(2), d_prime=2.5)
    beliefs = [
        doc.diagnose([np.zeros(1)], context=_ctx((1,))).diagnosis.confidence[0]
        for _ in range(500)
    ]
    assert np.std(beliefs) > 0.05


def test_sdt_doctor_higher_dprime_is_more_accurate() -> None:
    schema = LabelSchema.single()

    def accuracy(d_prime: float) -> float:
        doc = SDTDoctor(schema, rng=np.random.default_rng(3), d_prime=d_prime)
        n = 3000
        correct = 0
        for i in range(n):
            g = i % 2
            out = doc.diagnose([np.zeros(1)], context=_ctx((g,)))
            correct += int(out.diagnosis.labels[0] == g)
        return correct / n

    assert accuracy(3.5) > accuracy(1.5)


def test_sdt_doctor_requires_ground_truth() -> None:
    doc = SDTDoctor(LabelSchema.single())
    with pytest.raises(ValueError, match="ground_truth"):
        doc.diagnose([np.zeros(1)], context=PatientContext(patient_id=0, arrival_time=0.0))


def test_sdt_doctor_is_deterministic_under_seed() -> None:
    schema = LabelSchema.single()
    seqs = []
    for _ in range(2):
        doc = SDTDoctor(schema, rng=np.random.default_rng(7), d_prime=2.0)
        seqs.append(
            tuple(
                doc.diagnose([np.zeros(1)], context=_ctx((i % 2,))).diagnosis.labels[0]
                for i in range(50)
            )
        )
    assert seqs[0] == seqs[1]


def test_sdt_doctor_difficulty_lowers_discriminability() -> None:
    """A hard case (sig.difficulty=1) should cost accuracy when difficulty_slope>0."""
    schema = LabelSchema.single()

    def accuracy(difficulty: float) -> float:
        doc = SDTDoctor(schema, rng=np.random.default_rng(5), d_prime=3.0, difficulty_slope=2.5)
        n = 3000
        correct = 0
        for i in range(n):
            g = i % 2
            ctx = PatientContext(
                patient_id=i, arrival_time=0.0, ground_truth=(g,),
                signals={"difficulty": difficulty},
            )
            correct += int(doc.diagnose([np.zeros(1)], context=ctx).diagnosis.labels[0] == g)
        return correct / n

    assert accuracy(0.0) > accuracy(1.0) + 0.05


# ── FixedAccuracyAI (synthetic reader over embedded truth) ────────────────────


def test_synthetic_ai_reads_embedded_truth() -> None:
    schema = LabelSchema(labels=("a", "b", "c"))
    ai = FixedAccuracyAI(schema, rng=np.random.default_rng(0), sensitivity=0.999, specificity=0.999)
    # Near-perfect d' → reproduces the label vector the image carries.
    for gt in [(1, 0, 1), (0, 0, 0), (1, 1, 1)]:
        out = ai.diagnose([np.asarray(gt, dtype=float)])
        assert out.diagnosis.labels == gt


def test_synthetic_ai_higher_dprime_is_more_accurate() -> None:
    schema = LabelSchema.single()

    def accuracy(d_prime: float) -> float:
        ai = FixedAccuracyAI(schema, rng=np.random.default_rng(4), d_prime=d_prime)
        n = 3000
        correct = 0
        for i in range(n):
            g = i % 2
            out = ai.diagnose([np.asarray((g,), dtype=float)])
            correct += int(out.diagnosis.labels[0] == g)
        return correct / n

    assert accuracy(4.0) > accuracy(1.5)


def test_synthetic_ai_rejects_wrong_shape() -> None:
    ai = FixedAccuracyAI(LabelSchema.single(), d_prime=2.0)
    with pytest.raises(ValueError, match="embedded-truth"):
        ai.diagnose([np.asarray((1, 0), dtype=float)])


def test_embed_truth_generator_writes_labels_into_image() -> None:
    schema = LabelSchema.single()
    gen = PoissonPatientGenerator(
        schema=schema, arrival_rate=1.0, n_patients=20, prevalence=0.5,
        rng=np.random.default_rng(0), embed_truth=True,
    )
    for _, patient in gen:
        assert tuple(np.rint(patient.images[0]).astype(int)) == patient.ground_truth


# ── Factory wiring ───────────────────────────────────────────────────────────


def test_factory_builds_single_schema() -> None:
    assert build_label_schema("single").n == 1


def test_factory_builds_synthetic_ai_and_sdt_doctor() -> None:
    schema = LabelSchema.single()
    rng = np.random.default_rng(0)
    ai = build_ai(AIModelConfig(kind="fixed_accuracy", d_prime=3.0), schema, rng)
    assert isinstance(ai, FixedAccuracyAI)
    doc = build_capability(CapabilityConfig(kind="sdt", d_prime=2.5), schema, rng)
    assert isinstance(doc, SDTDoctor)


# ── End to end: a good AI + an SDT persona runs and behaves sensibly ──────────


def _scenario(weight_ai: float, doc_dprime: float, ai_dprime: float = 3.3) -> ScenarioConfig:
    return ScenarioConfig(
        name="sdt_e2e",
        seed=0,
        label_schema="single",
        patient_generator=PatientGenConfig(
            kind="poisson", arrival_rate=0.5, n_patients=800, prevalence=0.3, embed_truth=True
        ),
        ai=AIModelConfig(kind="fixed_accuracy", d_prime=ai_dprime, prior=0.3),
        persona=PersonaConfig(
            name="sdt_doc",
            capability=CapabilityConfig(kind="sdt", d_prime=doc_dprime, prior=0.3),
            sees_ai=True,
            trust=TrustConfig(kind="linear_pool", weight_ai=weight_ai),
        ),
        sort=SortConfig(kind="uncertainty_desc"),
    )


def _run(scenario: ScenarioConfig):
    built = build_from_scenario(scenario)
    result = SimulationEngine(
        patient_generator=built.generator,
        ai=built.ai,
        doctor=built.doctor,
        sort_strategy=built.sort,
        n_ai_workers=built.n_ai_workers,
        n_doctor_workers=built.n_doctor_workers,
    ).run()
    return compute_kpis(result.collector.records)


def test_end_to_end_good_ai_helps_a_weak_trusting_doctor() -> None:
    """A good AI, adopted, should lift a weak doctor's recall above going solo."""
    solo = _run(_scenario(weight_ai=0.0, doc_dprime=1.6))
    trusting = _run(_scenario(weight_ai=0.8, doc_dprime=1.6))
    # The AI (d'=3.3) is much stronger than the doctor (d'=1.6); leaning on it should
    # raise the team's recall over the unaided doctor, and open reliance opportunities.
    assert trusting.hybrid.recall_macro > solo.hybrid.recall_macro
    assert trusting.team.n_ai_reliance_opportunities > 0


def test_end_to_end_ai_accuracy_matches_the_dial() -> None:
    kpis = _run(_scenario(weight_ai=0.0, doc_dprime=2.5, ai_dprime=3.29))
    # d'=3.29 with a base-rate-calibrated criterion at prevalence 0.3 gives ~0.92
    # sensitivity and ~0.97 specificity — a strong, realistically asymmetric reader whose
    # prevalence-weighted accuracy lands near 0.95. Check both halves of the operating point.
    assert 0.85 < kpis.ai.recall_macro < 0.96
    assert 0.90 < kpis.ai.accuracy < 0.98
