"""Build runtime objects from a `ScenarioConfig`.

Kept separate from the config module so the config layer stays pure data.
"""

from __future__ import annotations

import numpy as np

from hai_sim.config.scenario import (
    AIModelConfig,
    DoctorConfig,
    PatientGenConfig,
    ScenarioConfig,
    SortConfig,
)
from hai_sim.core.protocols import DiagnosticActor, SortStrategy
from hai_sim.core.types import LabelSchema
from hai_sim.doctors.mock import RandomMockDoctor
from hai_sim.models.mock import RandomMockAI
from hai_sim.sim.patient_generator import PoissonPatientGenerator
from hai_sim.sort.random import RandomSort
from hai_sim.sort.uncertainty import UncertaintyDescSort


def build_label_schema(name: str) -> LabelSchema:
    if name == "default_v0":
        return LabelSchema.default_v0()
    if name == "chexpert":
        return LabelSchema.chexpert()
    raise ValueError(f"unknown label_schema: {name}")


def build_ai(cfg: AIModelConfig, schema: LabelSchema, rng: np.random.Generator) -> DiagnosticActor:
    if cfg.kind == "random_mock":
        return RandomMockAI(
            schema=schema,
            rng=rng,
            service_time=cfg.service_time,
            alpha_conf=cfg.alpha_conf,
            beta_conf=cfg.beta_conf,
            alpha_unc=cfg.alpha_unc,
            beta_unc=cfg.beta_unc,
        )
    raise ValueError(f"unknown ai.kind: {cfg.kind}")


def build_doctor(cfg: DoctorConfig, schema: LabelSchema, rng: np.random.Generator) -> DiagnosticActor:
    if cfg.kind == "random_mock":
        return RandomMockDoctor(
            schema=schema,
            rng=rng,
            service_time_mu=cfg.service_time_mu,
            service_time_sigma=cfg.service_time_sigma,
            alpha_conf=cfg.alpha_conf,
            beta_conf=cfg.beta_conf,
            alpha_unc=cfg.alpha_unc,
            beta_unc=cfg.beta_unc,
            use_ai_hint=cfg.use_ai_hint,
            assisted_policy=cfg.assisted_policy,
            ai_hint_uncertainty_threshold=cfg.ai_hint_uncertainty_threshold,
        )
    raise ValueError(f"unknown doctor.kind: {cfg.kind}")


def build_sort(cfg: SortConfig, rng: np.random.Generator) -> SortStrategy:
    if cfg.kind == "random":
        return RandomSort(rng=rng)
    if cfg.kind == "uncertainty_desc":
        return UncertaintyDescSort(aggregator=cfg.aggregator, weights=cfg.weights)
    raise ValueError(f"unknown sort.kind: {cfg.kind}")


def build_patient_generator(
    cfg: PatientGenConfig,
    schema: LabelSchema,
    rng: np.random.Generator,
) -> PoissonPatientGenerator:
    if cfg.kind == "poisson":
        return PoissonPatientGenerator(
            schema=schema,
            arrival_rate=cfg.arrival_rate,
            n_patients=cfg.n_patients,
            prevalence=cfg.prevalence,
            image_shape=cfg.image_shape,
            rng=rng,
        )
    raise ValueError(f"unknown patient_generator.kind: {cfg.kind}")


def build_rngs(seed: int) -> dict[str, np.random.Generator]:
    """Independent substreams so swapping one module doesn't perturb others."""
    seed_seq = np.random.SeedSequence(seed)
    children = seed_seq.spawn(5)
    return {
        "patient": np.random.default_rng(children[0]),
        "ai": np.random.default_rng(children[1]),
        "doctor": np.random.default_rng(children[2]),
        "sort": np.random.default_rng(children[3]),
        "_reserved": np.random.default_rng(children[4]),
    }


def build_from_scenario(scenario: ScenarioConfig):
    """Returns (schema, generator, ai, doctor, sort, n_ai_workers, n_doctor_workers)."""
    rngs = build_rngs(scenario.seed)
    schema = build_label_schema(scenario.label_schema)
    generator = build_patient_generator(scenario.patient_generator, schema, rngs["patient"])
    ai = build_ai(scenario.ai, schema, rngs["ai"])
    doctor = build_doctor(scenario.doctor, schema, rngs["doctor"])
    sort = build_sort(scenario.sort, rngs["sort"])
    return schema, generator, ai, doctor, sort, scenario.n_ai_workers, scenario.n_doctor_workers
