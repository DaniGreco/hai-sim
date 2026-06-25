"""Build runtime objects from a `ScenarioConfig`.

Kept separate from the config module so the config layer stays pure data.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from hai_sim.config.scenario import (
    AIModelConfig,
    DoctorConfig,
    PatientGenConfig,
    ScenarioConfig,
    SortConfig,
)
from hai_sim.core.protocols import DiagnosticActor, SortStrategy
from hai_sim.core.types import LabelSchema, Patient
from hai_sim.doctors.mock import RandomMockDoctor
from hai_sim.doctors.simple import FixedAccuracyDoctor
from hai_sim.models.mock import RandomMockAI
from hai_sim.sim.patient_generator import PoissonPatientGenerator
from hai_sim.sort.random import RandomSort
from hai_sim.sort.uncertainty import UncertaintyDescSort


def build_label_schema(name: str) -> LabelSchema:
    if name == "default_v0":
        return LabelSchema.default_v0()
    if name == "chexpert":
        return LabelSchema.chexpert()
    if name == "xrv":
        return LabelSchema.xrv()
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
    if cfg.kind == "torchxrv":
        from hai_sim.models.torchxrv import TorchXRayVisionAI
        from hai_sim.uq.methods import EnsembleUQ, EntropyUQ

        uq = EnsembleUQ(measure=cfg.ensemble_measure) if cfg.uq == "ensemble" else EntropyUQ()
        return TorchXRayVisionAI(
            schema=schema,
            weights=cfg.weights,
            uq=uq,
            decision_threshold=cfg.decision_threshold,
            service_time=cfg.service_time,
            device=cfg.device,
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
    if cfg.kind == "fixed_accuracy":
        return FixedAccuracyDoctor(
            schema=schema,
            rng=rng,
            sensitivity=cfg.sensitivity,
            specificity=cfg.specificity,
            service_time_mu=cfg.service_time_mu,
            service_time_sigma=cfg.service_time_sigma,
        )
    raise ValueError(f"unknown doctor.kind: {cfg.kind}")


def build_sort(cfg: SortConfig, rng: np.random.Generator) -> SortStrategy:
    if cfg.kind == "random":
        return RandomSort(rng=rng)
    if cfg.kind == "uncertainty_desc":
        return UncertaintyDescSort(aggregator=cfg.aggregator, weights=cfg.weights)
    raise ValueError(f"unknown sort.kind: {cfg.kind}")


def _nih_numeric_age_csv(csvpath: str) -> str:
    """Return a csv whose 'Patient Age' is numeric, cleaning the NIH sample if needed.

    The Kaggle NIH *sample* ships ``Patient Age`` as strings like ``060Y``, but
    xrv's ``NIH_Dataset`` does ``Patient Age * 1.0``. Extract the leading digits
    into a cached sibling csv and point the loader there. The full
    ``Data_Entry_2017.csv`` is already numeric and passes through untouched.
    """
    from pathlib import Path

    import pandas as pd

    src = Path(csvpath)
    df = pd.read_csv(src)
    age = df.get("Patient Age")
    if age is None or pd.api.types.is_numeric_dtype(age):
        return csvpath
    cleaned = src.with_suffix(".xrvclean.csv")
    if not cleaned.exists():
        df = df.copy()
        df["Patient Age"] = (
            df["Patient Age"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
        )
        df.to_csv(cleaned, index=False)
    return str(cleaned)


def build_xrv_dataset(cfg: PatientGenConfig) -> Any:
    """Construct a label-aligned TorchXRayVision dataset for the patient stream."""
    import torchvision
    import torchxrayvision as xrv

    if cfg.imgpath is None:
        raise ValueError("patient_generator.imgpath is required for kind=xrv_dataset")
    transform = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
    )
    common = {"imgpath": cfg.imgpath, "transform": transform}
    if cfg.dataset == "nih":
        kwargs = dict(common)
        if cfg.csvpath is not None:
            kwargs["csvpath"] = _nih_numeric_age_csv(cfg.csvpath)
        ds = xrv.datasets.NIH_Dataset(**kwargs)
    elif cfg.dataset == "rsna":
        ds = xrv.datasets.RSNA_Pneumonia_Dataset(**common)
    elif cfg.dataset == "chexpert":
        if cfg.csvpath is None:
            raise ValueError("chexpert dataset requires patient_generator.csvpath")
        ds = xrv.datasets.CheX_Dataset(csvpath=cfg.csvpath, **common)
    else:
        raise ValueError(f"unknown xrv dataset: {cfg.dataset}")

    # Align dataset labels/order to the canonical 18-pathology model space.
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, ds, silent=True)
    return ds


def build_patient_generator(
    cfg: PatientGenConfig,
    schema: LabelSchema,
    rng: np.random.Generator,
) -> Iterable[tuple[float, Patient]]:
    if cfg.kind == "poisson":
        return PoissonPatientGenerator(
            schema=schema,
            arrival_rate=cfg.arrival_rate,
            n_patients=cfg.n_patients,
            prevalence=cfg.prevalence,
            image_shape=cfg.image_shape,
            rng=rng,
        )
    if cfg.kind == "xrv_dataset":
        from hai_sim.sim.dataset_generator import XRVDatasetPatientGenerator

        ds = build_xrv_dataset(cfg)
        return XRVDatasetPatientGenerator(
            dataset=ds,
            schema=schema,
            arrival_rate=cfg.arrival_rate,
            n_patients=cfg.n_patients,
            rng=rng,
            sample_mode=cfg.sample_mode,
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
