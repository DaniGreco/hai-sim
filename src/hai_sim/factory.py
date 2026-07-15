"""Build runtime objects from a `ScenarioConfig`.

Kept separate from the config module so the config layer stays pure data.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from hai_sim.config.scenario import (
    AIModelConfig,
    CapabilityConfig,
    PatientGenConfig,
    PersonaConfig,
    ScenarioConfig,
    SeverityConfig,
    SortConfig,
    TrustConfig,
    ValidationScenarioConfig,
)
from hai_sim.core.protocols import DiagnosticActor, IntegrationPolicy, SeverityModel, SortStrategy
from hai_sim.core.types import LabelSchema, Patient
from hai_sim.data.mock import MockDataset
from hai_sim.data.xrv import XRVDataset
from hai_sim.doctors.assisted import (
    AssistedDoctor,
    ConfidenceGatedPoolPolicy,
    LinearPoolPolicy,
    LogOddsFusionPolicy,
)
from hai_sim.doctors.mock import RandomMockDoctor
from hai_sim.doctors.simple import FixedAccuracyDoctor
from hai_sim.labels import ProjectedActor, build_projection, project_generator
from hai_sim.models.mock import RandomMockAI
from hai_sim.personas import resolve_persona
from hai_sim.severity.models import (
    ActivationSeverity,
    ConstantSeverity,
    EmbeddingSeverity,
    ImageQualitySeverity,
)
from hai_sim.sim.dataset_generator import DatasetPatientGenerator
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


def build_ai(
    cfg: AIModelConfig,
    schema: LabelSchema,
    rng: np.random.Generator,
    emit_embedding: bool = False,
) -> DiagnosticActor:
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
            emit_embedding=emit_embedding,
        )
    if cfg.kind == "cached":
        from hai_sim.analysis.inference_cache import InferenceCache
        from hai_sim.models.cached import CachedEnsembleAI
        from hai_sim.uq.methods import EnsembleUQ, EntropyUQ

        if cfg.cache_stem is None:
            raise ValueError("ai.kind=cached requires ai.cache_stem")
        cache = InferenceCache.load(cfg.cache_stem)
        uq = EnsembleUQ(measure=cfg.ensemble_measure) if cfg.uq == "ensemble" else EntropyUQ()
        return CachedEnsembleAI(
            cache=cache,
            schema=schema,
            uq=uq,
            decision_threshold=cfg.decision_threshold,
            service_time=cfg.service_time,
        )
    raise ValueError(f"unknown ai.kind: {cfg.kind}")


def build_capability(
    cfg: CapabilityConfig, schema: LabelSchema, rng: np.random.Generator
) -> DiagnosticActor:
    """Build the persona's base reader (its own, AI-unaided diagnostic ability)."""
    if cfg.source == "reader":
        raise ValueError(
            "capability source 'reader' (calibration on a real radiologist's per-finding "
            "reads) needs a dataset advertising per-reader reads; not wired yet — use "
            "source: fixed for now."
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
        )
    raise ValueError(f"unknown capability.kind: {cfg.kind}")


def build_trust_policy(cfg: TrustConfig) -> IntegrationPolicy:
    """Build the swappable AI-integration policy that models the persona's trust."""
    if cfg.kind == "linear_pool":
        return LinearPoolPolicy(weight_ai=cfg.weight_ai)
    if cfg.kind == "confidence_gated":
        return ConfidenceGatedPoolPolicy(
            weight_ai=cfg.weight_ai, confidence_threshold=cfg.confidence_threshold
        )
    if cfg.kind == "log_odds_fusion":
        return LogOddsFusionPolicy(
            beta=cfg.beta, weight_doctor=cfg.weight_doctor, prior=cfg.prior
        )
    raise ValueError(f"unknown trust.kind: {cfg.kind}")


def build_persona(
    persona: str | PersonaConfig, schema: LabelSchema, rng: np.random.Generator
) -> DiagnosticActor:
    """Assemble a doctor persona = capability x visibility x trust into an actor.

    A solo persona (``sees_ai=False``) is just its base reader; an assisted persona
    wraps that base in an `AssistedDoctor` with the trust policy. ``persona`` may be a
    library name or an inline `PersonaConfig`.
    """
    cfg = resolve_persona(persona)
    base = build_capability(cfg.capability, schema, rng)
    if not cfg.sees_ai:
        return base
    trust = cfg.trust if cfg.trust is not None else TrustConfig()
    policy = build_trust_policy(trust)
    return AssistedDoctor(base=base, policy=policy, name=f"Persona[{cfg.name or 'inline'}]")


def build_sort(cfg: SortConfig, rng: np.random.Generator) -> SortStrategy:
    if cfg.kind == "random":
        return RandomSort(rng=rng)
    if cfg.kind == "uncertainty_desc":
        return UncertaintyDescSort(aggregator=cfg.aggregator, weights=cfg.weights)
    raise ValueError(f"unknown sort.kind: {cfg.kind}")


def _severity_needs_embedding(cfg: SeverityConfig | None) -> bool:
    """Only the embedding-based severity source reads the model's feature vector."""
    return cfg is not None and cfg.kind == "embedding"


def build_severity(cfg: SeverityConfig | None, schema: LabelSchema) -> SeverityModel | None:
    if cfg is None:
        return None
    if cfg.kind == "constant":
        return ConstantSeverity(value=cfg.value)
    if cfg.kind == "activation":
        if cfg.criticality is not None and len(cfg.criticality) != schema.n:
            raise ValueError(
                f"severity.criticality has {len(cfg.criticality)} weights, "
                f"expected {schema.n} (one per label)"
            )
        return ActivationSeverity(
            source=cfg.source,
            criticality=cfg.criticality,
            aggregator=cfg.aggregator,
            mode=cfg.mode,
        )
    if cfg.kind == "image_quality":
        return ImageQualitySeverity(metric=cfg.quality_metric)
    if cfg.kind == "embedding":
        return EmbeddingSeverity(reduce=cfg.embedding_reduce)
    raise ValueError(f"unknown severity.kind: {cfg.kind}")


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
        kwargs["views"] = list(cfg.views)
        kwargs["unique_patients"] = cfg.unique_patients
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
        if cfg.gt_source != "native":
            raise ValueError(
                f"kind=poisson (synthetic) only provides gt_source='native', "
                f"got {cfg.gt_source!r}"
            )
        return PoissonPatientGenerator(
            schema=schema,
            arrival_rate=cfg.arrival_rate,
            n_patients=cfg.n_patients,
            prevalence=cfg.prevalence,
            image_shape=cfg.image_shape,
            rng=rng,
        )
    if cfg.kind == "mock_dataset":
        mock = MockDataset(
            schema=schema,
            n_examples=cfg.n_examples,
            prevalence=cfg.prevalence,
            rng=rng,
            image_shape=cfg.image_shape,
        )
        return DatasetPatientGenerator(
            mock,
            arrival_rate=cfg.arrival_rate,
            n_patients=cfg.n_patients,
            rng=rng,
            sample_mode=cfg.sample_mode,
            gt_source=cfg.gt_source,
        )
    if cfg.kind == "xrv_dataset":
        xrv = XRVDataset(build_xrv_dataset(cfg), schema, name=cfg.dataset)
        return DatasetPatientGenerator(
            xrv,
            arrival_rate=cfg.arrival_rate,
            n_patients=cfg.n_patients,
            rng=rng,
            sample_mode=cfg.sample_mode,
            gt_source=cfg.gt_source,
        )
    if cfg.kind == "inference_cache":
        from hai_sim.analysis.inference_cache import InferenceCache
        from hai_sim.sim.cache_generator import CachedPatientGenerator

        if cfg.cache_stem is None:
            raise ValueError("patient_generator.kind=inference_cache requires cache_stem")
        if cfg.gt_source != "native":
            raise ValueError(
                f"inference_cache provides only gt_source='native', got {cfg.gt_source!r}"
            )
        cache = InferenceCache.load(cfg.cache_stem)
        return CachedPatientGenerator(
            cache=cache,
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
    """Returns (schema, generator, ai, doctor, sort, severity, n_ai_workers, n_doctor_workers).

    ``schema`` is the ACTIVE schema the run scores on: the base schema, or its
    projection when a ``label_view`` is set. The AI is built in its native (base)
    space and wrapped so its output — like the projected ground truth — lands in
    the active space; the doctor, sort and severity are built on the active schema.
    """
    rngs = build_rngs(scenario.seed)
    base_schema = build_label_schema(scenario.label_schema)
    view = scenario.label_view
    projection = build_projection(
        base_schema,
        mode=view.mode if view else "full",
        targets=view.targets if view else None,
        combine=view.combine if view else "noisy_or",
    )
    active_schema = projection.active

    base_generator = build_patient_generator(
        scenario.patient_generator, base_schema, rngs["patient"]
    )
    base_ai = build_ai(
        scenario.ai,
        base_schema,
        rngs["ai"],
        emit_embedding=_severity_needs_embedding(scenario.severity),
    )
    generator: Iterable[tuple[float, Patient]] = base_generator
    ai: DiagnosticActor = base_ai
    if not projection.is_identity:
        generator = project_generator(base_generator, projection)
        ai = ProjectedActor(base_ai, projection)

    doctor = build_persona(scenario.persona, active_schema, rngs["doctor"])
    sort = build_sort(scenario.sort, rngs["sort"])
    severity = build_severity(scenario.severity, active_schema)
    return (
        active_schema,
        generator,
        ai,
        doctor,
        sort,
        severity,
        scenario.n_ai_workers,
        scenario.n_doctor_workers,
    )


def build_validation(scenario: ValidationScenarioConfig):
    """Build the localization validator from a `ValidationScenarioConfig`.

    Returns ``(schema, dataset, validator)``. Reuses `build_ai`, so the model
    validated here is constructed identically to the one a simulation would run.
    """
    from hai_sim.analysis.localization.chestxdet import ChestXDetDataset
    from hai_sim.analysis.localization.metrics import Dice, EnergyInMask, IoU, PointingGame
    from hai_sim.analysis.localization.saliency import GradCAMSaliency
    from hai_sim.analysis.localization.validator import LocalizationValidator

    rngs = build_rngs(scenario.seed)
    schema = build_label_schema(scenario.label_schema)
    ai = build_ai(scenario.ai, schema, rngs["ai"])

    d = scenario.dataset
    if d.kind != "chestxdet":
        raise ValueError(f"unknown validation dataset kind: {d.kind}")
    dataset = ChestXDetDataset(d.annotation_path, d.image_dir, schema=schema, resize=d.resize)

    saliency = None
    if scenario.saliency == "grad_cam":
        from hai_sim.models.torchxrv import TorchXRayVisionAI

        if not isinstance(ai, TorchXRayVisionAI):
            raise ValueError(
                "saliency: grad_cam needs ai.kind=torchxrv (a model with conv features); "
                "use saliency: none for label-only validation"
            )
        saliency = GradCAMSaliency(ai)

    m = scenario.metrics
    metrics: list[Any] = []
    if m.pointing_game:
        metrics.append(PointingGame(tolerance=m.pointing_tolerance))
    if m.energy_in_mask:
        metrics.append(EnergyInMask())
    metrics.extend(IoU(t) for t in m.iou_thresholds)
    metrics.extend(Dice(t) for t in m.dice_thresholds)

    validator = LocalizationValidator(ai, dataset, saliency, metrics or None)
    return schema, dataset, validator
