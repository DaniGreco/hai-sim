"""From a YAML scenario to running objects — the one place every ``kind:`` is resolved.

This is the bridge that makes the playground a playground. Each ``build_*`` function
takes one validated config block and returns the object it names, so the set of values a
``kind:`` accepts is literally the set of branches below. **Adding a new AI model, a new
triage rule or a new severity signal means writing the class and adding one branch
here** — nothing else in the framework needs to learn about it.

Kept separate from `hai_sim.config.scenario` on purpose: that module stays pure data
(pydantic models with no imports of runtime code), so a config can be validated, dumped
and diffed without constructing anything or importing a deep-learning stack.

Heavy dependencies are imported *inside* the branch that needs them — torch and
torchxrayvision only when ``ai.kind: torchxrv`` is actually requested — so running the
mock stack never pays for a framework it does not use.

``build_from_scenario`` at the bottom assembles the whole run in the pipeline's own
order: patient stream, then AI, then triage, then doctor.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from hai_sim.config.scenario import (
    AIModelConfig,
    ArrivalConfig,
    CalibrationConfig,
    CapabilityConfig,
    DifficultyConfig,
    FatigueConfig,
    KPIGroupConfig,
    PatientGenConfig,
    PersonaConfig,
    ScenarioConfig,
    ServiceTimeConfig,
    SeverityConfig,
    SortConfig,
    TrustConfig,
    ValidationScenarioConfig,
)
from hai_sim.core.protocols import (
    ArrivalProcess,
    DiagnosticActor,
    DifficultyModel,
    FatigueCurve,
    IntegrationPolicy,
    SeverityModel,
    SortStrategy,
)
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
from hai_sim.metrics.slicing import Grouping, LabelGrouping, ValueGrouping
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
    if name == "single":
        return LabelSchema.single()
    if name == "chexpert":
        return LabelSchema.chexpert()
    if name == "xrv":
        return LabelSchema.xrv()
    raise ValueError(f"unknown label_schema: {name}")


def build_calibrator(cfg: CalibrationConfig | None) -> Any:
    """Build the post-hoc probability calibrator, or the identity when unconfigured."""
    from hai_sim.uq.calibration import (
        IdentityCalibrator,
        IsotonicCalibrator,
        PlattCalibrator,
        TemperatureScaling,
    )

    if cfg is None or cfg.kind == "identity":
        return IdentityCalibrator()
    if cfg.kind == "temperature":
        return TemperatureScaling(cfg.temperature)
    if cfg.kind == "platt":
        return PlattCalibrator(a=cfg.a, b=cfg.b)
    if cfg.kind == "isotonic":
        if not cfg.knots_x or len(cfg.knots_x) != len(cfg.knots_y):
            raise ValueError(
                "calibration.kind=isotonic needs knots_x and knots_y of equal, non-zero "
                "length; fit them with scripts/fit_calibration.py"
            )
        return IsotonicCalibrator(cfg.knots_x, cfg.knots_y)
    raise ValueError(f"unknown calibration.kind: {cfg.kind}")


def _build_uq(cfg: AIModelConfig) -> Any:
    """The run's UQ method, wrapped in its calibrator when the scenario asks for one.

    One place for all backends: every actor that reports uncertainty does so through a
    `UQMethod`, so calibration enters here rather than in four model wrappers.
    """
    from hai_sim.uq.calibration import CalibratedUQ
    from hai_sim.uq.methods import EnsembleUQ, EntropyUQ

    uq = EnsembleUQ(measure=cfg.ensemble_measure) if cfg.uq == "ensemble" else EntropyUQ()
    if cfg.calibration is None or cfg.calibration.kind == "identity":
        return uq
    return CalibratedUQ(uq, build_calibrator(cfg.calibration))


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

        return TorchXRayVisionAI(
            schema=schema,
            weights=cfg.weights,
            uq=_build_uq(cfg),
            decision_threshold=cfg.decision_threshold,
            service_time=cfg.service_time,
            device=cfg.device,
            emit_embedding=emit_embedding,
        )
    if cfg.kind == "medgemma":
        from hai_sim.models.medgemma import MedGemmaAI
        from hai_sim.models.readout import (
            GeneratedJsonReadout,
            IndependentQuestionReadout,
            LabelReadout,
            TeacherForcedLogprobReadout,
        )

        readout: LabelReadout
        if cfg.readout == "generated_json":
            readout = GeneratedJsonReadout()
        elif cfg.readout == "teacher_forced_logprob":
            readout = TeacherForcedLogprobReadout(both_polarities=cfg.both_polarities)
        else:
            readout = IndependentQuestionReadout()
        return MedGemmaAI(
            schema=schema,
            model_id=cfg.model_id,
            readout=readout,
            prompt_set=cfg.prompt_set,
            # One prompt yields one member, so member disagreement does not exist and
            # only entropy of the model's own probability is available; several do.
            uq=_build_uq(cfg),
            decision_threshold=cfg.decision_threshold,
            service_time=cfg.service_time,
            device=cfg.device,
            dtype=cfg.dtype,
        )
    if cfg.kind == "cached":
        from hai_sim.analysis.inference_cache import InferenceCache
        from hai_sim.models.cached import CachedEnsembleAI

        if cfg.cache_stem is None:
            raise ValueError("ai.kind=cached requires ai.cache_stem")
        cache = InferenceCache.load(cfg.cache_stem)
        return CachedEnsembleAI(
            cache=cache,
            schema=schema,
            uq=_build_uq(cfg),
            decision_threshold=cfg.decision_threshold,
            service_time=cfg.service_time,
        )
    if cfg.kind == "fixed_accuracy":
        from hai_sim.models.synthetic import FixedAccuracyAI

        return FixedAccuracyAI(
            schema=schema,
            rng=rng,
            d_prime=cfg.d_prime,
            criterion=cfg.criterion,
            sensitivity=cfg.sensitivity,
            specificity=cfg.specificity,
            prior=cfg.prior,
            service_time=cfg.service_time,
        )
    raise ValueError(f"unknown ai.kind: {cfg.kind}")


def build_service_time(
    cfg: ServiceTimeConfig | None, rng: np.random.Generator
) -> Any | None:
    """Build the reading-time model, composing the fatigue and scale wrappers."""
    if cfg is None:
        return None
    from hai_sim.fatigue.service import (
        ConstantServiceTime,
        FatigueModulatedServiceTime,
        LognormalServiceTime,
        ScaledServiceTime,
    )

    base: Any
    if cfg.kind == "constant":
        base = ConstantServiceTime(value=cfg.value)
    elif cfg.kind == "lognormal":
        base = LognormalServiceTime(mu=cfg.mu, sigma=cfg.sigma, rng=rng)
    else:
        raise ValueError(f"unknown service_time.kind: {cfg.kind}")
    # Fatigue first, then the constant scale, so `factor` multiplies the read the
    # reader would actually have taken rather than a hypothetical fresh one.
    if cfg.fatigue_slope > 0.0:
        base = FatigueModulatedServiceTime(base, slope=cfg.fatigue_slope)
    if cfg.factor != 1.0:
        base = ScaledServiceTime(base, factor=cfg.factor)
    return base


def build_difficulty(cfg: DifficultyConfig | None, seed: int = 0) -> DifficultyModel | None:
    """Build the per-case difficulty model. ``None`` → no difficulty signal at all."""
    if cfg is None:
        return None
    from hai_sim.difficulty.models import (
        ConstantDifficulty,
        ImageQualityDifficulty,
        PriorDifficulty,
        TableDifficulty,
        UncertaintyDifficulty,
    )

    if cfg.kind == "constant":
        return ConstantDifficulty(value=cfg.value)
    if cfg.kind == "prior":
        return PriorDifficulty(a=cfg.a, b=cfg.b, seed=seed)
    if cfg.kind == "uncertainty":
        return UncertaintyDifficulty(
            source=cfg.source, aggregator=cfg.aggregator, scale=cfg.scale
        )
    if cfg.kind == "image_quality":
        return ImageQualityDifficulty(metric=cfg.quality_metric, reference=cfg.reference)
    if cfg.kind == "table":
        if cfg.path is None:
            raise ValueError(
                "difficulty.kind=table requires difficulty.path — a CSV of "
                "example_id,difficulty. Build one from per-reader disagreement with "
                "hai_sim.difficulty.disagreement_table, or from any model you like."
            )
        return TableDifficulty.from_csv(
            cfg.path,
            id_column=cfg.id_column,
            value_column=cfg.value_column,
            default=cfg.default,
            strict=cfg.strict,
        )
    raise ValueError(f"unknown difficulty.kind: {cfg.kind}")


def _one_fatigue_curve(cfg: FatigueConfig, kind: str) -> FatigueCurve:
    from hai_sim.fatigue.curves import CircadianFatigue, LinearFatigue, VigilanceFatigue

    if kind == "linear":
        return LinearFatigue(per_unit=cfg.per_unit, basis=cfg.basis, cap=cfg.cap)
    if kind == "vigilance":
        return VigilanceFatigue(asymptote=cfg.asymptote, scale=cfg.scale, basis=cfg.basis)
    if kind == "circadian":
        return CircadianFatigue(
            amplitude=cfg.amplitude,
            mean=cfg.mean,
            period=cfg.period,
            acrophase=cfg.acrophase,
        )
    raise ValueError(f"unknown fatigue curve: {kind}")


def build_fatigue(cfg: FatigueConfig | None) -> FatigueCurve | None:
    """Build the fatigue curve. ``None`` (or ``kind: none``) → readers never tire."""
    if cfg is None or cfg.kind == "none":
        return None
    if cfg.kind == "composite":
        from hai_sim.fatigue.curves import CompositeFatigue

        if not cfg.parts:
            raise ValueError(
                "fatigue.kind=composite requires fatigue.parts, e.g. "
                "[vigilance, circadian] — the two clocks a night shift runs on"
            )
        return CompositeFatigue(
            [_one_fatigue_curve(cfg, part) for part in cfg.parts], combine=cfg.combine
        )
    return _one_fatigue_curve(cfg, cfg.kind)


def build_arrivals(
    cfg: ArrivalConfig | None, arrival_rate: float, rng: np.random.Generator
) -> ArrivalProcess:
    """Build the arrival process; unconfigured → homogeneous Poisson at ``arrival_rate``."""
    from hai_sim.arrivals.processes import (
        DeterministicArrival,
        NHPPArrival,
        PiecewiseConstantRate,
        PoissonArrival,
    )

    if cfg is None or cfg.kind == "poisson":
        return PoissonArrival(rate=arrival_rate, rng=rng)
    if cfg.kind == "deterministic":
        return DeterministicArrival(rate=arrival_rate)
    if cfg.kind == "nhpp":
        if not cfg.rates:
            raise ValueError(
                "arrivals.kind=nhpp requires arrivals.rates: one rate per slot. For the "
                "day-of-week stratification §4.9 calls necessary, that is 168 hourly "
                "rates with period 168."
            )
        rate = PiecewiseConstantRate(cfg.rates, period=cfg.period)
        return NHPPArrival(rate, rng=rng)
    raise ValueError(f"unknown arrivals.kind: {cfg.kind}")


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
            difficulty_slope=cfg.difficulty_slope,
            fatigue_slope=cfg.fatigue_slope,
            floor=cfg.floor,
            service_time_model=build_service_time(cfg.service_time, rng),
        )
    if cfg.kind == "sdt":
        from hai_sim.doctors.sdt import SDTDoctor

        return SDTDoctor(
            schema=schema,
            rng=rng,
            d_prime=cfg.d_prime,
            criterion=cfg.criterion,
            sensitivity=cfg.sensitivity,
            specificity=cfg.specificity,
            prior=cfg.prior,
            difficulty_slope=cfg.difficulty_slope,
            fatigue_slope=cfg.fatigue_slope,
            service_time_mu=cfg.service_time_mu,
            service_time_sigma=cfg.service_time_sigma,
            service_time_model=build_service_time(cfg.service_time, rng),
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


def build_kpi_groupings(
    cfgs: Sequence[KPIGroupConfig], schema: LabelSchema
) -> list[Grouping]:
    """Build the KPI conditioning axes; ``key: label`` needs the schema for its names."""
    groupings: list[Grouping] = []
    for cfg in cfgs:
        if cfg.key == "label":
            if cfg.edges:
                raise ValueError("kpi_groups 'label' takes no edges: it is categorical")
            groupings.append(LabelGrouping(label_names=schema.labels))
        else:
            groupings.append(
                ValueGrouping(
                    key=cfg.key, edges=cfg.edges, include_missing=cfg.include_missing
                )
            )
    return groupings


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
    # Resolution is a per-model property, not a dataset constant: the DenseNet was
    # trained at 224, MedGemma encodes at 896. Serving 224 to both would compare them on
    # pixels one of them never saw in training, so each backend gets its own resolution
    # and the scenario says which.
    transform = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(cfg.resize)]
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
    arrival_rng: np.random.Generator | None = None,
) -> Iterable[tuple[float, Patient]]:
    """Build the patient stream: *who* arrives (this function) driven by *when* they
    arrive (an `ArrivalProcess`, built here from ``cfg.arrivals``)."""
    arrivals = build_arrivals(cfg.arrivals, cfg.arrival_rate, arrival_rng or rng)
    if cfg.kind == "poisson":
        if cfg.gt_source != "native":
            raise ValueError(
                f"kind=poisson (synthetic) only provides gt_source='native', "
                f"got {cfg.gt_source!r}"
            )
        return PoissonPatientGenerator(
            schema=schema,
            arrivals=arrivals,
            n_patients=cfg.n_patients,
            prevalence=cfg.prevalence,
            image_shape=cfg.image_shape,
            rng=rng,
            embed_truth=cfg.embed_truth,
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
            arrivals=arrivals,
            n_patients=cfg.n_patients,
            rng=rng,
            sample_mode=cfg.sample_mode,
            gt_source=cfg.gt_source,
        )
    if cfg.kind == "xrv_dataset":
        xrv = XRVDataset(build_xrv_dataset(cfg), schema, name=cfg.dataset)
        return DatasetPatientGenerator(
            xrv,
            arrivals=arrivals,
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
            arrivals=arrivals,
            n_patients=cfg.n_patients,
            rng=rng,
            sample_mode=cfg.sample_mode,
        )
    raise ValueError(f"unknown patient_generator.kind: {cfg.kind}")


def build_rngs(seed: int) -> dict[str, np.random.Generator]:
    """Independent substreams so swapping one module doesn't perturb others.

    ``arrival`` is deliberately separate from ``patient``: *when* someone arrives and
    *which image* they carry must vary independently, or a comparison of two arrival
    processes would also be a comparison of two different patient sequences and could
    not be interpreted. Spawning is prefix-stable, so the first four substreams are
    unchanged from before this one existed — image sampling is bit-identical, only
    arrival times move.
    """
    seed_seq = np.random.SeedSequence(seed)
    children = seed_seq.spawn(5)
    return {
        "patient": np.random.default_rng(children[0]),
        "ai": np.random.default_rng(children[1]),
        "doctor": np.random.default_rng(children[2]),
        "sort": np.random.default_rng(children[3]),
        "arrival": np.random.default_rng(children[4]),
    }


@dataclass(frozen=True)
class BuiltScenario:
    """Everything a `SimulationEngine` needs, assembled from one validated config.

    A dataclass rather than a tuple because the signal plane made the tuple long enough
    that positional unpacking became a liability — adding a seam should not silently
    shift what everyone else's third element means.
    """

    schema: LabelSchema
    generator: Iterable[tuple[float, Patient]]
    ai: DiagnosticActor
    doctor: DiagnosticActor
    sort: SortStrategy
    severity: SeverityModel | None
    n_ai_workers: int
    n_doctor_workers: int
    difficulty: DifficultyModel | None = None
    fatigue: FatigueCurve | None = None

    def __iter__(self) -> Iterator[Any]:
        """Unpack as the original 8-tuple, so existing call sites keep working."""
        return iter(
            (
                self.schema,
                self.generator,
                self.ai,
                self.doctor,
                self.sort,
                self.severity,
                self.n_ai_workers,
                self.n_doctor_workers,
            )
        )


def build_from_scenario(scenario: ScenarioConfig) -> BuiltScenario:
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
        scenario.patient_generator, base_schema, rngs["patient"], rngs["arrival"]
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
    return BuiltScenario(
        schema=active_schema,
        generator=generator,
        ai=ai,
        doctor=doctor,
        sort=sort,
        severity=severity,
        n_ai_workers=scenario.n_ai_workers,
        n_doctor_workers=scenario.n_doctor_workers,
        # Seeded from the scenario so a Beta draw is the same image's difficulty across
        # replications — the property the study layer's paired contrasts rest on.
        difficulty=build_difficulty(scenario.difficulty, seed=scenario.seed),
        fatigue=build_fatigue(scenario.fatigue),
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
